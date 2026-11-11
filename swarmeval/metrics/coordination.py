"""Layer 3 -- coordination quality of one run, and its summary over runs (Phase 2).

``compute_coordination`` is the single entry point: it builds the execution
graph once and hands it to each analysis -- provenance coverage, utilization,
observed contribution, communication overhead, redundancy, dependencies
(critical path, parallelism) and context bloat -- and finally runs the smell
detectors (``diagnostics``) over the result.  Communication, redundancy,
dependencies and context bloat run from the ``ANALYSES`` registry, which
also gives each its compare metrics and report rows; an analysis outside
the four built-ins lands in ``extra``.

Classes:
- AnalysisInput: What every registry analysis receives for one run (``similarity`` already resolved).
- CoordinationAnalysis: One registry analysis: per-run ``compute``, ``summarize`` over runs, its compare ``metrics``
  ``(key, label, higher_is_better, unit, summary_field)``, its report ``rows`` (no separators) and the report
  ``group`` whose rows it shares a block with (default: its own block).
- CoordinationMetrics: Coordination analysis of one run. ``graph`` is kept for later analyses and exports but left
  out of ``to_dict`` (export it separately).
- CoordinationSummary: Coordination metrics over a set of runs. Coverage summaries are over the runs where the ratio
  is defined (``n`` says how many); ``None`` when no run had anything to measure.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..graph.builder import Coverage, ExecutionGraph, build_graph
from ..schema import Trajectory
from ..similarity import resolve_similarity
from ..stats import Summary, known_summary, mean, summarize
from .communication import (
    COMMUNICATION_METRICS,
    CommunicationMetrics,
    CommunicationSummary,
    communication_rows,
    compute_communication,
    summarize_communication,
)
from .context import CONTEXT_METRICS, ContextMetrics, ContextSummary, compute_context, context_rows, summarize_context
from .contribution import (
    AgentUtilization,
    Lineage,
    ObservedContribution,
    compute_contribution,
    compute_utilization,
    final_lineage,
    summarize_contribution,
)
from .dependency import (
    DEPENDENCY_METRICS,
    DependencyMetrics,
    DependencySummary,
    compute_dependency,
    dependency_rows,
    summarize_dependency,
)
from .efficiency import EfficiencyMetrics
from .redundancy import (
    REDUNDANCY_METRICS,
    RedundancyMetrics,
    RedundancySummary,
    compute_redundancy,
    redundancy_rows,
    summarize_redundancy,
)

if TYPE_CHECKING:
    from ..diagnostics.engine import DiagnosticsReport
    from ..diagnostics.smells import Diagnostic


def _dict(x: Any) -> Any:
    """``x.to_dict()`` when available, else ``x``."""
    return x.to_dict() if hasattr(x, "to_dict") else x


@dataclass(frozen=True)
class AnalysisInput:
    traj: Trajectory
    graph: ExecutionGraph
    efficiency: EfficiencyMetrics | None
    similarity: Any


@dataclass(frozen=True)
class CoordinationAnalysis:
    name: str
    compute: Callable[[AnalysisInput], Any]
    summarize: Callable[[list[Any]], Any]
    metrics: tuple[tuple[str, str, bool | None, str, str], ...]
    rows: Callable[[Any], list[tuple[str, str]]]
    group: str = ""


ANALYSES = (
    CoordinationAnalysis("dependency", lambda i: compute_dependency(i.traj, i.graph, i.efficiency),
                         summarize_dependency, DEPENDENCY_METRICS, dependency_rows),
    CoordinationAnalysis("communication", lambda i: compute_communication(i.traj, i.graph),
                         summarize_communication, COMMUNICATION_METRICS, communication_rows, "overhead"),
    CoordinationAnalysis("redundancy", lambda i: compute_redundancy(i.traj, i.graph, i.similarity),
                         summarize_redundancy, REDUNDANCY_METRICS, redundancy_rows, "overhead"),
    CoordinationAnalysis("context", lambda i: compute_context(i.traj), summarize_context, CONTEXT_METRICS,
                         context_rows, "overhead"),
)
_TYPED = frozenset({"dependency", "communication", "redundancy", "context"})


class _Analyses:
    extra: dict[str, Any]

    def get(self, name: str) -> Any:
        """The result of analysis ``name``: its typed field for a built-in, else from ``extra``."""
        return getattr(self, name) if name in _TYPED else self.extra.get(name)

    def _put(self, name: str, value: Any) -> None:
        """Store the result of analysis ``name`` where ``get`` finds it."""
        if name in _TYPED:
            setattr(self, name, value)
        else:
            self.extra[name] = value

    def _extra_dict(self) -> dict[str, Any]:
        """``{"extra": ...}`` when there are extra results, else nothing, so the default JSON is unchanged."""
        return {"extra": {k: _dict(v) for k, v in self.extra.items()}} if self.extra else {}


@dataclass
class CoordinationMetrics(_Analyses):
    run_id: str
    coverage: Coverage
    utilization: dict[str, AgentUtilization]
    contribution: dict[str, ObservedContribution]
    lineage: Lineage = field(default_factory=Lineage)
    communication: CommunicationMetrics | None = None
    redundancy: RedundancyMetrics | None = None
    dependency: DependencyMetrics | None = None
    context: ContextMetrics | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    graph: ExecutionGraph | None = field(default=None, repr=False, compare=False)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dead_agents(self) -> list[str]:
        """Agents whose ``dead`` is True (complete provenance, nothing used)."""
        return [a for a, c in self.contribution.items() if c.dead is True]

    @property
    def lineage_agents(self) -> list[str]:
        """Agents with an artifact in the final output's lineage."""
        return [a for a, c in self.contribution.items() if c.in_final_lineage]

    def to_dict(self) -> dict[str, Any]:
        """The metrics as a JSON-ready dict, without the graph."""
        return {
            "run_id": self.run_id, "coverage": self.coverage.to_dict(),
            "utilization": {k: v.to_dict() for k, v in self.utilization.items()},
            "contribution": {k: v.to_dict() for k, v in self.contribution.items()},
            "lineage": self.lineage.to_dict(), "lineage_agents": self.lineage_agents,
            "dead_agents": self.dead_agents,
            "communication": _dict(self.communication), "redundancy": _dict(self.redundancy),
            "dependency": _dict(self.dependency), "context": _dict(self.context),
            "diagnostics": [d.to_dict() for d in self.diagnostics], **self._extra_dict(),
        }


def compute_coordination(traj: Trajectory, graph: ExecutionGraph | None = None, similarity: Any = None,
                         smell_config: Any = None,
                         efficiency: EfficiencyMetrics | None = None,
                         analyses: Sequence[CoordinationAnalysis] = ANALYSES) -> CoordinationMetrics:
    """Coordination metrics of one run (``similarity``, ``smell_config`` as their resolvers take). An analysis passed
    only via ``analyses`` reaches the JSON (``extra``), not the text/HTML rows or compare: add it to ANALYSES."""
    from ..diagnostics.smells import detect_smells, resolve_smell_config
    graph = graph or build_graph(traj)
    lineage = final_lineage(traj)
    cm = CoordinationMetrics(
        run_id=traj.run_id, coverage=graph.coverage,
        utilization=compute_utilization(traj, graph, efficiency),
        contribution=compute_contribution(traj, graph, lineage), lineage=lineage, graph=graph)
    inp = AnalysisInput(traj, graph, efficiency, resolve_similarity(similarity))
    for a in analyses:
        cm._put(a.name, a.compute(inp))
    cm.diagnostics = detect_smells(traj, cm, resolve_smell_config(smell_config))
    return cm


@dataclass
class CoordinationSummary(_Analyses):
    n_runs: int
    message_receive_coverage: Summary | None
    derivation_coverage: Summary | None
    dead_agents_per_run: Summary
    utilization: dict[str, dict[str, Any]] = field(default_factory=dict)
    contribution: dict[str, dict[str, Any]] = field(default_factory=dict)
    communication: CommunicationSummary | None = None
    redundancy: RedundancySummary | None = None
    dependency: DependencySummary | None = None
    context: ContextSummary | None = None
    diagnostics: DiagnosticsReport | None = None
    smells_per_run: Summary | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dict."""
        return {
            "n_runs": self.n_runs,
            "message_receive_coverage": _dict(self.message_receive_coverage),
            "derivation_coverage": _dict(self.derivation_coverage),
            "dead_agents_per_run": self.dead_agents_per_run.to_dict(),
            "utilization": self.utilization, "contribution": self.contribution,
            "communication": _dict(self.communication), "redundancy": _dict(self.redundancy),
            "dependency": _dict(self.dependency), "context": _dict(self.context),
            "diagnostics": _dict(self.diagnostics), "smells_per_run": _dict(self.smells_per_run),
            **self._extra_dict(),
        }



def summarize_coordination(per_run: Sequence[CoordinationMetrics],
                           analyses: Sequence[CoordinationAnalysis] = ANALYSES) -> CoordinationSummary:
    """Coordination metrics over runs, summarizing each analysis the runs have."""
    from ..diagnostics.engine import summarize_diagnostics
    util: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cm in per_run:
        for aid, u in cm.utilization.items():
            for k in ("token_share", "llm_call_share", "tool_call_share", "busy_share"):
                v = getattr(u, k)
                if v is not None:
                    util[aid][k].append(v)
            for k in ("messages_sent", "artifacts_produced", "downstream_refs"):
                util[aid][k].append(float(getattr(u, k)))
    utilization = {aid: {k: mean(v) for k, v in d.items()} for aid, d in util.items()}
    summary = CoordinationSummary(
        n_runs=len(per_run),
        message_receive_coverage=known_summary([cm.coverage.message_receive_coverage for cm in per_run]),
        derivation_coverage=known_summary([cm.coverage.derivation_coverage for cm in per_run]),
        dead_agents_per_run=summarize([len(cm.dead_agents) for cm in per_run]),
        utilization=utilization,
        contribution=summarize_contribution([cm.contribution for cm in per_run]),
        diagnostics=summarize_diagnostics([list(cm.diagnostics or []) for cm in per_run]),
        smells_per_run=summarize([len({d.smell for d in cm.diagnostics}) for cm in per_run]))
    for a in analyses:
        got = [r for cm in per_run if (r := cm.get(a.name)) is not None]
        summary._put(a.name, a.summarize(got) if got else None)
    return summary
