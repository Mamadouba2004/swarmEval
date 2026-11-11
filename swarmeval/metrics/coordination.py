"""Layer 3 -- coordination quality of one run, and its summary over runs (Phase 2).

``compute_coordination`` is the single entry point: it builds the execution
graph once and hands it to each analysis -- provenance coverage, utilization,
observed contribution, communication overhead, redundancy, dependencies
(critical path, parallelism) and context bloat -- and finally runs the smell
detectors (``diagnostics``) over the result.

Classes:
- CoordinationMetrics: Coordination analysis of one run. ``graph`` is kept for later analyses and exports but left
  out of ``to_dict`` (export it separately).
- CoordinationSummary: Coordination metrics over a set of runs. Coverage summaries are over the runs where the ratio
  is defined (``n`` says how many); ``None`` when no run had anything to measure.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..graph.builder import Coverage, ExecutionGraph, build_graph
from ..schema import Trajectory
from ..similarity import resolve_similarity
from ..stats import Summary, known_summary, mean, summarize
from .communication import CommunicationMetrics, CommunicationSummary, compute_communication, summarize_communication
from .context import ContextMetrics, ContextSummary, compute_context, summarize_context
from .contribution import (
    AgentUtilization,
    Lineage,
    ObservedContribution,
    compute_contribution,
    compute_utilization,
    final_lineage,
    summarize_contribution,
)
from .dependency import DependencyMetrics, DependencySummary, compute_dependency, summarize_dependency
from .efficiency import EfficiencyMetrics
from .redundancy import RedundancyMetrics, RedundancySummary, compute_redundancy, summarize_redundancy

if TYPE_CHECKING:
    from ..diagnostics.engine import DiagnosticsReport
    from ..diagnostics.smells import Diagnostic


def _dict(x: Any) -> Any:
    """``x.to_dict()`` when available, else ``x``."""
    return x.to_dict() if hasattr(x, "to_dict") else x


@dataclass
class CoordinationMetrics:
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
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def compute_coordination(traj: Trajectory, graph: ExecutionGraph | None = None, similarity: Any = None,
                         smell_config: Any = None,
                         efficiency: EfficiencyMetrics | None = None) -> CoordinationMetrics:
    """Coordination metrics of one run; ``similarity`` and ``smell_config`` accept what their resolvers do.
    Pass a resolved ``similarity`` and the run's ``efficiency`` to avoid recomputing them over many runs."""
    from ..diagnostics.smells import detect_smells, resolve_smell_config
    graph = graph or build_graph(traj)
    lineage = final_lineage(traj)
    cm = CoordinationMetrics(
        run_id=traj.run_id, coverage=graph.coverage,
        utilization=compute_utilization(traj, graph, efficiency),
        contribution=compute_contribution(traj, graph, lineage), lineage=lineage,
        communication=compute_communication(traj, graph),
        redundancy=compute_redundancy(traj, graph, resolve_similarity(similarity)),
        dependency=compute_dependency(traj, graph, efficiency), context=compute_context(traj), graph=graph)
    cm.diagnostics = detect_smells(traj, cm, resolve_smell_config(smell_config))
    return cm


@dataclass
class CoordinationSummary:
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
        }



def summarize_coordination(per_run: Sequence[CoordinationMetrics]) -> CoordinationSummary:
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
    comm = [cm.communication for cm in per_run if cm.communication is not None]
    red = [cm.redundancy for cm in per_run if cm.redundancy is not None]
    dep = [cm.dependency for cm in per_run if cm.dependency is not None]
    ctx = [cm.context for cm in per_run if cm.context is not None]
    return CoordinationSummary(
        n_runs=len(per_run),
        message_receive_coverage=known_summary([cm.coverage.message_receive_coverage for cm in per_run]),
        derivation_coverage=known_summary([cm.coverage.derivation_coverage for cm in per_run]),
        dead_agents_per_run=summarize([len(cm.dead_agents) for cm in per_run]),
        utilization=utilization,
        contribution=summarize_contribution([cm.contribution for cm in per_run]),
        communication=summarize_communication(comm) if comm else None,
        redundancy=summarize_redundancy(red) if red else None,
        dependency=summarize_dependency(dep) if dep else None,
        context=summarize_context(ctx) if ctx else None,
        diagnostics=summarize_diagnostics([list(cm.diagnostics or []) for cm in per_run]),
        smells_per_run=summarize([len({d.smell for d in cm.diagnostics}) for cm in per_run]))
