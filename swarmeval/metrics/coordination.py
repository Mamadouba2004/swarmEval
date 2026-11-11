"""Layer 3 -- coordination quality of one run, and its summary over runs (Phase 2).

``compute_coordination`` is the single entry point: it builds the execution
graph once and hands it to each analysis.  PR A fills in provenance coverage,
utilization and observed contribution; communication, redundancy, dependency
(critical path, parallelism) and context analyses are added as sibling
modules and attach to the same ``CoordinationMetrics`` (``None`` until then).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..graph.builder import Coverage, ExecutionGraph, build_graph
from ..schema import Trajectory
from ..stats import Summary, mean, summarize
from .contribution import (AgentUtilization, Lineage, ObservedContribution, compute_contribution,
                           compute_utilization, final_lineage, summarize_contribution)
from .efficiency import EfficiencyMetrics


def _dict(x: Any) -> Any:
    return x.to_dict() if hasattr(x, "to_dict") else x


@dataclass
class CoordinationMetrics:
    """Coordination analysis of one run.  ``graph`` is kept for later
    analyses and exports but left out of ``to_dict`` (export it separately)."""
    run_id: str
    coverage: Coverage
    utilization: Dict[str, AgentUtilization]
    contribution: Dict[str, ObservedContribution]
    lineage: Lineage = field(default_factory=Lineage)
    communication: Any = None
    redundancy: Any = None
    dependency: Any = None
    context: Any = None
    graph: Optional[ExecutionGraph] = field(default=None, repr=False, compare=False)

    @property
    def dead_agents(self) -> List[str]:
        """Agents whose ``dead`` is True (complete provenance, nothing used)."""
        return [a for a, c in self.contribution.items() if c.dead is True]

    @property
    def lineage_agents(self) -> List[str]:
        return [a for a, c in self.contribution.items() if c.in_final_lineage]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "coverage": self.coverage.to_dict(),
            "utilization": {k: v.to_dict() for k, v in self.utilization.items()},
            "contribution": {k: v.to_dict() for k, v in self.contribution.items()},
            "lineage": self.lineage.to_dict(), "lineage_agents": self.lineage_agents,
            "dead_agents": self.dead_agents,
            "communication": _dict(self.communication), "redundancy": _dict(self.redundancy),
            "dependency": _dict(self.dependency), "context": _dict(self.context),
        }


def compute_coordination(traj: Trajectory, graph: Optional[ExecutionGraph] = None, similarity: Any = None,
                         smell_config: Any = None,
                         efficiency: Optional[EfficiencyMetrics] = None) -> CoordinationMetrics:
    """Coordination metrics of one run.  ``similarity`` and ``smell_config``
    are accepted for the redundancy and diagnostics analyses (not used yet);
    ``efficiency`` avoids recomputing the run's efficiency metrics."""
    graph = graph or build_graph(traj)
    lineage = final_lineage(traj)
    return CoordinationMetrics(
        run_id=traj.run_id, coverage=graph.coverage,
        utilization=compute_utilization(traj, graph, efficiency),
        contribution=compute_contribution(traj, graph, lineage), lineage=lineage, graph=graph)


@dataclass
class CoordinationSummary:
    """Coordination metrics over a set of runs.  Coverage summaries are over
    the runs where the ratio is defined (``n`` says how many); ``None`` when
    no run had anything to measure."""
    n_runs: int
    message_receive_coverage: Optional[Summary]
    derivation_coverage: Optional[Summary]
    dead_agents_per_run: Summary
    utilization: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contribution: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "message_receive_coverage": _dict(self.message_receive_coverage),
            "derivation_coverage": _dict(self.derivation_coverage),
            "dead_agents_per_run": self.dead_agents_per_run.to_dict(),
            "utilization": self.utilization, "contribution": self.contribution,
        }


def _known(values: Sequence[Optional[float]]) -> Optional[Summary]:
    known = [v for v in values if v is not None]
    return summarize(known) if known else None


def summarize_coordination(per_run: Sequence[CoordinationMetrics]) -> CoordinationSummary:
    util: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for cm in per_run:
        for aid, u in cm.utilization.items():
            for k in ("token_share", "llm_call_share", "tool_call_share", "busy_share"):
                v = getattr(u, k)
                if v is not None:
                    util[aid][k].append(v)
            for k in ("messages_sent", "artifacts_produced", "downstream_refs"):
                util[aid][k].append(float(getattr(u, k)))
    utilization = {aid: {k: mean(v) for k, v in d.items()} for aid, d in util.items()}
    return CoordinationSummary(
        n_runs=len(per_run),
        message_receive_coverage=_known([cm.coverage.message_receive_coverage for cm in per_run]),
        derivation_coverage=_known([cm.coverage.derivation_coverage for cm in per_run]),
        dead_agents_per_run=summarize([len(cm.dead_agents) for cm in per_run]),
        utilization=utilization,
        contribution=summarize_contribution([cm.contribution for cm in per_run]))
