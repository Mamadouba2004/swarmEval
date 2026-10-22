"""Layer 3 -- diagnostic engine: smell prevalence over a set of runs (Phase 2).

One run is one observation.  A smell found several times in a run (two
dead agents, two bloated agents) counts once for that run; the agents it
involved are counted once per run too.  Smells are grouped by name only, so
the report reads "dead_agent in 8/8 runs (researcher_c 8)" rather than one
line per agent.  The representative observation / hypothesis / intervention
is taken from the most severe occurrence (ties: the earliest run).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .smells import SEVERITIES, SEVERITY_RANK, Diagnostic

MAX_EXAMPLES = 3


@dataclass
class SmellPrevalence:
    """How often one smell was seen.  ``prevalence`` = runs_affected ÷ n_runs;
    ``severity_counts`` counts runs by their most severe occurrence;
    ``agents`` counts, per agent, the runs in which the smell involved it;
    ``inferred_runs`` the runs where every occurrence rested on inferred
    evidence; ``example_runs`` up to three run ids, most severe first."""
    smell: str
    runs_affected: int
    n_runs: int
    prevalence: float
    severity_counts: Dict[str, int]
    max_severity: str
    agents: Dict[str, int]
    example_runs: List[str]
    observation: str
    hypothesis: str
    intervention: str
    inferred_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticsReport:
    """Smell prevalence over runs, sorted by most severe occurrence, then
    prevalence, then name.  ``total`` counts diagnostics before the per-run
    deduplication."""
    n_runs: int
    smells: List[SmellPrevalence] = field(default_factory=list)
    total: int = 0

    def get(self, smell: str) -> SmellPrevalence:
        for s in self.smells:
            if s.smell == smell:
                return s
        raise KeyError(smell)

    @property
    def names(self) -> List[str]:
        return [s.smell for s in self.smells]

    def to_dict(self) -> Dict[str, Any]:
        return {"n_runs": self.n_runs, "total": self.total, "smells": [s.to_dict() for s in self.smells]}


def summarize_diagnostics(per_run: Union[Mapping[str, Sequence[Diagnostic]], Sequence[Sequence[Diagnostic]]],
                          n_runs: Optional[int] = None) -> DiagnosticsReport:
    """``per_run`` maps run id → that run's diagnostics, or is a list with one
    diagnostics list per run.  Include runs without smells (as empty lists)
    or pass ``n_runs``: they are part of the prevalence denominator."""
    if isinstance(per_run, Mapping):
        runs: List[Tuple[str, Sequence[Diagnostic]]] = list(per_run.items())
    else:
        runs = [(str(i), ds) for i, ds in enumerate(per_run)]
    n = n_runs if n_runs is not None else len(runs)
    by_smell: Dict[str, Dict[int, List[Diagnostic]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for i, (_, ds) in enumerate(runs):
        for d in ds:
            by_smell[d.smell][i].append(d)
            total += 1
    out: List[SmellPrevalence] = []
    for smell, hits in by_smell.items():
        sev_counts: Counter = Counter()
        agents: Counter = Counter()
        inferred = 0
        ranked: List[Tuple[int, int, Diagnostic]] = []      # (-severity, run index, most severe diagnostic)
        for i in sorted(hits):
            ds = hits[i]
            top = max(ds, key=lambda d: SEVERITY_RANK[d.severity])   # max keeps the first on ties
            sev_counts[top.severity] += 1
            agents.update({a for d in ds for a in d.agents})        # a set: once per run
            inferred += int(all(d.inferred for d in ds))
            ranked.append((-SEVERITY_RANK[top.severity], i, top))
        ranked.sort(key=lambda x: (x[0], x[1]))
        rep = ranked[0][2]
        out.append(SmellPrevalence(
            smell=smell, runs_affected=len(hits), n_runs=n, prevalence=len(hits) / n if n else 0.0,
            severity_counts={s: sev_counts[s] for s in SEVERITIES if sev_counts[s]},
            max_severity=SEVERITIES[-ranked[0][0]],
            agents=dict(sorted(agents.items(), key=lambda kv: (-kv[1], kv[0]))),
            example_runs=[runs[i][0] for _, i, _ in ranked[:MAX_EXAMPLES]],
            observation=rep.observation, hypothesis=rep.hypothesis, intervention=rep.intervention,
            inferred_runs=inferred))
    out.sort(key=lambda s: (-SEVERITY_RANK[s.max_severity], -s.prevalence, s.smell))
    return DiagnosticsReport(n_runs=n, smells=out, total=total)


def diagnostics_by_run(runs: Iterable[Any]) -> Dict[str, List[Diagnostic]]:
    """run id → diagnostics, from ``CoordinationMetrics`` objects."""
    return {cm.run_id: list(cm.diagnostics or []) for cm in runs}
