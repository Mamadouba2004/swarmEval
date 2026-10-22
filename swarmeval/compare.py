"""Run comparison (Phase 1): how a candidate run set differs from a baseline.

Both run sets are re-analysed on the tasks they share (optionally re-priced
with one pricing table) and compared metric by metric, per task, per paired
run and per agent.  Every metric keeps its own direction, so a candidate that
is cheaper but less accurate shows as a tradeoff, not as a single score.

Some coordination metrics have no better direction (potential parallelism,
Communication / Work: communication is not automatically waste); they are
reported as ``changed`` rather than improved or regressed.  Coordination
rows are ``unknown`` when either side was analysed without coordination.

Differences are measured, not tested: confidence intervals and significance
are Phase 3.  With few trials, treat small deltas as noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .evaluate import EvaluationResult, analyze_runs
from .pricing import PricingTable
from .runner import RunSet

# (key, label, higher_is_better, unit); higher_is_better None = neutral (no better direction)
METRICS: List[Tuple[str, str, Optional[bool], str]] = [
    ("success_rate", "Task Success", True, "pct"),
    ("mean_score", "Mean Score", True, "num"),
    ("run_errors", "Runs Crashed", False, "int"),
    ("cost_per_run", "Cost / Run", False, "usd"),
    ("cost_per_success", "Cost / Successful Task", False, "usd"),
    ("tokens_per_run", "Tokens / Run", False, "int"),
    ("llm_calls_per_run", "LLM Calls / Run", False, "num"),
    ("tool_calls_per_run", "Tool Calls / Run", False, "num"),
    ("messages_per_run", "Messages / Run", False, "num"),
    ("wall_clock_s", "Wall Clock / Run", False, "s"),
    ("agent_time_s", "Agent Time / Run", False, "s"),
    ("errors_per_run", "Errors / Run", False, "num"),
    # coordination (Phase 2)
    ("critical_path_s", "Critical Path / Run", False, "s"),
    ("actual_parallelism", "Actual Parallelism", True, "num"),
    ("potential_parallelism", "Potential Parallelism", None, "num"),
    ("comm_work_ratio", "Communication / Work", None, "num"),
    ("tool_redundancy", "Redundant Tool Calls", False, "pct"),
    ("artifact_redundancy", "Redundant Artifacts", False, "pct"),
    ("context_bloat", "Context Bloat", False, "num"),
    ("dead_agents_per_run", "Dead Agents / Run", False, "num"),
    ("smells_per_run", "Smells / Run", False, "num"),
]
COORDINATION_KEYS = ("critical_path_s", "actual_parallelism", "potential_parallelism", "comm_work_ratio",
                     "tool_redundancy", "artifact_redundancy", "context_bloat", "dead_agents_per_run",
                     "smells_per_run")

# Relative changes smaller than this are reported as unchanged.
TOLERANCE = 0.005


def _mean(s: Any) -> Optional[float]:
    return s.mean if s is not None and s.n else None


def _coordination_values(r: EvaluationResult) -> Dict[str, Optional[float]]:
    cs = r.coordination
    if cs is None:
        return {k: None for k in COORDINATION_KEYS}
    dep, comm, red, ctx = cs.dependency, cs.communication, cs.redundancy, cs.context
    return {
        "critical_path_s": _mean(dep.critical_path_s) if dep else None,
        "actual_parallelism": _mean(dep.actual_parallelism) if dep else None,
        "potential_parallelism": _mean(dep.potential_parallelism) if dep else None,
        "comm_work_ratio": _mean(comm.comm_work_ratio) if comm else None,
        "tool_redundancy": _mean(red.tool_redundancy) if red else None,
        "artifact_redundancy": _mean(red.artifact_redundancy) if red else None,
        "context_bloat": _mean(ctx.ratio) if ctx else None,
        "dead_agents_per_run": _mean(cs.dead_agents_per_run),
        "smells_per_run": _mean(cs.smells_per_run),
    }


def _values(r: EvaluationResult) -> Dict[str, Optional[float]]:
    eff, perf = r.efficiency, r.performance
    cost_ok = eff.cost_complete  # a partial cost is not comparable
    return {
        "success_rate": perf.success_rate,
        "mean_score": perf.mean_score,
        "run_errors": float(perf.run_errors),
        "cost_per_run": r.cost["per_run"] if cost_ok else None,
        "cost_per_success": eff.cost_per_success if cost_ok else None,
        "tokens_per_run": eff.tokens.mean,
        "llm_calls_per_run": eff.llm_calls.mean,
        "tool_calls_per_run": eff.tool_calls.mean,
        "messages_per_run": eff.messages.mean,
        "wall_clock_s": eff.wall_clock_s.mean,
        "agent_time_s": eff.agent_time_s.mean,
        "errors_per_run": eff.errors.mean,
        **_coordination_values(r),
    }


@dataclass
class MetricDelta:
    key: str
    label: str
    baseline: Optional[float]
    candidate: Optional[float]
    higher_is_better: Optional[bool]      # None = neutral: a change is neither better nor worse
    unit: str

    @property
    def delta(self) -> Optional[float]:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def relative(self) -> Optional[float]:
        d = self.delta
        if d is None or not self.baseline:
            return None
        return d / abs(self.baseline)

    @property
    def verdict(self) -> str:
        """``improved`` / ``regressed`` / ``unchanged``, ``changed`` for a
        neutral metric, or ``unknown`` when a side is missing (e.g. a cost
        that could not be priced)."""
        d = self.delta
        if d is None:
            return "unknown"
        scale = abs(self.baseline) if self.baseline else max(abs(self.candidate or 0.0), 1.0)
        if abs(d) <= TOLERANCE * scale:
            return "unchanged"
        if self.higher_is_better is None:
            return "changed"
        return "improved" if (d > 0) == self.higher_is_better else "regressed"

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "baseline": self.baseline, "candidate": self.candidate,
                "delta": self.delta, "relative": self.relative, "verdict": self.verdict,
                "higher_is_better": self.higher_is_better}


@dataclass
class Comparison:
    baseline: EvaluationResult
    candidate: EvaluationResult
    metrics: List[MetricDelta]
    per_task: Dict[str, Dict[str, Any]]
    flips: Dict[str, List[Tuple[str, int]]]          # "gained"/"lost": paired (task, trial) outcome changes
    paired_runs: int
    per_agent: Dict[str, Dict[str, Any]]
    shared_tasks: List[str]
    only_baseline: List[str] = field(default_factory=list)
    only_candidate: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def metric(self, key: str) -> MetricDelta:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    @property
    def improved(self) -> List[str]:
        return [m.key for m in self.metrics if m.verdict == "improved"]

    @property
    def regressed(self) -> List[str]:
        return [m.key for m in self.metrics if m.verdict == "regressed"]

    @property
    def is_tradeoff(self) -> bool:
        return bool(self.improved and self.regressed)

    def summary(self) -> Dict[str, Any]:
        return {"baseline": self.baseline.label, "candidate": self.candidate.label,
                "benchmark": self.baseline.benchmark, "shared_tasks": len(self.shared_tasks),
                "paired_runs": self.paired_runs, "improved": self.improved, "regressed": self.regressed,
                "gained": len(self.flips["gained"]), "lost": len(self.flips["lost"]), "warnings": self.warnings}

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "metrics": [m.to_dict() for m in self.metrics],
                "per_task": self.per_task, "flips": {k: [list(x) for x in v] for k, v in self.flips.items()},
                "per_agent": self.per_agent, "only_baseline": self.only_baseline,
                "only_candidate": self.only_candidate}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    def report(self) -> str:
        from .report.text import render_comparison
        return render_comparison(self)


def _subset(runs: RunSet, task_ids: List[str]) -> RunSet:
    keep = set(task_ids)
    return RunSet(runs.label, runs.benchmark_name, [t for t in runs.trajectories if t.task_id in keep],
                  dict(runs.metadata))


def _as_runs(x: Any) -> RunSet:
    if isinstance(x, EvaluationResult):
        return x.runs
    if isinstance(x, RunSet):
        return x
    raise TypeError(f"expected an EvaluationResult or RunSet, got {type(x).__name__}")


def compare(baseline: Any, candidate: Any, pricing: Optional[PricingTable] = None, coordination: bool = True,
            similarity: Any = None, smell_config: Any = None) -> Comparison:
    """Compare ``candidate`` against ``baseline`` (each an ``EvaluationResult``
    or ``RunSet``) on the tasks both contain.  Pass ``pricing`` to price both
    with the same table; otherwise each keeps the cost recorded at run time.
    ``coordination``, ``similarity`` and ``smell_config`` are passed to
    ``analyze_runs`` for both sides."""
    base_runs, cand_runs = _as_runs(baseline), _as_runs(candidate)
    warnings: List[str] = []
    if base_runs.benchmark_name != cand_runs.benchmark_name:
        warnings.append(f"different benchmarks: {base_runs.benchmark_name!r} vs {cand_runs.benchmark_name!r}")
    base_ids, cand_ids = base_runs.task_ids(), cand_runs.task_ids()
    shared = [t for t in base_ids if t in set(cand_ids)]
    only_b = [t for t in base_ids if t not in set(cand_ids)]
    only_c = [t for t in cand_ids if t not in set(base_ids)]
    if only_b or only_c:
        warnings.append(f"compared on {len(shared)} shared task(s); {len(only_b)} only in baseline, "
                        f"{len(only_c)} only in candidate")
    if not shared:
        raise ValueError("the two run sets have no task in common")

    kw = dict(pricing=pricing, coordination=coordination, similarity=similarity, smell_config=smell_config)
    b = analyze_runs(_subset(base_runs, shared), **kw)
    c = analyze_runs(_subset(cand_runs, shared), **kw)
    if b.performance.trials != c.performance.trials:
        warnings.append(f"different trial counts: {b.performance.trials} vs {c.performance.trials}")
    for side, r in (("baseline", b), ("candidate", c)):
        if not r.efficiency.cost_complete:
            warnings.append(f"{side} cost is incomplete (unpriced models); cost deltas are not reported")
    if b.performance.trials < 3 or c.performance.trials < 3:
        warnings.append("fewer than 3 trials per task: differences may be noise (significance testing is Phase 3)")

    bv, cv = _values(b), _values(c)
    metrics = [MetricDelta(k, label, bv[k], cv[k], hib, unit) for k, label, hib, unit in METRICS]

    per_task: Dict[str, Dict[str, Any]] = {}
    for tid in shared:
        bt, ct = b.performance.per_task[tid], c.performance.per_task[tid]
        per_task[tid] = {
            "success_rate": (bt["success_rate"], ct["success_rate"]),
            "score": (bt["score"], ct["score"]),
            "cost_usd": (bt["cost_usd"] if bt["cost_complete"] else None,
                         ct["cost_usd"] if ct["cost_complete"] else None),
            "latency_s": (bt["latency_s"], ct["latency_s"]),
        }

    # Paired runs: same task and trial.  Seeds derive from (seed, task, trial),
    # so with the same base seed both sides saw the same randomness.
    b_runs = {(t.task_id, t.trial): t for t in b.runs.trajectories}
    flips: Dict[str, List[Tuple[str, int]]] = {"gained": [], "lost": []}
    paired = 0
    seeds_differ = False
    for t in c.runs.trajectories:
        bt = b_runs.get((t.task_id, t.trial))
        if bt is None:
            continue
        paired += 1
        seeds_differ = seeds_differ or bt.seed != t.seed
        if t.succeeded and not bt.succeeded:
            flips["gained"].append((t.task_id, t.trial))
        elif bt.succeeded and not t.succeeded:
            flips["lost"].append((t.task_id, t.trial))
    if seeds_differ:
        warnings.append("paired runs used different seeds; per-run flips mix architecture and randomness")

    per_agent: Dict[str, Dict[str, Any]] = {}
    ba, ca = b.efficiency.per_agent, c.efficiency.per_agent
    for aid in list(ba) + [a for a in ca if a not in ba]:
        x, y = ba.get(aid), ca.get(aid)
        per_agent[aid] = {
            "status": "both" if x and y else ("removed" if x else "added"),
            "tokens": (x["tokens"] if x else None, y["tokens"] if y else None),
            "llm_calls": (x["llm_calls"] if x else None, y["llm_calls"] if y else None),
            "cost_usd": (x["cost_usd"] if x and x.get("cost_complete", True) else None,
                         y["cost_usd"] if y and y.get("cost_complete", True) else None),
            "self_time_s": (x["self_time_s"] if x else None, y["self_time_s"] if y else None),
        }

    return Comparison(b, c, metrics, per_task, flips, paired, per_agent, shared, only_b, only_c, warnings)
