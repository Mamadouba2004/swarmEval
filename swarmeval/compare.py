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

METRICS rows are (key, label, higher_is_better, unit); higher_is_better None marks a neutral metric. Runs are
paired by task and trial; seeds derive from (seed, task, trial), so with the same base seed both sides saw the
same randomness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .evaluate import EvaluationResult, analyze_runs
from .metrics.coordination import ANALYSES
from .pricing import PricingTable
from .runner import RunSet

METRICS: list[tuple[str, str, bool | None, str]] = [
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
    *(m[:4] for a in ANALYSES for m in a.metrics),
    ("dead_agents_per_run", "Dead Agents / Run", False, "num"),
    ("smells_per_run", "Smells / Run", False, "num"),
]
COORDINATION_KEYS = (*(m[0] for a in ANALYSES for m in a.metrics), "dead_agents_per_run", "smells_per_run")

TOLERANCE = 0.005


def _mean(s: Any) -> float | None:
    """Mean of a summary, or None when it is missing or empty."""
    return s.mean if s is not None and s.n else None


def _coordination_values(r: EvaluationResult) -> dict[str, float | None]:
    """Coordination metric values of a result, all None when coordination was not computed."""
    cs = r.coordination
    if cs is None:
        return {k: None for k in COORDINATION_KEYS}
    out: dict[str, float | None] = {}
    for a in ANALYSES:
        s = cs.get(a.name)
        for key, *_, summary_field in a.metrics:
            out[key] = _mean(getattr(s, summary_field)) if s is not None else None
    return {
        **out,
        "dead_agents_per_run": _mean(cs.dead_agents_per_run),
        "smells_per_run": _mean(cs.smells_per_run),
    }


def _values(r: EvaluationResult) -> dict[str, float | None]:
    """Every compared metric's value for one result; partial costs are None, not comparable."""
    eff, perf = r.efficiency, r.performance
    cost_ok = eff.cost_complete
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
    baseline: float | None
    candidate: float | None
    higher_is_better: bool | None
    unit: str

    @property
    def delta(self) -> float | None:
        """Candidate minus baseline, None when either side is missing."""
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def relative(self) -> float | None:
        """Delta relative to the baseline, None when undefined."""
        d = self.delta
        if d is None or not self.baseline:
            return None
        return d / abs(self.baseline)

    @property
    def verdict(self) -> str:
        """``improved``, ``regressed``, ``unchanged``, ``changed`` for a neutral metric,
        or ``unknown`` when a side is missing (e.g. an unpriced cost)."""
        d = self.delta
        if d is None:
            return "unknown"
        scale = abs(self.baseline) if self.baseline else max(abs(self.candidate or 0.0), 1.0)
        if abs(d) <= TOLERANCE * scale:
            return "unchanged"
        if self.higher_is_better is None:
            return "changed"
        return "improved" if (d > 0) == self.higher_is_better else "regressed"

    def to_dict(self) -> dict[str, Any]:
        """The delta as a JSON-ready dict."""
        return {"key": self.key, "label": self.label, "baseline": self.baseline, "candidate": self.candidate,
                "delta": self.delta, "relative": self.relative, "verdict": self.verdict,
                "higher_is_better": self.higher_is_better}


@dataclass
class Comparison:
    baseline: EvaluationResult
    candidate: EvaluationResult
    metrics: list[MetricDelta]
    per_task: dict[str, dict[str, Any]]
    flips: dict[str, list[tuple[str, int]]]
    paired_runs: int
    per_agent: dict[str, dict[str, Any]]
    shared_tasks: list[str]
    only_baseline: list[str] = field(default_factory=list)
    only_candidate: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def metric(self, key: str) -> MetricDelta:
        """The MetricDelta for ``key``; KeyError if absent."""
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    @property
    def improved(self) -> list[str]:
        """Keys of the metrics that improved."""
        return [m.key for m in self.metrics if m.verdict == "improved"]

    @property
    def regressed(self) -> list[str]:
        """Keys of the metrics that regressed."""
        return [m.key for m in self.metrics if m.verdict == "regressed"]

    @property
    def is_tradeoff(self) -> bool:
        """True if some metrics improved and others regressed."""
        return bool(self.improved and self.regressed)

    def summary(self) -> dict[str, Any]:
        """Headline counts and verdicts of the comparison."""
        return {"baseline": self.baseline.label, "candidate": self.candidate.label,
                "benchmark": self.baseline.benchmark, "shared_tasks": len(self.shared_tasks),
                "paired_runs": self.paired_runs, "improved": self.improved, "regressed": self.regressed,
                "gained": len(self.flips["gained"]), "lost": len(self.flips["lost"]), "warnings": self.warnings}

    def to_dict(self) -> dict[str, Any]:
        """The comparison as a JSON-ready dict."""
        return {"summary": self.summary(), "metrics": [m.to_dict() for m in self.metrics],
                "per_task": self.per_task, "flips": {k: [list(x) for x in v] for k, v in self.flips.items()},
                "per_agent": self.per_agent, "only_baseline": self.only_baseline,
                "only_candidate": self.only_candidate}

    def save(self, path: str) -> None:
        """Write the comparison as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    def report(self) -> str:
        """The comparison as a text report."""
        from .report.text import render_comparison
        return render_comparison(self)


def _subset(runs: RunSet, task_ids: list[str]) -> RunSet:
    """A copy of ``runs`` keeping only the given tasks."""
    keep = set(task_ids)
    return RunSet(runs.label, runs.benchmark_name, [t for t in runs.trajectories if t.task_id in keep],
                  dict(runs.metadata))


def _as_runs(x: Any) -> RunSet:
    """The RunSet of an EvaluationResult, or the RunSet itself."""
    if isinstance(x, EvaluationResult):
        return x.runs
    if isinstance(x, RunSet):
        return x
    raise TypeError(f"expected an EvaluationResult or RunSet, got {type(x).__name__}")


def _warnings(base_runs: RunSet, cand_runs: RunSet, b: EvaluationResult, c: EvaluationResult, n_shared: int,
              only_b: list[str], only_c: list[str], seeds_differ: bool) -> list[str]:
    """What makes the comparison less than like for like, in report order."""
    warnings: list[str] = []
    if base_runs.benchmark_name != cand_runs.benchmark_name:
        warnings.append(f"different benchmarks: {base_runs.benchmark_name!r} vs {cand_runs.benchmark_name!r}")
    if only_b or only_c:
        warnings.append(f"compared on {n_shared} shared task(s); {len(only_b)} only in baseline, "
                        f"{len(only_c)} only in candidate")
    for side, rs in (("baseline", base_runs), ("candidate", cand_runs)):
        if rs.metadata.get("interrupted"):
            warnings.append(f"{side} was interrupted after {len(rs)}/{rs.metadata.get('planned_runs', len(rs))} "
                            "runs; it is not a full run set")
    if b.performance.trials != c.performance.trials:
        warnings.append(f"different trial counts: {b.performance.trials} vs {c.performance.trials}")
    for side, r in (("baseline", b), ("candidate", c)):
        if not r.efficiency.cost_complete:
            warnings.append(f"{side} cost is incomplete (unpriced models); cost deltas are not reported")
    if b.performance.trials < 3 or c.performance.trials < 3:
        warnings.append("fewer than 3 trials per task: differences may be noise (significance testing is Phase 3)")
    if seeds_differ:
        warnings.append("paired runs used different seeds; per-run flips mix architecture and randomness")
    return warnings


def _flips(b: EvaluationResult, c: EvaluationResult) -> tuple[dict[str, list[tuple[str, int]]], int, bool]:
    """Runs paired by task and trial: the success flips, the number of pairs and whether any pair's seeds differ."""
    b_runs = {(t.task_id, t.trial): t for t in b.runs.trajectories}
    flips: dict[str, list[tuple[str, int]]] = {"gained": [], "lost": []}
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
    return flips, paired, seeds_differ


def _per_agent(b: EvaluationResult, c: EvaluationResult) -> dict[str, dict[str, Any]]:
    """Each agent's tokens, calls, cost and self time on both sides, and whether it was added or removed."""
    per_agent: dict[str, dict[str, Any]] = {}
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
    return per_agent


def compare(baseline: Any, candidate: Any, pricing: PricingTable | None = None, coordination: bool = True,
            similarity: Any = None, smell_config: Any = None) -> Comparison:
    """Compare ``candidate`` with ``baseline`` (EvaluationResult or RunSet) on shared tasks, priced alike
    with ``pricing``; ``coordination``, ``similarity`` and ``smell_config`` go to ``analyze_runs`` for both."""
    base_runs, cand_runs = _as_runs(baseline), _as_runs(candidate)
    base_ids, cand_ids = base_runs.task_ids(), cand_runs.task_ids()
    shared = [t for t in base_ids if t in set(cand_ids)]
    only_b = [t for t in base_ids if t not in set(cand_ids)]
    only_c = [t for t in cand_ids if t not in set(base_ids)]
    if not shared:
        raise ValueError("the two run sets have no task in common")

    kw = dict(pricing=pricing, coordination=coordination, similarity=similarity, smell_config=smell_config)
    b = analyze_runs(_subset(base_runs, shared), **kw)
    c = analyze_runs(_subset(cand_runs, shared), **kw)
    bv, cv = _values(b), _values(c)
    metrics = [MetricDelta(k, label, bv[k], cv[k], hib, unit) for k, label, hib, unit in METRICS]

    per_task: dict[str, dict[str, Any]] = {}
    for tid in shared:
        bt, ct = b.performance.per_task[tid], c.performance.per_task[tid]
        per_task[tid] = {
            "success_rate": (bt["success_rate"], ct["success_rate"]),
            "score": (bt["score"], ct["score"]),
            "cost_usd": (bt["cost_usd"] if bt["cost_complete"] else None,
                         ct["cost_usd"] if ct["cost_complete"] else None),
            "latency_s": (bt["latency_s"], ct["latency_s"]),
        }

    flips, paired, seeds_differ = _flips(b, c)
    warnings = _warnings(base_runs, cand_runs, b, c, len(shared), only_b, only_c, seeds_differ)
    return Comparison(b, c, metrics, per_task, flips, paired, _per_agent(b, c), shared, only_b, only_c, warnings)
