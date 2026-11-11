"""Layer 1 -- task performance aggregated over a set of runs.

``evaluator_errors`` counts runs whose evaluator crashed, from the explicit ``evaluator_error`` flag; the
"evaluator error" reason prefix is trusted only on Phase 1 runs saved before the flag existed.  After an
interrupt, the trial cut short is left out of ``per_trial`` rather than shown as a full one.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import Trajectory
from ..stats import mean


@dataclass
class PerformanceMetrics:
    n_runs: int = 0
    n_tasks: int = 0
    trials: int = 0
    n_successes: int = 0
    success_rate: float = 0.0
    mean_score: float = 0.0
    constraint_pass_rate: float | None = None
    run_errors: int = 0
    evaluator_errors: int = 0
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_trial: list[float] = field(default_factory=list)
    evaluators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The metrics as a JSON-ready dict."""
        return {
            "n_runs": self.n_runs, "n_tasks": self.n_tasks, "trials": self.trials,
            "n_successes": self.n_successes, "success_rate": self.success_rate,
            "mean_score": self.mean_score, "constraint_pass_rate": self.constraint_pass_rate,
            "run_errors": self.run_errors, "evaluator_errors": self.evaluator_errors,
            "per_task": self.per_task, "per_trial": self.per_trial,
            "evaluators": self.evaluators,
        }


def compute_performance(trajectories: Sequence[Trajectory],
                        per_run: Sequence[Any] | None = None, interrupted: bool = False) -> PerformanceMetrics:
    """Task performance over runs; ``per_run`` (their EfficiencyMetrics, same order) supplies per-task cost.
    With ``interrupted``, the trial cut short is left out of ``per_trial``."""
    pm = PerformanceMetrics()
    pm.n_runs = len(trajectories)
    if not trajectories:
        return pm
    pm.n_successes = sum(1 for t in trajectories if t.succeeded)
    pm.success_rate = pm.n_successes / pm.n_runs
    pm.mean_score = mean([t.score for t in trajectories])
    cons = [t.evaluation.constraints_passed for t in trajectories
            if t.evaluation and t.evaluation.constraints_passed is not None]
    if cons:
        pm.constraint_pass_rate = sum(1 for c in cons if c) / len(cons)
    pm.run_errors = sum(1 for t in trajectories if t.status != "success")
    pm.evaluator_errors = sum(1 for t in trajectories if t.evaluation and (
        t.evaluation.details.get("evaluator_error")
        or (t.metadata.get("schema_version") is None
            and str(t.evaluation.details.get("reason", "")).startswith("evaluator error"))))
    if per_run is not None:
        costs = [(e.cost_usd, e.cost_complete) for e in per_run]
    else:
        costs = [(t.cost_usd, t.cost_is_complete) for t in trajectories]
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    task_costs: dict[str, list[tuple[float | None, bool]]] = defaultdict(list)
    by_trial: dict[int, list[bool]] = defaultdict(list)
    for t, c in zip(trajectories, costs, strict=True):
        by_task[t.task_id].append(t)
        task_costs[t.task_id].append(c)
        by_trial[t.trial].append(t.succeeded)
    pm.n_tasks = len(by_task)
    pm.trials = max(by_trial) + 1 if by_trial else 0
    for tid, ts in by_task.items():
        known = [c for c, _ in task_costs[tid] if c is not None]
        pm.per_task[tid] = {"success_rate": mean([1.0 if x.succeeded else 0.0 for x in ts]),
                            "score": mean([x.score for x in ts]), "n": len(ts),
                            "cost_usd": mean(known) if known else None,
                            "cost_complete": all(ok for _, ok in task_costs[tid]),
                            "latency_s": mean([x.duration for x in ts]),
                            "category": ts[0].task.category, "difficulty": ts[0].task.difficulty}
    pm.per_trial = [mean([1.0 if s else 0.0 for s in by_trial[i]]) for i in sorted(by_trial)
                    if not interrupted or len(by_trial[i]) == pm.n_tasks]
    pm.evaluators = sorted({t.evaluation.evaluator for t in trajectories if t.evaluation})
    return pm
