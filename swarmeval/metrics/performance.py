"""Layer 1 -- task performance aggregated over a set of runs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    constraint_pass_rate: Optional[float] = None
    run_errors: int = 0
    evaluator_errors: int = 0
    per_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_trial: List[float] = field(default_factory=list)
    evaluators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs, "n_tasks": self.n_tasks, "trials": self.trials,
            "n_successes": self.n_successes, "success_rate": self.success_rate,
            "mean_score": self.mean_score, "constraint_pass_rate": self.constraint_pass_rate,
            "run_errors": self.run_errors, "evaluator_errors": self.evaluator_errors,
            "per_task": self.per_task, "per_trial": self.per_trial,
            "evaluators": self.evaluators,
        }


def compute_performance(trajectories: Sequence[Trajectory],
                        per_run: Optional[Sequence[Any]] = None) -> PerformanceMetrics:
    """Task performance over a set of runs.  ``per_run`` (the runs'
    ``EfficiencyMetrics``, same order) supplies per-task cost; without it the
    cost recorded on each trajectory is used."""
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
    pm.evaluator_errors = sum(1 for t in trajectories if t.evaluation
                              and str(t.evaluation.details.get("reason", "")).startswith("evaluator error"))
    if per_run is not None:
        costs = [(e.cost_usd, e.cost_complete) for e in per_run]
    else:
        costs = [(t.cost_usd, t.cost_is_complete) for t in trajectories]
    by_task: Dict[str, List[Trajectory]] = defaultdict(list)
    task_costs: Dict[str, List[Tuple[Optional[float], bool]]] = defaultdict(list)
    by_trial: Dict[int, List[bool]] = defaultdict(list)
    for t, c in zip(trajectories, costs):
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
    pm.per_trial = [mean([1.0 if s else 0.0 for s in by_trial[i]]) for i in sorted(by_trial)]
    pm.evaluators = sorted({t.evaluation.evaluator for t in trajectories if t.evaluation})
    return pm
