"""Top-level evaluation: run a swarm on a benchmark and summarise the result."""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .metrics.coordination import (CoordinationMetrics, CoordinationSummary, compute_coordination,
                                   summarize_coordination)
from .metrics.efficiency import EfficiencyMetrics, compute_efficiency
from .metrics.performance import PerformanceMetrics, compute_performance
from .pricing import PricingTable
from .recorder import Clock
from .runner import RunOptions, RunSet, SwarmRunner
from .schema import Benchmark, Trajectory
from .diagnostics.smells import resolve_smell_config
from .similarity import resolve_similarity
from .stats import Summary, mean, summarize


@dataclass
class EfficiencySummary:
    """Efficiency metrics aggregated over all runs in a RunSet."""
    tokens: Summary
    input_tokens: Summary
    output_tokens: Summary
    llm_calls: Summary
    tool_calls: Summary
    messages: Summary
    cost_usd: Summary                 # over runs whose cost is known (n says how many)
    cost_complete: bool
    cost_total: Optional[float]       # None when no run could be priced
    cost_per_success: Optional[float]
    wall_clock_s: Summary
    agent_time_s: Summary
    waiting_time_s: Summary
    errors: Summary
    per_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}


def summarize_efficiency(trajectories: Sequence[Trajectory], effs: Sequence[EfficiencyMetrics]) -> EfficiencySummary:
    S = summarize
    # Unknown costs stay unknown: they are left out of sums and flagged via
    # cost_complete, never counted as $0.
    cost_complete = all(e.cost_complete for e in effs)
    costs = [e.cost_usd for e in effs if e.cost_usd is not None]
    total_cost: Optional[float] = sum(costs) if costs else None
    successes = sum(1 for t in trajectories if t.succeeded)
    per_agent: Dict[str, Dict[str, Any]] = {}
    agent_runs: Dict[str, int] = Counter()
    roles: Dict[str, Optional[str]] = {}
    agent_cost_complete: Dict[str, bool] = {}
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in effs:
        for aid, ae in e.per_agent.items():
            agent_runs[aid] += 1
            roles[aid] = roles.get(aid) or ae.role
            agent_cost_complete[aid] = agent_cost_complete.get(aid, True) and ae.cost_complete
            acc[aid]["tokens"].append(ae.total_tokens)
            acc[aid]["input_tokens"].append(ae.input_tokens)
            acc[aid]["output_tokens"].append(ae.output_tokens)
            if ae.cost_usd is not None:
                acc[aid]["cost_usd"].append(ae.cost_usd)
            elif ae.llm_calls == 0:
                acc[aid]["cost_usd"].append(0.0)
            acc[aid]["llm_calls"].append(ae.llm_calls)
            acc[aid]["tool_calls"].append(ae.tool_calls)
            acc[aid]["messages_sent"].append(ae.messages_sent)
            acc[aid]["artifacts_produced"].append(ae.artifacts_produced)
            acc[aid]["self_time_s"].append(ae.self_time_s)
            acc[aid]["idle_time_s"].append(ae.idle_time_s)
            acc[aid]["errors"].append(ae.errors)
    n = max(1, len(effs))
    total_tokens = sum(e.total_tokens for e in effs) or 1
    for aid, d in acc.items():
        per_agent[aid] = {k: mean(v) for k, v in d.items() if k != "cost_usd"}
        agent_costs = d.get("cost_usd", [])
        per_agent[aid]["cost_usd"] = mean(agent_costs) if agent_costs else None
        per_agent[aid]["total_cost_usd"] = sum(agent_costs) if agent_costs else None
        per_agent[aid]["cost_complete"] = agent_cost_complete.get(aid, True)
        per_agent[aid]["role"] = roles.get(aid)
        per_agent[aid]["invocation_rate"] = agent_runs[aid] / n
        per_agent[aid]["token_share"] = sum(d["tokens"]) / total_tokens
    return EfficiencySummary(
        tokens=S(e.total_tokens for e in effs), input_tokens=S(e.input_tokens for e in effs),
        output_tokens=S(e.output_tokens for e in effs), llm_calls=S(e.llm_calls for e in effs),
        tool_calls=S(e.tool_calls for e in effs), messages=S(e.messages for e in effs),
        cost_usd=S(costs), cost_complete=cost_complete, cost_total=total_cost,
        # Known cost divided by *all* successes.  When pricing is incomplete this
        # understates the true figure (unpriced runs contribute nothing to the
        # numerator but still count in the denominator); cost_complete=False
        # flags it and the report marks it with "*".
        cost_per_success=(total_cost / successes) if successes and total_cost is not None else None,
        wall_clock_s=S(e.wall_clock_s for e in effs), agent_time_s=S(e.agent_time_s for e in effs),
        waiting_time_s=S(e.waiting_time_s for e in effs), errors=S(e.errors for e in effs), per_agent=per_agent)


class EvaluationResult:
    """Performance, efficiency and (when computed) coordination of one swarm
    on one benchmark.  ``per_run_coordination`` / ``coordination`` are
    ``None`` when the analysis was switched off; a run whose analysis failed
    has ``None`` in ``per_run_coordination`` and its error, by run id, in
    ``coordination_errors``."""

    def __init__(self, runs: RunSet, per_run: List[EfficiencyMetrics], performance: PerformanceMetrics,
                 efficiency: EfficiencySummary,
                 per_run_coordination: Optional[List[Optional[CoordinationMetrics]]] = None,
                 coordination: Optional[CoordinationSummary] = None,
                 coordination_errors: Optional[Dict[str, str]] = None) -> None:
        self.runs = runs
        self.per_run = per_run
        self.performance = performance
        self.efficiency = efficiency
        self.per_run_coordination = per_run_coordination
        self.coordination = coordination
        self.coordination_errors = coordination_errors or {}

    @property
    def label(self) -> str:
        return self.runs.label

    @property
    def benchmark(self) -> str:
        return self.runs.benchmark_name

    @property
    def success_rate(self) -> float:
        return self.performance.success_rate

    @property
    def cost(self) -> Dict[str, Any]:
        """Known cost; values are ``None`` when nothing could be priced and
        ``complete`` is False when some calls were unpriced."""
        per_run = self.efficiency.cost_usd.mean if self.efficiency.cost_usd.n else None
        return {"total": self.efficiency.cost_total, "per_run": per_run,
                "per_success": self.efficiency.cost_per_success, "complete": self.efficiency.cost_complete}

    @property
    def latency(self) -> Dict[str, float]:
        return {"wall_clock": self.efficiency.wall_clock_s.mean, "agent_time": self.efficiency.agent_time_s.mean,
                "waiting": self.efficiency.waiting_time_s.mean}

    @property
    def agents(self) -> List[str]:
        return list(self.efficiency.per_agent.keys())

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label, "benchmark": self.benchmark, "runs": len(self.runs),
            "tasks": self.performance.n_tasks, "trials": self.performance.trials,
            "success_rate": round(self.success_rate, 4),
            "mean_score": round(self.performance.mean_score, 4),
            "run_errors": self.performance.run_errors,
            "cost": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.cost.items()},  # None = unknown
            "latency": {k: round(v, 3) for k, v in self.latency.items()},
            "tokens_per_run": round(self.efficiency.tokens.mean, 1),
            "agents": self.agents,
        }

    def to_dict(self, include_runs: bool = False) -> Dict[str, Any]:
        coord: List[Optional[CoordinationMetrics]] = (list(self.per_run_coordination)
                                                      if self.per_run_coordination is not None
                                                      else [None] * len(self.per_run))
        d = {
            "summary": self.summary(),
            "performance": self.performance.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "coordination": self.coordination.to_dict() if self.coordination is not None else None,
            "coordination_errors": dict(self.coordination_errors),
            "per_run": [{"run_id": t.run_id, "task_id": t.task_id, "trial": t.trial, "status": t.status,
                         "success": t.succeeded, "score": t.score, "cost_usd": e.cost_usd,
                         "tokens": e.total_tokens, "llm_calls": e.llm_calls, "tool_calls": e.tool_calls,
                         "wall_clock_s": e.wall_clock_s,
                         # None (not []) when coordination was not computed
                         "dead_agents": c.dead_agents if c is not None else None,
                         "message_receive_coverage": (c.coverage.message_receive_coverage
                                                      if c is not None else None)}
                        for t, e, c in zip(self.runs.trajectories, self.per_run, coord)],
        }
        if include_runs:
            d["runs"] = self.runs.to_dict()
        return d

    def save(self, path: str, include_runs: bool = True) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(include_runs=include_runs), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str, pricing: Optional[PricingTable] = None) -> "EvaluationResult":
        """Reload a saved result (re-priced with ``pricing`` if given)."""
        return analyze_runs(load_runset(path), pricing=pricing)

    def report(self, **kwargs: Any) -> str:
        from .report.text import render_text
        return render_text(self, **kwargs)

    def __repr__(self) -> str:
        total = self.efficiency.cost_total
        cost = "unknown" if total is None else f"${total:.2f}" + ("" if self.efficiency.cost_complete else "*")
        return (f"EvaluationResult(label={self.label!r}, runs={len(self.runs)}, success={self.success_rate:.3f}, "
                f"cost={cost})")


class AnalysisFailed(Exception):
    """Raised by ``evaluate()`` when the analysis fails after the runs have
    executed.  ``runs`` keeps the trajectories and ``result`` is the plain
    evaluation (``coordination=False``); the original error is ``__cause__``."""

    def __init__(self, message: str, runs: RunSet, result: EvaluationResult) -> None:
        super().__init__(message)
        self.runs = runs
        self.result = result


def load_runset(path: str) -> RunSet:
    """Load trajectories from a saved RunSet or a saved EvaluationResult.
    Raises ``ValueError`` for a result saved without its trajectories."""
    with open(path) as f:
        d = json.load(f)
    if "runs" in d:
        return RunSet.from_dict(d["runs"])
    if "trajectories" in d:
        return RunSet.from_dict(d)
    raise ValueError(f"{path} has no trajectories; save with include_runs=True to reload")


def analyze_runs(runs: RunSet, pricing: Optional[PricingTable] = None, coordination: bool = True,
                 similarity: Any = None, smell_config: Any = None) -> EvaluationResult:
    """Recompute the evaluation from trajectories (e.g. a saved RunSet).
    With ``pricing``, every LLM call is re-priced from its recorded model and
    token counts instead of using the cost stored at run time.
    ``coordination=False`` skips the execution-graph analysis; ``similarity``
    is the artifact-redundancy backend and ``smell_config`` the smell
    thresholds (``SmellConfig``, dict of overrides or JSON path).  A run whose
    analysis fails is left out of the coordination summary with a warning;
    when two or more runs all fail with the same exception type, it is raised."""
    per_run = [compute_efficiency(t, pricing) for t in runs.trajectories]
    performance = compute_performance(runs.trajectories, per_run)
    efficiency = summarize_efficiency(runs.trajectories, per_run)
    per_run_coord: Optional[List[Optional[CoordinationMetrics]]] = None
    coord_summary: Optional[CoordinationSummary] = None
    errors: Dict[str, str] = {}
    first_exc: Optional[Exception] = None
    kinds: set[type] = set()
    if coordination:
        similarity = resolve_similarity(similarity)   # once, not per run
        smell_config = resolve_smell_config(smell_config)
        per_run_coord = []
        for t, e in zip(runs.trajectories, per_run):
            try:
                cm: Optional[CoordinationMetrics] = compute_coordination(
                    t, similarity=similarity, smell_config=smell_config, efficiency=e)
            except Exception as exc:  # an odd trace must not cost the plain evaluation
                cm = None
                errors[t.run_id] = f"{type(exc).__name__}: {exc}"
                first_exc = first_exc or exc
                kinds.add(type(exc))
            per_run_coord.append(cm)
        done = [cm for cm in per_run_coord if cm is not None]
        if first_exc is not None:
            if not done and len(per_run_coord) > 1 and len(kinds) == 1:
                raise first_exc  # every run, one cause: e.g. a missing optional dependency, not an odd trace
            warnings.warn(f"coordination analysis failed on {len(per_run_coord) - len(done)}/{len(per_run_coord)} "
                          f"runs: {type(first_exc).__name__}: {first_exc}", stacklevel=2)
        coord_summary = summarize_coordination(done) if done else None
    return EvaluationResult(runs, per_run, performance, efficiency, per_run_coordination=per_run_coord,
                            coordination=coord_summary, coordination_errors=errors)


def evaluate(swarm: Any, benchmark: Benchmark, trials: int = 1, seed: int = 0, evaluator: Any = None,
             clock_factory: Optional[Callable[[], Clock]] = None, pricing: Optional[PricingTable] = None,
             label: str = "baseline", progress: bool = False, raise_errors: bool = False,
             on_run: Optional[Callable[[Trajectory], None]] = None, capture_text: bool = True,
             max_text_chars: int = 20000, coordination: bool = True, similarity: Any = None,
             smell_config: Any = None) -> EvaluationResult:
    """Run ``swarm`` on ``benchmark`` ``trials`` times and return an EvaluationResult.

    ``swarm`` may be a Swarm instance or class, a ``factory() -> Swarm`` or a
    bare ``run(task, ctx)`` function (sync or async).  ``evaluator``, when
    given, overrides the task and benchmark evaluators.  ``capture_text=False``
    keeps only short previews of prompts, outputs and artifacts.
    ``coordination``, ``similarity`` and ``smell_config`` are passed to
    ``analyze_runs``.
    """
    opts = RunOptions(trials=trials, seed=seed, clock_factory=clock_factory, pricing=pricing, evaluator=evaluator,
                      progress=progress, raise_errors=raise_errors, on_run=on_run, capture_text=capture_text,
                      max_text_chars=max_text_chars)
    runs = SwarmRunner(swarm, benchmark, label=label, options=opts).run()
    try:
        return analyze_runs(runs, coordination=coordination, similarity=similarity, smell_config=smell_config)
    except Exception as exc:  # the runs are done (and paid for): hand them back with the plain evaluation
        raise AnalysisFailed(f"coordination analysis failed: {type(exc).__name__}: {exc}", runs,
                             analyze_runs(runs, coordination=False)) from exc
