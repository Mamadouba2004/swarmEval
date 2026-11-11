"""Top-level evaluation: run a swarm on a benchmark and summarise the result.

``evaluate()`` accepts a Swarm instance or class, a ``factory() -> Swarm``
or a bare ``run(task, ctx)`` function (sync or async).  ``evaluator``, when
given, overrides the task and benchmark evaluators; ``capture_text=False``
keeps only short previews of prompts, outputs and artifacts.  The similarity
backend is resolved before any run, so a missing optional dependency fails
fast.  Ctrl-C raises ``RunInterrupted`` (a ``KeyboardInterrupt``) whose
``result`` is the evaluation of the runs finished before it.  If the analysis
fails after the runs, ``AnalysisFailed`` carries them and the plain
evaluation, with the per-run errors in ``coordination_errors``.

``analyze_runs()`` recomputes everything from trajectories, so a saved RunSet
can be re-analysed.  With ``pricing`` every LLM call is re-priced from its
recorded model and token counts.  ``coordination=False`` skips the
execution-graph analysis; ``similarity`` is the artifact-redundancy backend
and ``smell_config`` the smell thresholds (``SmellConfig``, a dict of
overrides or a JSON path).  A run whose analysis fails is left out of the
coordination summary with a warning; when two or more runs all fail with the
same exception type, the error is raised.

Unknown costs stay unknown: they are left out of sums and flagged via ``cost_complete``, never counted as $0.
Cost per successful task is known cost divided by all successes, so with incomplete pricing it understates the
true figure; ``cost_complete=False`` flags this and the report marks it with "*".

Classes:
- EfficiencySummary: Efficiency metrics aggregated over all runs in a RunSet.
- EvaluationResult: Performance, efficiency and (when computed) coordination of one swarm on one benchmark.
  ``per_run_coordination`` / ``coordination`` are ``None`` when the analysis was switched off; a run whose analysis
  failed has ``None`` in ``per_run_coordination`` and its error, by run id, in ``coordination_errors``.
- AnalysisFailed: Raised by ``evaluate()`` when the analysis fails after the runs have executed. ``runs`` keeps the
  trajectories and ``result`` is the plain evaluation (``coordination=False``); the original error is ``__cause__``.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .diagnostics.smells import resolve_smell_config
from .metrics.coordination import (
    ANALYSES,
    CoordinationAnalysis,
    CoordinationMetrics,
    CoordinationSummary,
    compute_coordination,
    summarize_coordination,
)
from .metrics.efficiency import EfficiencyMetrics, compute_efficiency
from .metrics.performance import PerformanceMetrics, compute_performance
from .pricing import PricingTable
from .recorder import Clock
from .runner import RunInterrupted, RunOptions, RunSet, SwarmRunner
from .schema import Benchmark, Trajectory
from .similarity import resolve_similarity
from .stats import Summary, mean, summarize, summary_dict


@dataclass
class EfficiencySummary:
    tokens: Summary
    input_tokens: Summary
    output_tokens: Summary
    llm_calls: Summary
    tool_calls: Summary
    messages: Summary
    cost_usd: Summary
    cost_complete: bool
    cost_total: float | None
    cost_per_success: float | None
    wall_clock_s: Summary
    agent_time_s: Summary
    waiting_time_s: Summary
    errors: Summary
    per_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    duration_complete: bool = True
    unknown_duration_calls: int = 0
    streamed_cost_calls: int = 0
    unpriced_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dict."""
        return summary_dict(self)


def summarize_efficiency(trajectories: Sequence[Trajectory], effs: Sequence[EfficiencyMetrics]) -> EfficiencySummary:
    """Efficiency over runs; unknown costs are left out of sums and flagged incomplete."""
    S = summarize
    cost_complete = all(e.cost_complete for e in effs)
    costs = [e.cost_usd for e in effs if e.cost_usd is not None]
    total_cost: float | None = sum(costs) if costs else None
    successes = sum(1 for t in trajectories if t.succeeded)
    per_agent: dict[str, dict[str, Any]] = {}
    agent_runs: dict[str, int] = Counter()
    roles: dict[str, str | None] = {}
    agent_cost_complete: dict[str, bool] = {}
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
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
        cost_per_success=(total_cost / successes) if successes and total_cost is not None else None,
        wall_clock_s=S(e.wall_clock_s for e in effs), agent_time_s=S(e.agent_time_s for e in effs),
        waiting_time_s=S(e.waiting_time_s for e in effs), errors=S(e.errors for e in effs), per_agent=per_agent,
        duration_complete=all(e.duration_complete for e in effs),
        unknown_duration_calls=sum(e.unknown_duration_calls for e in effs),
        streamed_cost_calls=sum(e.streamed_cost_calls for e in effs),
        unpriced_calls=sum(e.unpriced_calls for e in effs))


class EvaluationResult:

    def __init__(self, runs: RunSet, per_run: list[EfficiencyMetrics], performance: PerformanceMetrics,
                 efficiency: EfficiencySummary,
                 per_run_coordination: list[CoordinationMetrics | None] | None = None,
                 coordination: CoordinationSummary | None = None,
                 coordination_errors: dict[str, str] | None = None) -> None:
        self.runs = runs
        self.per_run = per_run
        self.performance = performance
        self.efficiency = efficiency
        self.per_run_coordination = per_run_coordination
        self.coordination = coordination
        self.coordination_errors = coordination_errors or {}

    @property
    def label(self) -> str:
        """Label of the evaluated configuration."""
        return self.runs.label

    @property
    def benchmark(self) -> str:
        """Name of the benchmark."""
        return self.runs.benchmark_name

    @property
    def success_rate(self) -> float:
        """Share of runs that succeeded."""
        return self.performance.success_rate

    @property
    def cost(self) -> dict[str, Any]:
        """Known cost; values are ``None`` when nothing could be priced and
        ``complete`` is False when some calls were unpriced."""
        per_run = self.efficiency.cost_usd.mean if self.efficiency.cost_usd.n else None
        return {"total": self.efficiency.cost_total, "per_run": per_run,
                "per_success": self.efficiency.cost_per_success, "complete": self.efficiency.cost_complete}

    @property
    def latency(self) -> dict[str, float]:
        """Mean wall-clock, agent and waiting time per run."""
        return {"wall_clock": self.efficiency.wall_clock_s.mean, "agent_time": self.efficiency.agent_time_s.mean,
                "waiting": self.efficiency.waiting_time_s.mean}

    @property
    def agents(self) -> list[str]:
        """Agents seen in any run."""
        return list(self.efficiency.per_agent.keys())

    def summary(self) -> dict[str, Any]:
        """Headline numbers of the evaluation."""
        return {
            "label": self.label, "benchmark": self.benchmark, "runs": len(self.runs),
            "tasks": self.performance.n_tasks, "trials": self.performance.trials,
            "success_rate": round(self.success_rate, 4),
            "mean_score": round(self.performance.mean_score, 4),
            "run_errors": self.performance.run_errors,
            "cost": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.cost.items()},
            "latency": {k: round(v, 3) for k, v in self.latency.items()},
            "tokens_per_run": round(self.efficiency.tokens.mean, 1),
            "agents": self.agents,
        }

    def to_dict(self, include_runs: bool = False) -> dict[str, Any]:
        """The result as a JSON-ready dict, with the run set when ``include_runs``."""
        coord: list[CoordinationMetrics | None] = (list(self.per_run_coordination)
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
                         "dead_agents": c.dead_agents if c is not None else None,
                         "message_receive_coverage": (c.coverage.message_receive_coverage
                                                      if c is not None else None)}
                        for t, e, c in zip(self.runs.trajectories, self.per_run, coord, strict=True)],
        }
        if include_runs:
            d["runs"] = self.runs.to_dict()
        return d

    def save(self, path: str, include_runs: bool = True) -> None:
        """Write the result as JSON, including trajectories by default."""
        with open(path, "w") as f:
            json.dump(self.to_dict(include_runs=include_runs), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str, pricing: PricingTable | None = None) -> EvaluationResult:
        """Reload a saved result (re-priced with ``pricing`` if given)."""
        return analyze_runs(load_runset(path), pricing=pricing)

    def report(self, **kwargs: Any) -> str:
        """The result as a text report."""
        from .report.text import render_text
        return render_text(self, **kwargs)

    def __repr__(self) -> str:
        total = self.efficiency.cost_total
        cost = "unknown" if total is None else f"${total:.2f}" + ("" if self.efficiency.cost_complete else "*")
        return (f"EvaluationResult(label={self.label!r}, runs={len(self.runs)}, success={self.success_rate:.3f}, "
                f"cost={cost})")


class AnalysisFailed(Exception):

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


def analyze_runs(runs: RunSet, pricing: PricingTable | None = None, coordination: bool = True,
                 similarity: Any = None, smell_config: Any = None,
                 analyses: Sequence[CoordinationAnalysis] = ANALYSES) -> EvaluationResult:
    """Recompute the evaluation from trajectories, re-pricing every LLM call when ``pricing`` is given;
    coordination failures are handled as described in the module overview."""
    per_run = [compute_efficiency(t, pricing) for t in runs.trajectories]
    performance = compute_performance(runs.trajectories, per_run,
                                      interrupted=runs.metadata.get("interrupted", False))
    efficiency = summarize_efficiency(runs.trajectories, per_run)
    per_run_coord: list[CoordinationMetrics | None] | None = None
    coord_summary: CoordinationSummary | None = None
    errors: dict[str, str] = {}
    first_exc: Exception | None = None
    kinds: set[type] = set()
    if coordination:
        similarity = resolve_similarity(similarity)
        smell_config = resolve_smell_config(smell_config)
        per_run_coord = []
        for t, e in zip(runs.trajectories, per_run, strict=True):
            try:
                cm: CoordinationMetrics | None = compute_coordination(
                    t, similarity=similarity, smell_config=smell_config, efficiency=e, analyses=analyses)
            except Exception as exc:
                cm = None
                errors[t.run_id] = f"{type(exc).__name__}: {exc}"
                first_exc = first_exc or exc
                kinds.add(type(exc))
            per_run_coord.append(cm)
        done = [cm for cm in per_run_coord if cm is not None]
        if first_exc is not None:
            if not done and len(per_run_coord) > 1 and len(kinds) == 1:
                first_exc.coordination_errors = errors  # type: ignore[attr-defined]  # for the fallback report
                raise first_exc
            warnings.warn(f"coordination analysis failed on {len(per_run_coord) - len(done)}/{len(per_run_coord)} "
                          f"runs: {type(first_exc).__name__}: {first_exc}", stacklevel=2)
        coord_summary = summarize_coordination(done, analyses) if done else None
    return EvaluationResult(runs, per_run, performance, efficiency, per_run_coordination=per_run_coord,
                            coordination=coord_summary, coordination_errors=errors)


def failure_errors(exc: BaseException, runs: RunSet) -> dict[str, str]:
    """The per-run errors behind a failed coordination analysis, for the plain
    fallback result: those ``analyze_runs`` attached, else ``exc`` for every run."""
    return getattr(exc, "coordination_errors", None) or {
        t.run_id: f"{type(exc).__name__}: {exc}" for t in runs.trajectories}


def evaluate(swarm: Any, benchmark: Benchmark, trials: int = 1, seed: int = 0, evaluator: Any = None,
             clock_factory: Callable[[], Clock] | None = None, pricing: PricingTable | None = None,
             label: str = "baseline", progress: bool = False, raise_errors: bool = False,
             on_run: Callable[[Trajectory], None] | None = None, capture_text: bool = True,
             max_text_chars: int = 20000, coordination: bool = True, similarity: Any = None,
             smell_config: Any = None) -> EvaluationResult:
    """Run ``swarm`` (a Swarm, its class, a factory or a ``run(task, ctx)`` function) on ``benchmark``
    ``trials`` times and return an EvaluationResult; interrupts and failures as in the module overview."""
    opts = RunOptions(trials=trials, seed=seed, clock_factory=clock_factory, pricing=pricing, evaluator=evaluator,
                      progress=progress, raise_errors=raise_errors, on_run=on_run, capture_text=capture_text,
                      max_text_chars=max_text_chars)
    if coordination:
        similarity = resolve_similarity(similarity)
    analysis = dict(coordination=coordination, similarity=similarity, smell_config=smell_config)
    try:
        runs = SwarmRunner(swarm, benchmark, label=label, options=opts).run()
    except RunInterrupted as exc:
        try:
            exc.result = analyze_runs(exc.runs, **analysis)
        except Exception:
            exc.result = analyze_runs(exc.runs, coordination=False)
        raise
    try:
        return analyze_runs(runs, **analysis)
    except Exception as exc:
        plain = analyze_runs(runs, coordination=False)
        plain.coordination_errors = failure_errors(exc, runs)
        raise AnalysisFailed(f"coordination analysis failed: {type(exc).__name__}: {exc}", runs, plain) from exc
