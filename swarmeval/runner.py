"""Swarm runner: executes a swarm over a benchmark with repeated trials,
recording a Trajectory per run and evaluating each outcome.

Every evaluator is built before the first run, so a bad spec fails before any (possibly paid) run. A swarm
crash is recorded as a result; a ``UsageError`` (misuse of SwarmEval) propagates to the caller. Ctrl-C keeps
the finished runs and is re-raised as ``RunInterrupted``, so a script stops instead of starting its next
evaluation.

Classes:
- Swarm: Interface swarms implement. ``run`` returns the final output and records everything it does through
  ``ctx``.
- UsageError: SwarmEval was called incorrectly. Unlike a swarm crash this is never recorded as a run result; it
  propagates to the caller.
- RunSet: All trajectories from one evaluation of one swarm on one benchmark. Trajectories are the source of truth:
  every report can be recomputed from a saved RunSet.
- RunInterrupted: Ctrl-C during ``SwarmRunner.run``. ``runs`` is the partial RunSet (the interrupted run is
  dropped); ``evaluate()`` sets ``result`` to its analysis.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from . import evaluators as _evaluators
from .pricing import PricingTable
from .recorder import Clock, RunContext, WallClock
from .schema import Benchmark, Evaluation, EventType, Status, Task, Trajectory


class Swarm:

    def run(self, task: Task, ctx: RunContext) -> Any:  # pragma: no cover - interface
        """Run the swarm on ``task``, recording through ``ctx``; return the final output."""
        raise NotImplementedError


class FunctionSwarm(Swarm):
    def __init__(self, fn: Callable[[Task, RunContext], Any]) -> None:
        self.fn = fn

    def run(self, task: Task, ctx: RunContext) -> Any:
        """Call the wrapped function."""
        return self.fn(task, ctx)


SwarmFactory = Callable[[], Swarm]


def as_factory(swarm: Any) -> SwarmFactory:
    """Normalise a Swarm instance, Swarm class, zero-argument factory or bare ``run(task, ctx)`` function
    to a factory."""
    if isinstance(swarm, type):
        if not callable(getattr(swarm, "run", None)):
            raise TypeError(f"{swarm.__name__} has no run(task, ctx) method")
        return swarm
    if hasattr(swarm, "run") and callable(swarm.run):
        return lambda: swarm
    if callable(swarm):
        try:
            n = len([p for p in inspect.signature(swarm).parameters.values()
                     if p.default is inspect.Parameter.empty and p.kind in
                     (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        except (TypeError, ValueError):
            n = 0
        if n == 2:
            fn = swarm
            return lambda: FunctionSwarm(fn)
        return swarm
    raise TypeError("swarm must be a Swarm, a factory() -> Swarm, or a run(task, ctx) function")


class UsageError(RuntimeError):
    pass


_LOOP_MSG = ("an async swarm cannot be evaluated from inside a running event loop; "
             "call evaluate() from synchronous code (e.g. in a worker thread)")


def _in_event_loop() -> bool:
    """True if called from inside a running asyncio event loop."""
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _await(aw: Any) -> Any:
    """Run an async swarm's ``run`` coroutine to completion."""
    import asyncio
    if not _in_event_loop():
        return asyncio.run(_as_coro(aw))
    if inspect.iscoroutine(aw):
        aw.close()
    raise UsageError(_LOOP_MSG)


async def _as_coro(aw: Any) -> Any:
    """Wrap any awaitable in a coroutine for ``asyncio.run``."""
    return await aw


def derive_seed(base: int, task_id: str, trial: int) -> int:
    """Deterministic per-run seed so a run set can be reproduced exactly."""
    h = hashlib.sha1(f"{base}|{task_id}|{trial}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class RunOptions:
    trials: int = 1
    seed: int = 0
    clock_factory: Callable[[], Clock] | None = None
    pricing: PricingTable | None = None
    evaluator: Any = None
    raise_errors: bool = False
    capture_text: bool = True
    max_text_chars: int = 20000
    on_run: Callable[[Trajectory], None] | None = None
    progress: bool = False


@dataclass
class RunSet:
    label: str
    benchmark_name: str
    trajectories: list[Trajectory] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def task_ids(self) -> list[str]:
        """Task ids in first-seen order."""
        seen: list[str] = []
        for t in self.trajectories:
            if t.task_id not in seen:
                seen.append(t.task_id)
        return seen

    def by_task(self) -> dict[str, list[Trajectory]]:
        """Trajectories grouped by task id."""
        out: dict[str, list[Trajectory]] = {}
        for t in self.trajectories:
            out.setdefault(t.task_id, []).append(t)
        return out

    def to_dict(self) -> dict[str, Any]:
        """The run set as a JSON-ready dict."""
        return {"label": self.label, "benchmark_name": self.benchmark_name,
                "trajectories": [t.to_dict() for t in self.trajectories], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunSet:
        """Rebuild a run set from a dict."""
        return cls(d.get("label", "runs"), d.get("benchmark_name", ""),
                   [Trajectory.from_dict(t) for t in d.get("trajectories", [])], d.get("metadata", {}))

    def save(self, path: str) -> None:
        """Write the run set as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str) -> RunSet:
        """Load a run set from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class RunInterrupted(KeyboardInterrupt):

    def __init__(self, runs: RunSet) -> None:
        super().__init__(f"interrupted after {len(runs)} runs")
        self.runs = runs
        self.result: Any = None


class SwarmRunner:
    def __init__(self, swarm: Any, benchmark: Benchmark, label: str = "baseline",
                 options: RunOptions | None = None, **kwargs: Any) -> None:
        self.factory = as_factory(swarm)
        self.benchmark = benchmark
        self.label = label
        self.options = options or RunOptions(**kwargs)
        self.benchmark.validate()
        self._evaluators: dict[str, _evaluators.Evaluator | None] = {}
        for task in self.benchmark.tasks:
            ev = _evaluators.resolve(task, self.benchmark.default_evaluator, override=self.options.evaluator)
            if ev is not None:
                ev.check(task)
            self._evaluators[task.task_id] = ev

    def _clock(self) -> Clock:
        """A fresh clock for one run."""
        return self.options.clock_factory() if self.options.clock_factory else WallClock()

    def run_task(self, task: Task, trial: int = 0) -> Trajectory:
        """Run one task once, recording and evaluating it; a crash is recorded as a result."""
        opts = self.options
        seed = derive_seed(opts.seed, task.task_id, trial)
        ctx = RunContext(task, clock=self._clock(), seed=seed, trial=trial, pricing=opts.pricing,
                         capture_text=opts.capture_text, max_text_chars=opts.max_text_chars)
        swarm = self.factory()
        if inspect.iscoroutinefunction(getattr(swarm, "run", None)) and _in_event_loop():
            raise UsageError(_LOOP_MSG)
        ctx.start()
        output: Any = None
        status = Status.SUCCESS.value
        error: str | None = None
        try:
            output = swarm.run(task, ctx)
            if inspect.isawaitable(output):
                output = _await(output)
            if ctx.trajectory.final_output is None:
                ctx.set_output(output)
            else:
                output = ctx.trajectory.final_output
        except UsageError:
            raise
        except Exception as exc:
            status = Status.ERROR.value
            error = f"{type(exc).__name__}: {exc}"
            ctx.record_exception(exc)
            if opts.raise_errors:
                raise
        traj = ctx.finish(status, error)
        traj.evaluation = self._evaluate(task, output, traj)
        ctx.record(EventType.EVAL, status="success" if traj.evaluation.success else "error",
                   content=traj.evaluation.evaluator, metadata={"score": traj.evaluation.score,
                                                                 "success": traj.evaluation.success})
        traj.metadata["swarm"] = type(swarm).__name__
        if opts.on_run:
            opts.on_run(traj)
        return traj

    def _evaluate(self, task: Task, output: Any, traj: Trajectory) -> Evaluation:
        """Evaluate one run's output with the task's prebuilt evaluator."""
        if task.task_id in self._evaluators:
            evaluator = self._evaluators[task.task_id]
        else:
            evaluator = _evaluators.resolve(task, self.benchmark.default_evaluator, override=self.options.evaluator)
        if traj.status != Status.SUCCESS.value and output is None:
            return Evaluation(False, 0.0, evaluator.name if evaluator else "completion",
                              {"reason": "run failed", "error": traj.error})
        if evaluator is None:
            ok = traj.status == Status.SUCCESS.value and output is not None
            return Evaluation(ok, 1.0 if ok else 0.0, "completion")
        try:
            return evaluator.evaluate(task, output, traj)
        except Exception as exc:
            return Evaluation(False, 0.0, evaluator.name, {"reason": f"evaluator error: {exc}",
                                                         "evaluator_error": True})

    def run(self) -> RunSet:
        """Run every task for every trial; on Ctrl-C, raise ``RunInterrupted`` carrying the runs finished so far."""
        rs = RunSet(self.label, self.benchmark.name)
        total = len(self.benchmark.tasks) * self.options.trials
        done = 0
        interrupted = False
        try:
            for trial in range(self.options.trials):
                for task in self.benchmark.tasks:
                    rs.trajectories.append(self.run_task(task, trial))
                    done += 1
                    if self.options.progress:
                        sys.stderr.write(f"\r[{self.label}] {done}/{total} runs")
                        sys.stderr.flush()
        except KeyboardInterrupt:
            interrupted = True
            rs.metadata["interrupted"] = True
            rs.metadata["planned_runs"] = total
        if self.options.progress:
            sys.stderr.write("\n")
        if interrupted:
            raise RunInterrupted(rs)
        return rs
