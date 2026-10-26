"""Swarm runner: executes a swarm over a benchmark with repeated trials,
recording a Trajectory per run and evaluating each outcome."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import evaluators as _evaluators
from .pricing import PricingTable
from .recorder import Clock, RunContext, WallClock
from .schema import Benchmark, Evaluation, EventType, Status, Task, Trajectory


class Swarm:
    """Interface swarms implement.  ``run`` returns the final output and
    records everything it does through ``ctx``."""

    def run(self, task: Task, ctx: RunContext) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class FunctionSwarm(Swarm):
    def __init__(self, fn: Callable[[Task, RunContext], Any]) -> None:
        self.fn = fn

    def run(self, task: Task, ctx: RunContext) -> Any:
        return self.fn(task, ctx)


SwarmFactory = Callable[[], Swarm]


def as_factory(swarm: Any) -> SwarmFactory:
    """Accept a Swarm instance, a Swarm class, a zero-argument
    ``factory() -> Swarm`` or a bare ``run(task, ctx)`` function and
    normalise to a factory."""
    if isinstance(swarm, type):
        if not callable(getattr(swarm, "run", None)):
            raise TypeError(f"{swarm.__name__} has no run(task, ctx) method")
        return swarm  # the class itself is the factory: a fresh instance per run
    if hasattr(swarm, "run") and callable(getattr(swarm, "run")):
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
        return swarm  # assume factory()
    raise TypeError("swarm must be a Swarm, a factory() -> Swarm, or a run(task, ctx) function")


class UsageError(RuntimeError):
    """SwarmEval was called incorrectly.  Unlike a swarm crash this is never
    recorded as a run result; it propagates to the caller."""


_LOOP_MSG = ("an async swarm cannot be evaluated from inside a running event loop; "
             "call evaluate() from synchronous code (e.g. in a worker thread)")


def _in_event_loop() -> bool:
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
    return await aw


def derive_seed(base: int, task_id: str, trial: int) -> int:
    """Deterministic per-run seed so a run set can be reproduced exactly."""
    h = hashlib.sha1(f"{base}|{task_id}|{trial}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class RunOptions:
    trials: int = 1
    seed: int = 0
    clock_factory: Optional[Callable[[], Clock]] = None
    pricing: Optional[PricingTable] = None
    evaluator: Any = None
    raise_errors: bool = False
    capture_text: bool = True
    max_text_chars: int = 20000
    on_run: Optional[Callable[[Trajectory], None]] = None
    progress: bool = False


@dataclass
class RunSet:
    """All trajectories from one evaluation of one swarm on one benchmark.
    Trajectories are the source of truth: every report can be recomputed
    from a saved RunSet."""
    label: str
    benchmark_name: str
    trajectories: List[Trajectory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def task_ids(self) -> List[str]:
        seen: List[str] = []
        for t in self.trajectories:
            if t.task_id not in seen:
                seen.append(t.task_id)
        return seen

    def by_task(self) -> Dict[str, List[Trajectory]]:
        out: Dict[str, List[Trajectory]] = {}
        for t in self.trajectories:
            out.setdefault(t.task_id, []).append(t)
        return out

    def get(self, task_id: str, trial: int) -> Optional[Trajectory]:
        for t in self.trajectories:
            if t.task_id == task_id and t.trial == trial:
                return t
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "benchmark_name": self.benchmark_name,
                "trajectories": [t.to_dict() for t in self.trajectories], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSet":
        return cls(d.get("label", "runs"), d.get("benchmark_name", ""),
                   [Trajectory.from_dict(t) for t in d.get("trajectories", [])], d.get("metadata", {}))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str) -> "RunSet":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class SwarmRunner:
    def __init__(self, swarm: Any, benchmark: Benchmark, label: str = "baseline",
                 options: Optional[RunOptions] = None, **kwargs: Any) -> None:
        self.factory = as_factory(swarm)
        self.benchmark = benchmark
        self.label = label
        self.options = options or RunOptions(**kwargs)
        self.benchmark.validate()
        # Build every evaluator up front so a bad spec fails before any
        # (possibly paid) run, and each run reuses the same instance.
        self._evaluators: Dict[str, Optional[_evaluators.Evaluator]] = {}
        for task in self.benchmark.tasks:
            ev = _evaluators.resolve(task, self.benchmark.default_evaluator, override=self.options.evaluator)
            if ev is not None:
                ev.check(task)
            self._evaluators[task.task_id] = ev

    def _clock(self) -> Clock:
        return self.options.clock_factory() if self.options.clock_factory else WallClock()

    def run_task(self, task: Task, trial: int = 0) -> Trajectory:
        opts = self.options
        seed = derive_seed(opts.seed, task.task_id, trial)
        ctx = RunContext(task, clock=self._clock(), seed=seed, trial=trial, pricing=opts.pricing,
                         capture_text=opts.capture_text, max_text_chars=opts.max_text_chars)
        swarm = self.factory()
        if inspect.iscoroutinefunction(getattr(swarm, "run", None)) and _in_event_loop():
            raise UsageError(_LOOP_MSG)  # fail before recording anything
        ctx.start()
        output: Any = None
        status = Status.SUCCESS.value
        error: Optional[str] = None
        try:
            output = swarm.run(task, ctx)
            if inspect.isawaitable(output):
                output = _await(output)
            if ctx.trajectory.final_output is None:
                ctx.set_output(output)
            else:
                output = ctx.trajectory.final_output
        except UsageError:  # misuse of SwarmEval, not a swarm failure: surface it
            raise
        except Exception as exc:  # the swarm crashed -- that is a result, not a bug in SwarmEval
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
        if task.task_id in self._evaluators:
            evaluator = self._evaluators[task.task_id]
        else:  # a task not in the benchmark (run_task called directly)
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
            return Evaluation(False, 0.0, evaluator.name, {"reason": f"evaluator error: {exc}"})

    def run(self) -> RunSet:
        rs = RunSet(self.label, self.benchmark.name)
        total = len(self.benchmark.tasks) * self.options.trials
        done = 0
        try:
            for trial in range(self.options.trials):
                for task in self.benchmark.tasks:
                    rs.trajectories.append(self.run_task(task, trial))
                    done += 1
                    if self.options.progress:
                        sys.stderr.write(f"\r[{self.label}] {done}/{total} runs")
                        sys.stderr.flush()
        except KeyboardInterrupt:  # keep the finished runs; the interrupted one is dropped
            rs.metadata["interrupted"] = True
            rs.metadata["planned_runs"] = total
        if self.options.progress:
            sys.stderr.write("\n")
        return rs
