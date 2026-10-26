"""Regression tests for the repo audit of main posted on PR #3 (rohitp30).

A swarm here raises KeyboardInterrupt to simulate Ctrl-C.  pytest treats an
escaping KeyboardInterrupt as the user stopping the session, so those calls
are wrapped: an interrupt that escapes is a test failure, not an abort."""
import asyncio
from types import SimpleNamespace

import pytest

from swarmeval import Benchmark, EvaluationResult, RunContext, SimClock, Task, compute_efficiency, evaluate, \
    evaluators as E
from swarmeval.cli import main
from swarmeval.integrations.anthropic import instrument_client
from swarmeval.pricing import PricingTable
from swarmeval.report.text import box


def five_tasks():
    return Benchmark("b", [Task(f"t{i}", "q", evaluator={"type": "exact_match", "expected": "42"})
                           for i in range(1, 6)])


def no_escape(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped: finished runs were lost")


# -- audit 1: an interrupt keeps the finished runs ---------------------------- #
def test_interrupt_returns_runs_finished_before_it():
    def swarm(task, ctx):
        with ctx.agent("solo"):
            if task.task_id == "t3":
                raise KeyboardInterrupt  # through an open span, as in a real Ctrl-C
        return "42"

    res = no_escape(evaluate, swarm, five_tasks(), clock_factory=SimClock)
    assert [t.task_id for t in res.runs.trajectories] == ["t1", "t2"]
    assert res.runs.metadata["interrupted"] is True
    assert "interrupted after 2/5 runs" in res.report()


def test_cli_run_saves_partial_results_on_interrupt(tmp_path, monkeypatch):
    (tmp_path / "intswarm.py").write_text(
        "def run(task, ctx):\n    if task.task_id == 't3':\n        raise KeyboardInterrupt\n    return '42'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    five_tasks().save(str(tmp_path / "bench.json"))
    rc = no_escape(main, ["run", "--swarm", "intswarm:run", "--benchmark", str(tmp_path / "bench.json"),
                          "--out", str(tmp_path / "out")])
    saved = EvaluationResult.load(str(tmp_path / "out" / "baseline.json"))
    assert rc == 130
    assert len(saved.runs.trajectories) == 2 and saved.runs.metadata["interrupted"] is True


# -- audit 2: a streamed call is unknown cost, never $0 ----------------------- #
def streaming_client(is_async=False):
    stream = object()  # an SDK stream has no .usage or .content

    def create(**kw):
        return stream

    async def acreate(**kw):
        return stream

    return stream, instrument_client(SimpleNamespace(messages=SimpleNamespace(create=acreate if is_async else create)))


def started_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock())
    ctx.start()
    return ctx


@pytest.mark.parametrize("is_async", [False, True])
def test_streamed_call_is_recorded_with_unknown_cost(is_async):
    stream, client = streaming_client(is_async)
    ctx = started_ctx()
    with ctx.agent("a"):
        out = client.messages.create(model="claude-sonnet-5", messages=[], stream=True)
        out = asyncio.run(out) if is_async else out
    traj = ctx.finish()
    (ev,) = traj.llm_calls()
    assert out is stream
    assert ev.cost_usd is None and ev.metadata.get("streamed") is True
    assert compute_efficiency(traj).cost_usd is None
    assert compute_efficiency(traj, pricing=PricingTable()).cost_usd is None  # re-pricing must not make it $0


def test_streamed_call_inside_open_llm_call_does_not_zero_it():
    stream, client = streaming_client()
    ctx = started_ctx()
    with ctx.agent("a") as a:
        with a.llm_call(model="claude-sonnet-5"):
            out = client.messages.create(model="claude-sonnet-5", messages=[], stream=True)
    (ev,) = ctx.finish().llm_calls()
    assert out is stream
    assert ev.cost_usd is None and ev.metadata.get("streamed") is True


# -- audit 3: a crashing evaluator is visible --------------------------------- #
def broken(task, output, traj):
    raise NameError("typo")


def report_line(report, label):
    (line,) = [l for l in report.splitlines() if label in l]
    return line


def test_evaluator_exception_is_counted_and_flagged():
    res = evaluate(lambda t, c: "42", five_tasks(), evaluator=E.CallableEvaluator(broken), clock_factory=SimClock)
    assert res.performance.evaluator_errors == 5
    line = report_line(res.report(), "Evaluator Errors")
    assert "5/5" in line and "⚠" in line


def test_evaluator_errors_survive_save_and_load(tmp_path):
    res = evaluate(lambda t, c: "42", five_tasks(), evaluator=E.CallableEvaluator(broken), clock_factory=SimClock)
    res.save(str(tmp_path / "r.json"))
    assert EvaluationResult.load(str(tmp_path / "r.json")).performance.evaluator_errors == 5


# -- audit 4: composite weights must match its evaluators --------------------- #
def test_weight_count_mismatch_fails_before_any_run():
    spec = {"type": "composite", "weights": [1],
            "evaluators": [{"type": "exact_match", "expected": "a"}, {"type": "exact_match", "expected": "b"}]}
    with pytest.raises(ValueError, match="weights"):
        Benchmark("b", [Task("q", "what?", evaluator=spec)]).validate()


# -- audit 5: the report box widens instead of cutting values ----------------- #
def test_box_keeps_long_values_and_flags():
    value = "1,234,567  (in 1,000,000 / out 234,567)"
    out = box("SWARM EVALUATION", [("Tokens / Run", value, "⚠"), "short"])
    assert value in out and "⚠" in out
    assert len({len(line) for line in out.splitlines()}) == 1
