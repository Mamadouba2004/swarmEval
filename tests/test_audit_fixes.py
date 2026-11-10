"""Regression tests for the repo audit of main posted on PR #3 (rohitp30).

A swarm here raises KeyboardInterrupt to simulate Ctrl-C.  pytest treats an
escaping KeyboardInterrupt as the user stopping the session, so those calls
are wrapped: an interrupt that escapes is a test failure, not an abort."""
import asyncio
from types import SimpleNamespace

import pytest

from swarmeval import Benchmark, EvaluationResult, RunContext, SimClock, Task, evaluate, evaluators as E
from swarmeval.cli import main
from swarmeval.evaluate import analyze_runs
from swarmeval.integrations.anthropic import instrument_client
from swarmeval.metrics.efficiency import compute_efficiency
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
    no_escape(main, ["run", "--swarm", "intswarm:run", "--benchmark", str(tmp_path / "bench.json"),
                     "--out", str(tmp_path / "out")])
    saved = EvaluationResult.load(str(tmp_path / "out" / "baseline.json"))
    assert len(saved.runs.trajectories) == 2 and saved.runs.metadata["interrupted"] is True


def interrupted_in_second_trial():
    """5 tasks x 2 trials, Ctrl-C at the 8th run: trial 2 is partial."""
    calls = []

    def swarm(task, ctx):
        calls.append(task.task_id)
        if len(calls) == 8:
            raise KeyboardInterrupt
        return "42" if len(calls) <= 5 else "no"

    return no_escape(evaluate, swarm, five_tasks(), trials=2, clock_factory=SimClock)


def test_interrupted_header_and_per_trial_success_are_honest():
    """PR #8 review: the header claimed "5 tasks x 2 trials = 7 runs" and the
    partial trial's success rate was shown as if it were a full trial."""
    res = interrupted_in_second_trial()
    report = res.report()
    header = next(l for l in report.splitlines() if "·" in l)
    assert "= 7 runs" not in header and "7/10 runs" in header
    assert res.performance.per_trial == [1.0]  # the partial trial is excluded
    assert "per-trial success: 100.0%  (incomplete trial excluded)" in report


# -- audit 2: a streamed call is not recorded as a $0 call -------------------- #
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
def test_streamed_call_is_passed_through_unrecorded(is_async):
    """The audit's fix: a stream has no usage yet, so recording it would
    store 0 tokens at $0.00.  Pass it through and record nothing."""
    stream, client = streaming_client(is_async)
    ctx = started_ctx()
    with ctx.agent("a"):
        out = client.messages.create(model="claude-sonnet-5", messages=[], stream=True)
        out = asyncio.run(out) if is_async else out
    assert out is stream
    assert ctx.finish().llm_calls() == []


@pytest.mark.parametrize("is_async", [False, True])
def test_stream_inside_llm_call_marks_its_cost_unknown(is_async):
    """PR #8 review: a stream inside ``span.llm_call`` must not leave that
    call priced at $0.00 as if complete; its cost is unknown."""
    stream, client = streaming_client(is_async)
    ctx = started_ctx()
    with ctx.agent("a") as span:
        with span.llm_call(model="claude-sonnet-5"):
            out = client.messages.create(model="claude-sonnet-5", messages=[], stream=True)
            out = asyncio.run(out) if is_async else out
    assert out is stream
    traj = ctx.finish()
    (call,) = traj.llm_calls()
    assert call.cost_usd is None and not traj.cost_is_complete
    assert not compute_efficiency(traj, PricingTable()).cost_complete  # re-pricing keeps it unknown


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


def test_evaluator_error_is_flagged_explicitly():
    """PR #8 review: a crashed evaluator is marked by a flag, not only by the
    wording of its reason; runs saved before the flag still count."""
    res = evaluate(lambda t, c: "42", five_tasks(), evaluator=E.CallableEvaluator(broken), clock_factory=SimClock)
    assert all(t.evaluation.details["evaluator_error"] is True for t in res.runs.trajectories)
    for t in res.runs.trajectories:
        t.evaluation.details["reason"] = "crashed"
    assert analyze_runs(res.runs).performance.evaluator_errors == 5
    for t in res.runs.trajectories:  # as saved before the flag existed
        t.evaluation.details = {"reason": "evaluator error: typo"}
    assert analyze_runs(res.runs).performance.evaluator_errors == 5


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


def test_box_keeps_value_and_flag_after_a_long_label():
    """PR #8 review: a label past 28 characters pushed the value off the line."""
    label = "Cost / Successful Task (all agents)"
    out = box("SWARM EVALUATION", [(label, "$0.1234", "⚠")])
    assert label in out and "$0.1234" in out and "⚠" in out
    assert len({len(line) for line in out.splitlines()}) == 1
