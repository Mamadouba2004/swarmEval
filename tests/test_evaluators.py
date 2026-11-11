"""Tests for the deterministic evaluators and their JSON spec round trip."""
import pytest

from swarmeval import RunContext, SimClock, Task
from swarmeval import evaluators as E


def T(**kw):
    return Task("t", "input", **kw)


def test_contains_all_and_forbidden():
    ev = E.ContainsAll(["alpha", "beta"], forbidden=["gamma"])
    r = ev.evaluate(T(), "Alpha and BETA are here")
    assert r.success and r.score == 1.0
    r = ev.evaluate(T(), "alpha only")
    assert not r.success and r.score == pytest.approx(0.5) and r.details["missing"] == ["beta"]
    r = ev.evaluate(T(), "alpha beta gamma")
    assert not r.success and r.constraints_passed is False


def test_contains_all_uses_task_expected_when_no_keywords():
    assert E.ContainsAll().evaluate(T(expected=["a", "b"]), "a and b").success
    assert E.ContainsAny().evaluate(T(expected=["a", "b"]), "only b").success


def test_exact_regex_numeric():
    assert E.ExactMatch().evaluate(T(expected="Paris"), " paris ").success
    assert E.Regex(r"\bok\b").evaluate(T(), "all OK here").success
    assert E.NumericTolerance(tolerance=0.01).evaluate(T(expected=3.14), "pi is about 3.141").success
    assert not E.NumericTolerance().evaluate(T(expected=3), "no numbers").success


def test_composite_and_constraints_and_spec_roundtrip():
    spec = {"type": "composite", "threshold": 0.6, "required": ["regex"],
            "evaluators": [{"type": "contains_all", "keywords": ["a", "b"]}, {"type": "regex", "pattern": "done"}]}
    ev = E.from_spec(spec)
    assert isinstance(ev, E.Composite)
    assert ev.evaluate(T(), "a b done").success
    assert not ev.evaluate(T(), "a b").success
    assert E.from_spec(ev.to_spec()).evaluate(T(), "a b done").success
    checks = [{"type": "regex", "pattern": "x"}, {"type": "regex", "pattern": "y"}]
    cons = E.from_spec({"type": "constraints", "checks": checks})
    assert cons.evaluate(T(), "xy").constraints_passed is True
    assert cons.evaluate(T(), "x").constraints_passed is False


def test_tool_use_evaluator_reads_trajectory():
    ctx = RunContext(Task("t", "q"), clock=SimClock())
    ctx.start()
    with ctx.agent("a") as a:
        a.call_tool("search", lambda: "ok")
    traj = ctx.finish()
    assert E.ToolUseEvaluator(expected_tools=["search"]).evaluate(T(), "out", traj).success
    r = E.ToolUseEvaluator(expected_tools=["fetch"], forbidden_tools=["search"]).evaluate(T(), "out", traj)
    assert not r.success and r.details["missing"] == ["fetch"] and r.details["forbidden"] == ["search"]


def test_callable_and_unknown_spec():
    ev = E.from_spec(lambda task, out, traj: out == "yes")
    assert ev.evaluate(T(), "yes").success
    ev2 = E.from_spec(lambda task, out, traj: 0.8)
    assert ev2.evaluate(T(), "x").success and ev2.evaluate(T(), "x").score == 0.8
    with pytest.raises(ValueError):
        E.from_spec({"type": "nope"})


def test_task_spec_serialises_evaluator_instances():
    t = Task("t", "q", evaluator=E.Regex("done"))
    d = t.to_dict()
    assert d["evaluator"] == {"type": "regex", "pattern": "done", "flags": E.Regex.DEFAULT_FLAGS}
    assert E.from_spec(Task.from_dict(d).evaluator).evaluate(t, "done").success
