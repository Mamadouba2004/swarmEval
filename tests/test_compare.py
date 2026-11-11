"""Tests for run-to-run comparison: metric deltas, paired flips, warnings and the compare CLI."""
import json

import pytest

from swarmeval import Benchmark, LLMResult, SimClock, Task, compare, evaluate
from swarmeval.cli import main
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.pricing import PricingTable


def bench(ids=("t1", "t2", "t3", "t4"), name="b"):
    return Benchmark(name, [Task(t, f"question {t}", expected="42", evaluator={"type": "exact_match"})
                            for t in ids])


def swarm(model="claude-opus-5", solves=("t1", "t2", "t3", "t4"), tokens=1000, agents=("solver",)):
    def run(task, ctx):
        for aid in agents:
            with ctx.agent(aid, model=model) as a:
                a.llm(lambda: LLMResult("x", tokens, tokens // 10))
                ctx.clock.advance(1.0)
        return "42" if task.task_id in solves else "no"
    return run


def run(label, **kw):
    b = kw.pop("benchmark", None) or bench()
    return evaluate(swarm(**kw), b, trials=3, clock_factory=SimClock, label=label)


def test_tradeoff_cheaper_but_less_accurate():
    base = run("opus", model="claude-opus-5")
    cand = run("sonnet", model="claude-sonnet-5", solves=("t1", "t2", "t3"))
    cmp = compare(base, cand)
    assert cmp.metric("success_rate").delta == pytest.approx(-0.25)
    assert cmp.metric("success_rate").verdict == "regressed"
    assert cmp.metric("cost_per_run").verdict == "improved"
    assert cmp.metric("tokens_per_run").verdict == "unchanged"
    assert cmp.is_tradeoff and "cost_per_run" in cmp.improved and "success_rate" in cmp.regressed
    assert cmp.paired_runs == 12 and len(cmp.flips["lost"]) == 3 and not cmp.flips["gained"]
    assert {t for t, _ in cmp.flips["lost"]} == {"t4"}
    assert cmp.per_task["t4"]["success_rate"] == (1.0, 0.0)
    text = cmp.report()
    assert "TRADEOFF" in text and "t4" in text and "3 pass→fail" in text


def test_identical_runs_show_no_difference():
    a, b = run("a"), run("b")
    cmp = compare(a, b)
    assert not cmp.improved and not cmp.regressed
    assert "no measurable difference" in cmp.report()


def test_only_shared_tasks_are_compared():
    base = run("base", benchmark=bench(("t1", "t2", "t3")))
    cand = run("cand", benchmark=bench(("t2", "t3", "t4")))
    cmp = compare(base, cand)
    assert cmp.shared_tasks == ["t2", "t3"] and cmp.only_baseline == ["t1"] and cmp.only_candidate == ["t4"]
    assert cmp.baseline.performance.n_runs == 6 and any("shared task" in w for w in cmp.warnings)
    with pytest.raises(ValueError, match="no task in common"):
        compare(run("x", benchmark=bench(("t1",))), run("y", benchmark=bench(("t9",))))


def test_unpriced_cost_is_not_compared_as_zero():
    base = run("base")
    cand = run("local", model="llama-3-70b")
    cmp = compare(base, cand)
    assert cmp.metric("cost_per_run").candidate is None and cmp.metric("cost_per_run").verdict == "unknown"
    assert any("incomplete" in w for w in cmp.warnings)
    repriced = compare(base, cand, pricing=PricingTable({"llama-3-70b": (0.5, 0.5)}))
    assert repriced.metric("cost_per_run").verdict == "improved"


def test_agents_added_and_removed():
    base = run("flat", agents=("solver",))
    cand = run("hier", agents=("planner", "solver"))
    cmp = compare(base, cand)
    assert cmp.per_agent["solver"]["status"] == "both"
    assert cmp.per_agent["planner"]["status"] == "added" and cmp.per_agent["planner"]["tokens"][0] is None
    assert cmp.metric("wall_clock_s").verdict == "regressed"


def test_few_trials_and_different_benchmarks_warn():
    a = evaluate(swarm(), bench(name="one"), trials=1, clock_factory=SimClock)
    b = evaluate(swarm(), bench(name="two"), trials=1, clock_factory=SimClock)
    warnings = " ".join(compare(a, b).warnings)
    assert "different benchmarks" in warnings and "fewer than 3 trials" in warnings


def test_compare_demo_swarm_against_itself_is_reproducible():
    a = evaluate(make_swarm, build_benchmark(2), trials=2, clock_factory=SimClock, label="a")
    b = evaluate(make_swarm, build_benchmark(2), trials=2, clock_factory=SimClock, label="b")
    cmp = compare(a, b)
    assert not cmp.flips["gained"] and not cmp.flips["lost"] and not cmp.regressed


def test_cli_compare(tmp_path, capsys):
    base, cand = run("base"), run("cand", solves=("t1",))
    pb, pc, out = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "cmp.json"
    base.save(str(pb))
    cand.runs.save(str(pc))
    assert main(["compare", str(pb), str(pc), "--out", str(out)]) == 0
    assert "COMPARISON  base → cand" in capsys.readouterr().out
    d = json.loads(out.read_text())
    assert d["summary"]["lost"] == 9 and "success_rate" in d["summary"]["regressed"]
