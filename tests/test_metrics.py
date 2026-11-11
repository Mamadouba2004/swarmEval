"""Hand-built trajectories with known answers for the metric layer."""
import pytest

from swarmeval import RunContext, SimClock, Task, compute_efficiency, compute_performance
from swarmeval.evaluate import analyze_runs
from swarmeval.runner import RunSet
from swarmeval.schema import Evaluation


def fan_out_trajectory(parallel=True):
    """coordinator -> (a, b, c) -> synthesizer, with known token counts and timings."""
    clock = SimClock()
    ctx = RunContext(Task("t", "question"), clock=clock, seed=0)
    ctx.start()
    with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as co:
        with co.llm_call() as call:
            clock.advance(1.0)
            call.input_tokens, call.output_tokens = 100, 50
        for w in ("a", "b", "c"):
            co.send(w, f"research topic {w}", kind="task_assignment")
    t0 = clock.now()
    ends = []
    arts = {}
    for w in ("a", "b", "c"):
        if parallel:
            clock.set(t0)
        with ctx.agent(w, role="research", model="claude-sonnet-5") as ag:
            ag.receive()
            ag.call_tool("search", lambda q: (clock.advance(0.5), f"result {q}")[1], q="topic")
            with ag.llm_call() as call:
                clock.advance(2.0)
                call.input_tokens, call.output_tokens = 200, 100
            arts[w] = ag.produce(f"notes_{w}", "notes " * 20)
            ag.send("synthesizer", "notes", kind="result", artifacts=[arts[w]])
        ends.append(clock.now())
    if parallel:
        clock.set(max(ends))
    with ctx.agent("synthesizer", role="synthesizer", model="claude-opus-5") as sy:
        sy.receive()
        sy.wait("think", 0.5)
        with sy.llm_call(input_artifacts=list(arts.values())) as call:
            clock.advance(3.0)
            call.input_tokens, call.output_tokens, call.output = 5000, 300, "final answer"
        final = sy.produce("final", "final answer")
    ctx.set_output("final answer", artifacts=[final])
    return ctx.finish()


def test_efficiency_counts_tokens_calls_and_cost():
    eff = compute_efficiency(fan_out_trajectory())
    assert eff.llm_calls == 5 and eff.tool_calls == 3 and eff.messages == 6
    assert eff.input_tokens == 100 + 3 * 200 + 5000
    assert eff.output_tokens == 50 + 3 * 100 + 300
    assert eff.total_tokens == eff.input_tokens + eff.output_tokens
    expected_cost = (100 * 5 + 50 * 25) / 1e6 + 3 * (200 * 2 + 100 * 10) / 1e6 + (5000 * 5 + 300 * 25) / 1e6
    assert eff.cost_usd == pytest.approx(expected_cost) and eff.cost_complete
    assert eff.per_agent["synthesizer"].input_tokens == 5000
    assert eff.per_agent["a"].tool_calls == 1 and eff.per_agent["a"].messages_sent == 1
    assert eff.per_agent["coordinator"].messages_sent == 3
    assert eff.per_agent["synthesizer"].artifacts_produced == 1


def test_efficiency_timing_parallel_vs_sequential():
    par = compute_efficiency(fan_out_trajectory(parallel=True))
    seq = compute_efficiency(fan_out_trajectory(parallel=False))
    assert par.wall_clock_s == pytest.approx(7.0)
    assert seq.wall_clock_s == pytest.approx(1.0 + 3 * 2.5 + 3.5)
    assert par.agent_time_s == pytest.approx(seq.agent_time_s) == pytest.approx(1.0 + 7.5 + 3.5)
    assert par.waiting_time_s == pytest.approx(0.5)
    assert par.per_agent["synthesizer"].busy_time_s == pytest.approx(3.0)


def test_nested_span_self_time_excludes_children():
    clock = SimClock()
    ctx = RunContext(Task("t", "q"), clock=clock)
    ctx.start()
    with ctx.agent("parent") as p:
        clock.advance(1.0)
        with p.sub_agent("child") as c:
            with c.llm_call() as call:
                clock.advance(2.0)
                call.input_tokens = 1
        clock.advance(1.0)
    eff = compute_efficiency(ctx.finish())
    assert eff.per_agent["parent"].self_time_s == pytest.approx(2.0)
    assert eff.per_agent["child"].self_time_s == pytest.approx(2.0)
    assert eff.agent_time_s == pytest.approx(4.0)


def _traj(task_id, trial, success, score, status="success", constraints=None):
    t = fan_out_trajectory()
    t.task_id = task_id
    t.task = Task(task_id, "q", category="research")
    t.trial = trial
    t.status = status
    t.evaluation = Evaluation(success, score, "contains_all", constraints_passed=constraints)
    return t


def test_performance_aggregation():
    trajs = [_traj("t1", 0, True, 1.0, constraints=True), _traj("t2", 0, False, 0.5, constraints=False),
             _traj("t1", 1, True, 1.0, constraints=True), _traj("t2", 1, True, 1.0, status="error", constraints=True)]
    pm = compute_performance(trajs)
    assert pm.n_runs == 4 and pm.n_tasks == 2 and pm.trials == 2
    assert pm.n_successes == 3 and pm.success_rate == pytest.approx(0.75)
    assert pm.mean_score == pytest.approx(0.875)
    assert pm.constraint_pass_rate == pytest.approx(0.75)
    assert pm.run_errors == 1
    assert pm.per_task["t1"]["success_rate"] == 1.0 and pm.per_task["t2"]["success_rate"] == 0.5
    assert pm.per_trial == [0.5, 1.0]
    assert pm.evaluators == ["contains_all"]


def test_analyze_runs_summary_and_cost_per_success():
    trajs = [_traj("t1", 0, True, 1.0), _traj("t2", 0, False, 0.0)]
    res = analyze_runs(RunSet("lbl", "bench", trajs))
    assert res.success_rate == 0.5
    assert res.cost["total"] == pytest.approx(sum(t.cost_usd for t in trajs))
    assert res.cost["per_success"] == pytest.approx(res.cost["total"])
    assert res.agents == ["coordinator", "a", "b", "c", "synthesizer"]
    s = res.summary()
    assert s["runs"] == 2 and s["label"] == "lbl" and s["tasks"] == 2
    text = res.report()
    assert "SWARM EVALUATION" in text and "synthesizer" in text and "t2" in text
