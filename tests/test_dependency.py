"""Critical path, parallelism and agent-level cycles (metrics/dependency.py)."""
import asyncio

import pytest

from swarmeval import RunContext, SimClock, Task, compute_coordination
from swarmeval.evaluate import analyze_runs, load_runset
from swarmeval.graph.builder import build_graph
from swarmeval.metrics.dependency import compute_dependency, reply_flags
from swarmeval.schema import EventType

from test_metrics import fan_out_trajectory

FIXTURE = "tests/fixtures/phase1_runs.json"


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def llm(span, clock, seconds):
    with span.llm_call() as call:
        clock.advance(seconds)
        call.input_tokens, call.output_tokens = 10, 5


def smells(cm):
    return {d.smell for d in cm.diagnostics}


def as_phase1(traj):
    """Strip what Phase 1 recorders did not write: receives and the schema version."""
    traj.events = [e for e in traj.events if e.event_type != EventType.MESSAGE_RECEIVE.value]
    traj.metadata.pop("schema_version", None)
    return traj


def test_fan_out_same_critical_path_parallel_or_serial():
    par = compute_coordination(fan_out_trajectory(parallel=True))
    ser = compute_coordination(fan_out_trajectory(parallel=False))
    # coordinator 1.0 -> one worker (search 0.5 + llm 2.0) -> synthesizer llm 3.0 (its 0.5 wait weighs 0)
    for cm in (par, ser):
        assert cm.dependency.critical_path_s == pytest.approx(6.5)
        assert cm.dependency.work_s == pytest.approx(1.0 + 3 * 2.5 + 3.0)
        assert cm.dependency.depth == 4
        assert cm.dependency.potential_parallelism == pytest.approx(11.5 / 6.5)
    assert par.dependency.wall_s == pytest.approx(7.0)
    assert ser.dependency.wall_s == pytest.approx(1.0 + 3 * 2.5 + 3.5)    # serial wall ≈ sum
    assert par.dependency.actual_parallelism == pytest.approx(11.5 / 7.0)
    assert ser.dependency.headroom_s == pytest.approx(12.0 - 6.5)
    assert ser.dependency.headroom_ratio == pytest.approx(12.0 / 6.5)
    assert ser.dependency.unrecorded_s == pytest.approx(0.0)
    cp = par.dependency.critical_path
    assert set(cp.per_agent_s) == {"coordinator", "a", "synthesizer"}         # earliest worker on ties
    assert "serialized_execution" in smells(ser) and "serialized_execution" not in smells(par)
    d = next(d for d in ser.diagnostics if d.smell == "serialized_execution")
    assert d.severity == "medium" and not d.inferred and d.evidence["ratio"] == pytest.approx(12.0 / 6.5)


def test_serialized_execution_is_low_and_inferred_without_receive_events():
    cm = compute_coordination(as_phase1(fan_out_trajectory(parallel=False)))
    assert cm.coverage.message_receive_coverage == 0.0
    d = [d for d in cm.diagnostics if d.smell == "serialized_execution"]
    assert d and d[0].severity == "low" and d[0].inferred


def test_gathered_calls_count_once_on_the_critical_path():
    ctx = make_ctx()
    clock = ctx.clock

    async def model(seconds):
        clock.advance(seconds)
        return "out"

    async def main():
        async with ctx.agent("a") as a:
            await ctx.gather(a.allm(model, 1.0), a.allm(model, 2.0))
            await a.allm(model, 0.5)

    asyncio.run(main())
    dep = compute_dependency(ctx.finish())
    assert dep.critical_path_s == pytest.approx(2.0 + 0.5)       # max of the gathered calls, not their sum
    assert dep.work_s == pytest.approx(3.5) and dep.wall_s == pytest.approx(2.5)
    assert dep.actual_parallelism == pytest.approx(3.5 / 2.5) and dep.depth == 2


def test_nested_calls_are_not_double_counted_as_work():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        def tool():
            clock.advance(0.5)
            llm(a, clock, 2.0)
            return "r"
        a.call_tool("agentic", tool)
    dep = compute_dependency(ctx.finish())
    assert dep.work_s == pytest.approx(2.5) and dep.critical_path_s == pytest.approx(2.5)


def test_unrecorded_idle_time_is_reported_not_read_as_headroom():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        llm(a, clock, 1.0)
        clock.advance(4.0)              # untracked: no call, no wait
        llm(a, clock, 1.0)
    cm = compute_coordination(ctx.finish())
    dep = cm.dependency
    assert dep.critical_path_s == pytest.approx(2.0) and dep.headroom_s == pytest.approx(4.0)
    assert dep.unrecorded_s == pytest.approx(4.0)
    assert "serialized_execution" not in smells(cm)


def test_request_response_is_not_a_cycle_but_three_agents_in_a_ring_are():
    ctx = make_ctx()
    with ctx.agent("boss") as b:
        req = b.send("w", "do x", kind="request")
    with ctx.agent("w") as w:
        w.receive()
        w.send("boss", "done", kind="result", reply_to=req)
    with ctx.agent("boss") as b:
        b.receive()
        b.send("w", "do y", kind="task_assignment")
    with ctx.agent("w") as w:
        w.receive()
        w.send("boss", "done y", kind="result")              # answers the assignment without reply_to
    traj = ctx.finish()
    assert [r for _, r in reply_flags(traj)] == [False, True, False, True]
    assert compute_dependency(traj).cycles == []

    ctx = make_ctx()
    for src, dst in (("a", "b"), ("b", "c"), ("c", "a")):
        with ctx.agent(src) as s:
            s.receive()
            s.send(dst, f"{src} tells {dst}", kind="information")
    dep = compute_dependency(ctx.finish())
    assert dep.cycles == [["a", "b", "c"]]


def test_isolated_agents():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "hi")
    with ctx.agent("b") as b:
        b.receive()
    with ctx.agent("loner") as x:
        llm(x, ctx.clock, 1.0)
    assert compute_dependency(ctx.finish()).isolated_agents == ["loner"]


def test_phase1_fixture_dependency_and_report():
    res = analyze_runs(load_runset(FIXTURE))
    for cm in res.per_run_coordination:
        dep = cm.dependency
        assert dep.critical_path_s > 0 and dep.inferred_edges_on_path > 0
        assert dep.potential_parallelism >= 1.0 and dep.cycles == []
        for d in cm.diagnostics:
            if d.smell == "serialized_execution":
                assert d.severity == "low" and d.inferred
    text = res.report()
    assert "COORDINATION" in text and "inferred from timing" in text
    assert "dependencies may be over- or under-counted" in text
    assert res.coordination.dependency.critical_path_s.n == len(res.runs)


def test_critical_path_is_deterministic_on_the_demo():
    from swarmeval import evaluate
    from swarmeval.examples.research_swarm import build_benchmark, make_swarm
    a = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    b = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    for x, y in zip(a.per_run_coordination, b.per_run_coordination):
        gx, gy = x.graph, y.graph
        # zero-weight nodes tied in time (receives of concurrent results) may be recorded in either
        # order; the working units on the path, and its length, may not
        kinds = lambda cm, g: [(g.unit(n).agent_id, g.unit(n).kind) for n in cm.dependency.critical_path.node_ids
                               if g.unit(n).kind in ("llm", "tool")]
        assert kinds(x, gx) == kinds(y, gy)
        assert x.dependency.critical_path_s == pytest.approx(y.dependency.critical_path_s)
    dep = a.per_run_coordination[0].dependency
    # the demo is a chain: coordinator -> researcher -> coordinator -> critic -> coordinator -> synthesizer
    assert dep.headroom_s == pytest.approx(0.0, abs=1e-6) and dep.unrecorded_s == pytest.approx(0.0, abs=1e-6)
    assert set(dep.critical_path.per_agent_s) >= {"coordinator", "critic", "synthesizer"}
    assert build_graph(a.runs.trajectories[0]).dag_units  # graph rebuilt from the trajectory alone
