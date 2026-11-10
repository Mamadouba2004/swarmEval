"""Execution graph: work-unit DAG rules and the agent graph."""
import asyncio
import json
from collections import defaultdict

from swarmeval import RunContext, SimClock, Task, build_graph, evaluate
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics import compute_contribution
from swarmeval.schema import EventType


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def llm(span, clock, seconds, **kw):
    with span.llm_call(**kw) as call:
        clock.advance(seconds)
        call.input_tokens, call.output_tokens = 10, 5
    return call


def node_of(traj, pred):
    return next(e.event_id for e in traj.events if pred(e))


def is_acyclic(g):
    indeg = defaultdict(int)
    for e in g.edges:
        indeg[e.dst] += 1
    ready = [u.node_id for u in g.units if indeg[u.node_id] == 0]
    seen = 0
    while ready:
        n = ready.pop()
        seen += 1
        for s in g.succs(n):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    return seen == len(g.units)


def test_sequential_calls_are_chained():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        c1 = llm(a, clock, 1.0, label="1")
        c2 = llm(a, clock, 1.0, label="2")
        c3 = llm(a, clock, 1.0, label="3")
    g = build_graph(ctx.finish())
    ids = [e.event_id for e in g.units if e.kind == "llm"]
    assert len(ids) == 3
    seq = {(e.src, e.dst) for e in g.edges_of("sequence")}
    assert seq == {(ids[0], ids[1]), (ids[1], ids[2])}
    assert g.preds(ids[2]) == [ids[1]]


def test_gathered_calls_are_not_chained_and_the_next_call_depends_on_both():
    ctx = make_ctx()
    clock = ctx.clock

    async def model(seconds):
        clock.advance(seconds)
        return "out"

    async def main():
        async with ctx.agent("a") as a:
            await ctx.gather(a.allm(model, 1.0), a.allm(model, 2.0))   # [0,1] and [0,2]: one looks nested
            await a.allm(model, 0.5)

    asyncio.run(main())
    traj = ctx.finish()
    g = build_graph(traj)
    calls = sorted((u for u in g.units if u.kind == "llm"), key=lambda u: (u.start, u.end))
    short, long_, after = calls
    assert short.nested_in is None and long_.nested_in is None
    assert (short.start, short.end, long_.end) == (0.0, 1.0, 2.0) and after.start == 2.0
    pairs = {(e.src, e.dst) for e in g.edges}
    assert (short.node_id, long_.node_id) not in pairs and (long_.node_id, short.node_id) not in pairs
    assert set(g.preds(after.node_id)) == {short.node_id, long_.node_id}


def test_tool_with_nested_llm_is_not_double_counted():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        def tool():
            clock.advance(0.5)
            llm(a, clock, 2.0)
            clock.advance(0.5)
            return "r"
        a.call_tool("agentic_tool", tool)
        llm(a, clock, 1.0)
    g = build_graph(ctx.finish())
    tool_u = next(u for u in g.units if u.kind == "tool")
    inner, after = sorted((u for u in g.units if u.kind == "llm"), key=lambda u: u.start)
    assert inner.nested_in == tool_u.node_id and inner.start == 0.5
    assert all(inner.node_id not in (e.src, e.dst) for e in g.edges)
    assert inner not in g.dag_units
    assert sum(u.weight for u in g.dag_units) == 3.0 + 1.0     # tool (3s, incl. inner) + following call
    assert g.preds(after.node_id) == [tool_u.node_id]


def test_phase1_style_nesting_falls_back_to_intervals():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        with a.tool_call("outer"):
            clock.advance(0.5)
            llm(a, clock, 1.0)
    traj = ctx.finish()
    del traj.metadata["schema_version"]
    for e in traj.events:
        e.metadata.pop("within_call", None)
    g = build_graph(traj)
    tool_u = next(u for u in g.units if u.kind == "tool")
    assert next(u for u in g.units if u.kind == "llm").nested_in == tool_u.node_id


def _send_receive(record_receive=True):
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        llm(a, clock, 1.0)
        a.send("b", "go")
    clock.advance(1.0)
    with ctx.agent("b") as b:
        llm(b, clock, 0.5)          # unrelated work before reading the message
        if record_receive:
            b.receive()
        llm(b, clock, 1.0)
    return ctx.finish()


def test_recorded_receive_gives_a_real_message_edge():
    traj = _send_receive()
    g = build_graph(traj)
    (e,) = g.edges_of("message")
    assert not e.inferred and g.unit(e.dst).kind == "receive"
    assert e.ref == traj.events_of(EventType.MESSAGE)[0].message_id
    later = [u for u in g.units if u.kind == "llm" and u.agent_id == "b"][1]
    assert e.dst in g.preds(later.node_id)
    assert g.coverage.message_receive_coverage == 1.0
    (link,) = g.messages
    assert link.received and link.used and link.used_inferred   # no artifacts: use judged from timing


def test_without_receive_the_message_edge_is_inferred_to_the_first_unit_after_the_send():
    traj = _send_receive(record_receive=False)
    del traj.metadata["schema_version"]         # Phase 1 data: receives were never recorded
    g = build_graph(traj)
    (e,) = g.edges_of("message")
    first_b = next(u for u in g.units if u.agent_id == "b" and u.kind == "llm")
    assert e.inferred and e.dst == first_b.node_id
    assert g.coverage.message_receive_coverage == 0.0
    edge = next(x for x in g.agent_edges if x.kind == "message")
    assert (edge.source, edge.target, edge.count, edge.inferred_count) == ("a", "b", 1, 1)


def test_schema_v2_message_without_receive_is_not_inferred_as_delivered():
    """v2 records every receive, so a message without one never arrived."""
    traj = _send_receive(record_receive=False)
    g = build_graph(traj)
    assert not g.edges_of("message")
    (link,) = g.messages
    assert link.receive_node is None and not link.received and not link.used
    assert compute_contribution(traj, g)["a"].status == "unused"


def test_forwarding_an_artifact_is_not_consuming_it():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        art = a.produce("notes", "facts", derived_from=[])
        a.send("hub", "notes", artifacts=[art])
    with ctx.agent("hub") as hub:
        (m,) = hub.receive()
        llm(hub, clock, 0.5)                      # routing decision, does not read the notes
        hub.send("b", m.content, artifacts=m.artifacts)
    with ctx.agent("b") as b:
        b.receive()
        b.consume(art)
        llm(b, clock, 1.0, input_artifacts=[art])
    g = build_graph(ctx.finish())
    assert g.artifact_consumers[art.artifact_id] == ["b"]
    produce = next(u for u in g.units if u.kind == "produce")
    art_dsts = {g.unit(e.dst).agent_id for e in g.edges_of("artifact")}
    assert art_dsts == {"b"} and all(e.src == produce.node_id for e in g.edges_of("artifact"))
    kinds = {(e.source, e.target, e.kind) for e in g.agent_edges}
    assert ("a", "b", "artifact") in kinds and ("a", "hub", "artifact") not in kinds
    a_to_hub = next(m for m in g.messages if m.sender == "a")
    hub_to_b = next(m for m in g.messages if m.sender == "hub")
    assert a_to_hub.used_inferred                  # hub's call may have read the text, not the notes
    assert hub_to_b.used and not hub_to_b.used_inferred


def test_invocation_edges_link_parent_frontier_child_and_resume():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("boss") as boss:
        llm(boss, clock, 1.0)
        with boss.sub_agent("worker") as w:
            llm(w, clock, 2.0)
        llm(boss, clock, 1.0)
    g = build_graph(ctx.finish())
    b1, b2 = [u for u in g.units if u.agent_id == "boss" and u.kind == "llm"]
    (wk,) = [u for u in g.units if u.agent_id == "worker"]
    inv = {(e.src, e.dst) for e in g.edges_of("invocation")}
    assert inv == {(b1.node_id, wk.node_id), (wk.node_id, b2.node_id)}
    assert g.preds(b2.node_id) == [wk.node_id]          # not chained to b1 directly: the child sits between
    assert any(e.kind == "invocation" and (e.source, e.target) == ("boss", "worker") for e in g.agent_edges)
    assert g.fan_out("boss") == 0 and g.fan_in("worker") == 0      # invocation is not communication fan


def test_wait_on_links_what_was_awaited():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        art = a.produce("notes", "x")
        msg = a.send("b", "x")
    with ctx.agent("b") as b:
        b.wait("inputs", 1.0, on=[art, msg])
    g = build_graph(ctx.finish())
    wait = next(u for u in g.units if u.kind == "wait")
    refs = {e.ref for e in g.edges_of("wait") if e.dst == wait.node_id}
    assert refs == {art.artifact_id, msg.message_id}


def test_demo_graph_is_acyclic_fully_recorded_and_serialisable():
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    for cm in res.per_run_coordination:
        g = cm.graph
        assert is_acyclic(g) and g.dropped_edges == 0
        assert g.coverage.message_receive_coverage == 1.0 and g.coverage.derivation_coverage == 1.0
        assert not any(e.inferred for e in g.edges)
        # researchers consume no artifact: derived_from=[] is no evidence they record use
        instrumented = {a for a, v in g.coverage.consumers_instrumented.items() if v}
        assert instrumented == {"coordinator", "critic", "synthesizer"}
        json.dumps(g.to_dict())
        # the coordinator relays everything: it talks to every other agent
        assert g.fan_out("coordinator") == 5 and g.fan_in("coordinator") == 4
