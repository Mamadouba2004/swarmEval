"""Communication overhead: body tokens, Communication / Work, kinds, repeats,
broadcasts, hub share, relay fraction, message usage."""
import json

import pytest

from swarmeval import RunContext, SimClock, Task, compute_communication, evaluate
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics import summarize_communication
from swarmeval.metrics.communication import body_tokens


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def llm(span, clock, seconds=1.0, tokens=(10, 5), **kw):
    with span.llm_call(**kw) as call:
        clock.advance(seconds)
        call.input_tokens, call.output_tokens = tokens


def test_attached_artifact_tokens_counted_once():
    """A 5-token body carrying a 1000-token artifact is 5 tokens of
    communication: the artifact's tokens belong to its production."""
    ctx = make_ctx()
    with ctx.agent("a") as a:
        llm(a, ctx.clock, tokens=(100, 50))
        art = a.produce("big", "x" * 4000)            # 1000 tokens
        a.send("b", "y" * 20, kind="result", artifacts=[art])   # 5-token body
    with ctx.agent("b") as b:
        b.receive()
        llm(b, ctx.clock, tokens=(40, 10))
    traj = ctx.finish()
    ev = traj.messages()[0]
    assert ev.content_tokens == 1005 and body_tokens(ev, traj) == 5
    cm = compute_communication(traj)
    assert cm.body_tokens == 5 and cm.attached_artifact_tokens == 1000
    assert cm.llm_tokens == 200
    assert cm.comm_work_ratio == pytest.approx(5 / 200)
    assert cm.by_category["work_product"] == {"messages": 1, "body_tokens": 5}
    json.dumps(cm.to_dict())


def test_no_llm_calls_ratio_unknown():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "hello")
    cm = compute_communication(ctx.finish())
    assert cm.messages == 1 and cm.comm_work_ratio is None


def test_empty_run():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        llm(a, ctx.clock)
    cm = compute_communication(ctx.finish())
    assert cm.messages == 0 and cm.comm_work_ratio == 0.0
    assert cm.hub_agent is None and cm.hub_share is None and cm.used_edge_ratio is None
    assert cm.receive_delay_mean_s is None


def test_kind_split_repeats_and_broadcasts():
    ctx = make_ctx()
    with ctx.agent("lead") as lead:
        lead.send("w1", "start on part one", kind="task_assignment")
        lead.send("w1", "status?", kind="coordination")
        lead.send("w1", "status?", kind="coordination")       # repeat: same sender, receiver, content
        lead.send("w2", "where is it?", kind="request")
        lead.send("w2", "fyi the deadline moved", kind="information")
        lead.send("w3", "fyi the deadline moved", kind="information")   # broadcast with the one above
        lead.send("w1", "nice work", kind="feedback")
        lead.send("w1", "misc", kind="chatter")                 # unknown kind -> other
    with ctx.agent("w1") as w1:
        w1.send("lead", "part one done", kind="result")
    cm = compute_communication(ctx.finish())
    cats = {k: v["messages"] for k, v in cm.by_category.items()}
    assert cats == {"coordination": 4, "information": 2, "work_product": 2, "other": 1}
    assert cm.repeated_messages == 1
    assert cm.broadcasts == 1 and cm.broadcast_messages == 2
    assert cm.per_agent["lead"].messages_sent == 8 and cm.per_agent["w1"].messages_received == 5
    assert cm.per_agent["lead"].fan_out == 3 and cm.per_agent["lead"].fan_in == 1


def test_same_text_sent_outside_spans_is_not_one_broadcast():
    ctx = make_ctx()
    ctx.send("lead", "w1", "ping")
    ctx.clock.advance(5.0)
    ctx.send("lead", "w2", "ping")             # later, unrelated: no span ties the two together
    cm = compute_communication(ctx.finish())
    assert cm.broadcasts == 0 and cm.broadcast_messages == 0


def test_star_topology_hub_share_and_relay():
    """Every message goes through the hub; the hub forwards a's artifact."""
    ctx = make_ctx()
    with ctx.agent("hub") as hub:
        for w in ("a", "b", "c"):
            hub.send(w, f"task for {w}", kind="task_assignment")
    with ctx.agent("a") as a:
        a.receive()
        llm(a, ctx.clock)
        notes = a.produce("notes", "a's findings", derived_from=[])
        a.send("hub", "done", kind="result", artifacts=[notes])
    with ctx.agent("hub") as hub:
        hub.receive()
        hub.send("b", "here are a's notes", kind="information", artifacts=[notes])
    with ctx.agent("b") as b:
        b.receive()
        b.consume(notes)
        llm(b, ctx.clock)
    cm = compute_communication(ctx.finish())
    assert cm.hub_agent == "hub" and cm.hub_share == 1.0
    assert cm.per_agent["hub"].relayed_messages == 1
    assert cm.per_agent["hub"].relay_fraction == pytest.approx(1 / 4)
    assert cm.per_agent["a"].relay_fraction == 0.0          # sent its own artifact
    assert cm.per_agent["c"].relay_fraction is None         # sent nothing
    # a -> b through artifact consumption counts towards fan-out/in
    assert cm.per_agent["a"].fan_out == 2 and cm.per_agent["b"].fan_in == 2
    assert cm.receive_delay_mean_s is not None and cm.receive_delay_max_s >= cm.receive_delay_mean_s


def test_used_edge_ratio_and_inferred():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("notes", "facts", derived_from=[])
        a.send("b", "use these", artifacts=[art])          # used: b consumes the artifact
        a.send("c", "ignore me", artifacts=[art])          # delivered, artifact never used
        a.send("b", "plain text")                          # no artifact: used by timing (inferred)
    with ctx.agent("b") as b:
        b.receive()
        b.consume(art)
        llm(b, ctx.clock)
    with ctx.agent("c") as c:
        c.receive()
    cm = compute_communication(ctx.finish())
    assert cm.used_messages == 2 and cm.used_inferred == 1
    assert cm.used_edge_ratio == pytest.approx(2 / 3)


def test_demo_communication():
    """Demo: 9 messages, all to or from the coordinator (hub share 1.0); the
    coordinator relays the researchers' notes to the critic and the synthesizer
    (2 of its 5 messages); researcher_c's result is the one message nobody uses.
    The three task assignments carry no artifact, so their use is inferred."""
    res = evaluate(make_swarm, build_benchmark(2), trials=3, clock_factory=SimClock)
    for traj, cm in zip(res.runs.trajectories, res.per_run_coordination):
        c = cm.communication
        assert c.messages == 9 and c.hub_agent == "coordinator" and c.hub_share == 1.0
        assert {k: v["messages"] for k, v in c.by_category.items()} == {
            "coordination": 5, "information": 0, "work_product": 4, "other": 0}
        assert c.per_agent["coordinator"].relay_fraction == pytest.approx(2 / 5)
        assert c.used_messages == 8 and c.used_edge_ratio == pytest.approx(8 / 9) and c.used_inferred == 3
        # researcher_a and researcher_c get the identical assignment (same
        # scope) from one coordinator span: one broadcast of two messages
        assert c.repeated_messages == 0 and c.broadcasts == 1 and c.broadcast_messages == 2
        # by hand: message bodies over LLM prompt+output tokens
        body = sum(max(0, m.content_tokens - sum(traj.artifacts[a].tokens for a in m.input_artifacts))
                   for m in traj.messages())
        work = sum(e.input_tokens + e.output_tokens for e in traj.llm_calls())
        assert c.comm_work_ratio == pytest.approx(body / work)
        assert 0.01 < c.comm_work_ratio < 0.02     # bodies are small; the relayed notes ride as artifacts
    s = summarize_communication([cm.communication for cm in res.per_run_coordination])
    assert s.hub_agents == {"coordinator": 6} and s.hub_share.mean == 1.0
    assert res.coordination.communication.comm_work_ratio.n == 6
    json.dumps(res.to_dict())
