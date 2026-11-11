"""Tests for the recorder: spans, LLM and tool calls, messages, artifacts, errors and serialization."""
import threading
import time

import pytest

from swarmeval import LLMResult, RunContext, SimClock, Task, Trajectory, WallClock
from swarmeval.schema import EventType


def make_ctx(clock=None):
    return RunContext(Task("t1", "do the thing", expected="42"), clock=clock or SimClock(), seed=1)


def test_spans_llm_tool_message_artifact_are_recorded():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a", role="worker", model="claude-sonnet-5") as a:
        out = a.llm(lambda p: LLMResult("hello", 100, 20), "prompt", prompt="prompt")
        assert out == "hello"
        r = a.call_tool("calc", lambda x, y: x + y, 1, 2)
        assert r == 3
        art = a.produce("notes", "some notes here")
        msg = a.send("b", "take this", kind="result", artifacts=[art])
        assert msg.receiver == "b"
    with ctx.agent("b", role="writer") as b:
        msgs = b.receive()
        assert len(msgs) == 1 and msgs[0].artifacts == [art.artifact_id]
        assert b.receive() == []
        b.consume(art)
        with b.llm_call(model="claude-opus-5", input_artifacts=[art]) as call:
            ctx.clock.advance(1.5)
            call.input_tokens, call.output_tokens, call.output = 500, 50, "final"
    ctx.set_output("final", artifacts=[art])
    traj = ctx.finish()
    assert traj.status == "success"
    assert len(traj.spans) == 2 and traj.agent_ids() == ["a", "b"]
    llm = traj.llm_calls()
    assert [e.agent_id for e in llm] == ["a", "b"]
    assert llm[0].input_tokens == 100 and llm[0].output_tokens == 20 and llm[0].cost_usd is not None
    assert llm[1].latency_ms == pytest.approx(1500.0)
    assert llm[1].input_artifacts == [art.artifact_id]
    tool = traj.tool_calls()[0]
    assert tool.tool_name == "calc" and tool.tool_args == {"args": [1, 2]} and tool.content == 3
    m = traj.messages()[0]
    assert m.sender == "a" and m.receiver == "b" and m.kind == "result"
    assert art.artifact_id in traj.artifacts
    assert traj.total_tokens == 670
    assert traj.final_output == "final" and traj.output_artifacts == [art.artifact_id]
    types = [e.event_type for e in traj.events]
    assert types[0] == EventType.RUN_START.value and types[-1] == EventType.RUN_END.value


def test_receive_for_another_agent_inside_a_span_has_no_span_or_parent():
    ctx = make_ctx()
    ctx.send("c", "b", "x")
    with ctx.agent("a"):
        ctx.receive("b")
    (ev,) = ctx.finish().events_of(EventType.MESSAGE_RECEIVE)
    assert ev.agent_id == "b" and ev.span_id is None and ev.parent_event_id is None


def test_send_for_another_agent_inside_a_span_has_no_span_or_parent():
    ctx = make_ctx()
    with ctx.agent("a"):
        ctx.send("b", "c", "x")
    (ev,) = ctx.finish().events_of(EventType.MESSAGE)
    assert ev.agent_id == "b" and ev.span_id is None and ev.parent_event_id is None


def test_llm_without_usage_falls_back_to_estimates():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a", model="claude-haiku-4-5") as a:
        a.llm(lambda: "four words of output", prompt="a prompt of some length")
    ev = ctx.finish().llm_calls()[0]
    assert ev.input_tokens > 0 and ev.output_tokens > 0
    assert ev.metadata["input_tokens_estimated"] and ev.metadata["output_tokens_estimated"]
    assert ev.metadata["prompt"] == "a prompt of some length"


def test_unknown_model_has_unknown_cost():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a", model="some-other-model") as a:
        a.llm(lambda: LLMResult("x", 10, 10))
    traj = ctx.finish()
    assert traj.llm_calls()[0].cost_usd is None
    assert traj.cost_usd is None and not traj.cost_is_complete


def test_nested_spans_have_parent():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("coord") as c:
        with c.sub_agent("child") as ch:
            assert ch.parent is c
            assert ctx.current_span is ch
        assert ctx.current_span is c
    spans = list(ctx.finish().spans.values())
    child = next(s for s in spans if s.agent_id == "child")
    parent = next(s for s in spans if s.agent_id == "coord")
    assert child.parent_span_id == parent.span_id


def test_wait_and_annotations_are_recorded():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a") as a:
        a.wait("rate_limit", 2.0)
        a.note("checkpoint", step=1)
    traj = ctx.finish()
    w = traj.events_of(EventType.WAIT)[0]
    assert w.kind == "rate_limit" and w.latency_ms == pytest.approx(2000.0)
    assert traj.events_of(EventType.ANNOTATION)[0].metadata == {"step": 1}
    assert traj.duration == pytest.approx(2.0)


def test_parallel_propagates_span_and_wall_clock_overlaps():
    ctx = make_ctx(clock=WallClock())
    ctx.start()

    def worker(name):
        with ctx.agent(name) as w:
            with w.llm_call(model="claude-sonnet-5") as call:
                time.sleep(0.05)
                call.input_tokens = 10
            return threading.current_thread().name

    with ctx.agent("coord"):
        names = ctx.parallel(lambda: worker("w1"), lambda: worker("w2"))
    traj = ctx.finish()
    assert len(set(names)) == 2
    children = [s for s in traj.spans.values() if s.agent_id in ("w1", "w2")]
    parent = next(s for s in traj.spans.values() if s.agent_id == "coord")
    assert all(s.parent_span_id == parent.span_id for s in children)
    a, b = sorted(children, key=lambda s: s.start)
    assert b.start < a.end


def test_error_inside_span_is_recorded_and_reraised():
    ctx = make_ctx()
    ctx.start()
    with pytest.raises(ValueError):
        with ctx.agent("a") as a:
            with a.llm_call(model="m"):
                raise ValueError("boom")
    traj = ctx.finish("error", "boom")
    assert traj.llm_calls()[0].status == "error"
    assert any(e.event_type == EventType.ERROR.value for e in traj.events)
    assert list(traj.spans.values())[0].status == "error"
    assert traj.status == "error"


def test_tool_error_is_recorded_and_reraised():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a") as a:
        with pytest.raises(RuntimeError):
            a.call_tool("flaky", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    ev = ctx.finish().tool_calls()[0]
    assert ev.status == "error" and "RuntimeError" in ev.error


def test_trajectory_roundtrip(tmp_path):
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a") as a:
        art = a.produce("n", "content")
        a.send("b", "m", artifacts=[art])
        a.llm(lambda: LLMResult("o", 5, 5), model="claude-sonnet-5")
    ctx.set_output("done")
    traj = ctx.finish()
    back = Trajectory.from_dict(traj.to_dict())
    assert len(back.events) == len(traj.events)
    assert back.artifacts[art.artifact_id].content == "content"
    assert back.spans.keys() == traj.spans.keys()
    assert back.total_tokens == traj.total_tokens and back.final_output == "done"
    p = tmp_path / "t.json"
    traj.save(str(p))
    assert Trajectory.load(str(p)).run_id == traj.run_id


def test_text_capture_can_be_disabled():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), capture_text=False)
    ctx.start()
    with ctx.agent("a") as a:
        a.produce("big", "x" * 5000)
    ev = ctx.finish().events_of(EventType.ARTIFACT)[0]
    assert len(ev.content) < 200


def test_exception_tracker_follows_cause_chain():
    from swarmeval.recorder import _ExceptionTracker
    errors = _ExceptionTracker()
    e = ValueError("inner")
    errors.mark(e, "ev_x")
    try:
        raise RuntimeError("outer") from e
    except RuntimeError as x:
        wrapped = x
    assert errors.origin_of(wrapped) == (e, "ev_x")
    assert errors.propagation_meta(wrapped) == {"propagated": True, "origin_event_id": "ev_x"}
    assert errors.propagation_meta(KeyError("unrelated")) == {}


def test_mailbox_put_drain_peek():
    from swarmeval.recorder import _Mailbox
    box = _Mailbox()
    assert box.peek("b") == [] and box.drain("b") == []
    box.put("b", "m1")
    box.put("c", "other")
    box.put("b", "m2")
    peeked = box.peek("b")
    peeked.append("not stored")
    assert box.peek("b") == ["m1", "m2"]
    assert box.drain("b") == ["m1", "m2"]
    assert box.drain("b") == [] and box.peek("c") == ["other"]


def test_run_context_is_a_swarm_context():
    from swarmeval import SwarmContext
    assert isinstance(make_ctx(), SwarmContext)


def test_function_swarm_runs_with_a_minimal_context():
    import random

    from swarmeval import FunctionSwarm

    class Stub:
        clock = SimClock()
        rng = random.Random(0)
        outputs: list = []

        def agent(self, agent_id, role=None, model=None, **metadata):
            raise NotImplementedError

        def parallel(self, *fns, max_workers=None):
            return [f() for f in fns]

        async def gather(self, *aws):
            return [await a for a in aws]

        def set_output(self, output, artifacts=()):
            self.outputs.append(output)

    ctx = Stub()
    FunctionSwarm(lambda task, c: c.set_output(task.input.upper())).run(Task("t", "hi"), ctx)
    assert ctx.outputs == ["HI"]
