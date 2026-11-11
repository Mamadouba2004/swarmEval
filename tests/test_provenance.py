"""Schema v2 provenance recording: message ids, receive events, reply_to,
derived_from, wait targets -- and that Phase 1 data still loads."""
import json
import os
import sys
import threading

import pytest

from swarmeval import RunContext, SimClock, Task, Trajectory, build_graph
from swarmeval.evaluate import analyze_runs
from swarmeval.runner import RunSet
from swarmeval.schema import SCHEMA_VERSION, Artifact, EventType

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "phase1_runs.json")


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def test_send_and_receive_share_message_id_and_receive_is_attributed_to_receiver():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("notes", "some notes", derived_from=[])
        sent = a.send("b", "take this", kind="result", artifacts=[art])
    ctx.clock.advance(2.0)
    with ctx.agent("b") as b:
        (got,) = b.receive()
    traj = ctx.finish()
    (msg_ev,) = traj.events_of(EventType.MESSAGE)
    (recv_ev,) = traj.events_of(EventType.MESSAGE_RECEIVE)
    assert sent.message_id == got.message_id == msg_ev.message_id == recv_ev.message_id
    assert recv_ev.agent_id == "b" and recv_ev.span_id == b.span_id and recv_ev.sender == "a"
    assert recv_ev.input_artifacts == [art.artifact_id]       # transported, not consumed
    assert recv_ev.content_tokens == 0 and recv_ev.kind == "result"
    assert recv_ev.metadata["delay_s"] == pytest.approx(2.0)
    assert traj.metadata["schema_version"] == SCHEMA_VERSION == 2


def test_receive_in_a_parallel_branch_waits_for_the_send_under_simclock():
    """Each ctx.parallel branch has its own SimClock time: a receive in a
    branch still at t=0 of a message sent at t=5 is stamped at t=5, not before
    the send (which would drop the message edge as pointing backwards)."""
    ctx = make_ctx()
    sent = threading.Event()

    def sender():
        with ctx.agent("a") as a:
            ctx.clock.advance(5.0)
            a.send("b", "done")
        sent.set()

    def receiver():
        sent.wait(5)
        with ctx.agent("b") as b:
            (got,) = b.receive()
            assert ctx.now() == pytest.approx(5.0)

    ctx.parallel(sender, receiver)
    traj = ctx.finish()
    (recv_ev,) = traj.events_of(EventType.MESSAGE_RECEIVE)
    assert recv_ev.timestamp == pytest.approx(5.0) and recv_ev.metadata["delay_s"] == 0.0
    g = build_graph(traj)
    assert g.dropped_edges == 0 and g.messages[0].received
    assert g.coverage.message_receive_coverage == 1.0


def test_reply_to_accepts_message_or_id_and_round_trips(tmp_path):
    ctx = make_ctx()
    with ctx.agent("a") as a:
        q = a.send("b", "question?", kind="request")
    with ctx.agent("b") as b:
        (m,) = b.receive()
        r1 = b.send("a", "answer", kind="result", reply_to=m)
        r2 = ctx.send("b", "a", "more", reply_to=m.message_id)
    with ctx.agent("a") as a:
        replies = a.receive()
    traj = ctx.finish()
    assert r1.reply_to == r2.reply_to == q.message_id
    assert [x.reply_to for x in replies] == [q.message_id, q.message_id]
    path = str(tmp_path / "t.json")
    traj.save(path)
    back = Trajectory.load(path)
    msgs = back.events_of(EventType.MESSAGE)
    assert msgs[0].reply_to is None and msgs[1].reply_to == msgs[2].reply_to == q.message_id
    recvs = back.events_of(EventType.MESSAGE_RECEIVE)
    assert [e.reply_to for e in recvs] == [None, q.message_id, q.message_id]


def test_inbox_records_nothing_and_receive_outside_own_span_has_no_span():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "hi")
        n = len(ctx.trajectory.events)
        assert len(ctx.inbox("b")) == 1 and len(ctx.trajectory.events) == n
        ctx.receive("b")   # read on b's behalf while a's span is active
    (ev,) = ctx.finish().events_of(EventType.MESSAGE_RECEIVE)
    assert ev.agent_id == "b" and ev.span_id is None


def test_derived_from_none_vs_empty_survives_save_and_load(tmp_path):
    ctx = make_ctx()
    with ctx.agent("a") as a:
        raw = a.produce("raw", "x")
        base = a.produce("base", "y", derived_from=[])
        derived = a.produce("derived", "z", derived_from=[raw, base.artifact_id])
    traj = ctx.finish()
    assert raw.derived_from is None and base.derived_from == []
    assert derived.derived_from == [raw.artifact_id, base.artifact_id]
    produce_events = {e.output_artifacts[0]: e for e in traj.events_of(EventType.ARTIFACT)}
    assert produce_events[derived.artifact_id].input_artifacts == [raw.artifact_id, base.artifact_id]
    path = str(tmp_path / "t.json")
    traj.save(path)
    back = Trajectory.load(path)
    assert back.artifacts[raw.artifact_id].derived_from is None
    assert back.artifacts[base.artifact_id].derived_from == []
    assert back.artifacts[derived.artifact_id].derived_from == [raw.artifact_id, base.artifact_id]


def test_wait_on_messages_artifacts_and_agents():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("notes", "n")
        msg = a.send("b", "ping")
    with ctx.agent("b") as b:
        b.wait("reply", 1.0, on=msg)
        with b.waiting("both", on=[art, "a"]):
            ctx.clock.advance(0.5)
        b.wait("plain", 0.1)
    w1, w2, w3 = ctx.finish().events_of(EventType.WAIT)
    assert w1.metadata == {"on_messages": [msg.message_id]} and w1.input_artifacts == []
    assert w2.metadata == {"on_agents": ["a"]} and w2.input_artifacts == [art.artifact_id]
    assert w2.latency_ms == pytest.approx(500.0)
    assert w3.metadata == {} and w3.input_artifacts == []   # unchanged Phase 1 shape


def test_nested_call_records_its_parent_but_gathered_calls_do_not():
    import asyncio

    ctx = make_ctx()

    async def fn(seconds):
        ctx.clock.advance(seconds)
        return "x"

    async def main():
        async with ctx.agent("a") as a:
            await ctx.gather(a.allm(fn, 1.0), a.allm(fn, 2.0))
            with a.tool_call("outer") as outer:
                a.llm(lambda: "inner")
            return outer

    outer = asyncio.run(main())
    traj = ctx.finish()
    gathered = traj.llm_calls()[:2]
    assert all("within_call" not in e.metadata for e in gathered)
    inner = traj.llm_calls()[2]
    assert inner.metadata["within_call"] == outer.event_id == traj.tool_calls()[0].event_id


def test_save_load_save_is_identical(tmp_path):
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("notes", "n", derived_from=[])
        m = a.send("b", "x", artifacts=[art])
    with ctx.agent("b") as b:
        b.receive()
        b.send("a", "y", reply_to=m)
        b.wait("w", 0.2, on=[m, art, "a"])
    traj = ctx.finish()
    p1, p2 = str(tmp_path / "1.json"), str(tmp_path / "2.json")
    traj.save(p1)
    Trajectory.load(p1).save(p2)
    assert open(p1).read() == open(p2).read()


def test_unknown_and_missing_fields_are_tolerated():
    ev = {"event_id": "e", "run_id": "r", "task_id": "t", "event_type": "message", "timestamp": 0.0,
          "some_future_field": 1}
    from swarmeval.schema import Event
    e = Event.from_dict(ev)
    assert e.message_id is None and e.reply_to is None
    a = Artifact.from_dict({"artifact_id": "x", "name": "n", "producer": "p", "span_id": None, "produced_at": 0.0,
                            "content": "c", "content_hash": "h", "tokens": 1, "future": True})
    assert a.derived_from is None


# --------------------------------------------------------------------------- #
# Phase 1 data
# --------------------------------------------------------------------------- #
def test_phase1_fixture_loads_and_analyses_with_inferred_edges():
    runs = RunSet.load(FIXTURE)
    assert len(runs) == 2
    assert all("schema_version" not in t.metadata for t in runs.trajectories)
    assert not any(t.events_of(EventType.MESSAGE_RECEIVE) for t in runs.trajectories)
    res = analyze_runs(runs)
    for cm in res.per_run_coordination:
        g = cm.graph
        msg_edges = g.edges_of("message")
        assert msg_edges and all(e.inferred for e in msg_edges)
        assert all(not m.received and m.inferred for m in g.messages)
        assert cm.coverage.message_receive_coverage == 0.0
        assert cm.coverage.derivation_coverage == 0.0
        # lineage is still found, through span consumption, and flagged as such
        assert set(cm.lineage_agents) == {"researcher_a", "researcher_b", "critic", "synthesizer"}
        assert cm.contribution["researcher_a"].lineage_inferred
        assert not cm.contribution["synthesizer"].lineage_inferred
    d = res.to_dict()
    json.dumps(d)
    assert [r["message_receive_coverage"] for r in d["per_run"]] == [0.0, 0.0]
    assert d["coordination"]["message_receive_coverage"]["mean"] == 0.0


def test_analyze_runs_without_coordination():
    res = analyze_runs(RunSet.load(FIXTURE), coordination=False)
    assert res.per_run_coordination is None and res.coordination is None
    d = res.to_dict()
    assert d["coordination"] is None and d["per_run"][0]["dead_agents"] is None


def test_coordination_failure_on_one_run_keeps_the_evaluation(monkeypatch):
    import swarmeval.metrics.coordination as coordination
    runs = RunSet.load(FIXTURE)
    bad = runs.trajectories[0].run_id
    real = coordination.build_graph

    def build_graph(traj):
        if traj.run_id == bad:
            raise KeyError("odd trace")
        return real(traj)

    monkeypatch.setattr(coordination, "build_graph", build_graph)
    with pytest.warns(UserWarning, match="failed on 1/2 runs"):
        res = analyze_runs(runs)
    assert len(res.per_run) == 2 and res.performance is not None
    assert res.per_run_coordination[0] is None and res.per_run_coordination[1] is not None
    assert res.coordination_errors == {bad: "KeyError: 'odd trace'"}
    assert res.coordination.n_runs == 1
    d = res.to_dict()
    json.dumps(d)
    assert d["coordination_errors"] == {bad: "KeyError: 'odd trace'"}
    assert d["per_run"][0]["dead_agents"] is None and d["per_run"][1]["dead_agents"] is not None


def _demo_runs():
    from swarmeval import evaluate
    from swarmeval.examples.research_swarm import build_benchmark, make_swarm
    return evaluate(make_swarm, build_benchmark(8), trials=2, clock_factory=SimClock).runs


def test_coordination_failure_on_some_runs_warns(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]  # the package re-exports evaluate() under the same name
    runs = _demo_runs()
    assert len(runs) == 16
    real, calls = ev.compute_coordination, []

    def flaky(traj, **kw):
        calls.append(traj.run_id)
        if len(calls) % 2 == 0:
            raise RuntimeError("boom")
        return real(traj, **kw)

    monkeypatch.setattr(ev, "compute_coordination", flaky)
    with pytest.warns(UserWarning, match=r"coordination analysis failed on 8/16 runs: RuntimeError: boom"):
        res = analyze_runs(runs)
    assert len(res.coordination_errors) == 8
    assert res.coordination.n_runs == 8


@pytest.mark.parametrize("n_runs", [1, 2])
def test_odd_traces_failing_every_run_keep_the_evaluation(monkeypatch, n_runs):
    """One run, or runs failing for different reasons, are odd traces, not a configuration error."""
    ev = sys.modules["swarmeval.evaluate"]
    runs = _demo_runs()
    runs.trajectories = runs.trajectories[:n_runs]
    kinds = iter([KeyError("odd trace"), ValueError("another odd trace")])

    def odd(traj, **kw):
        raise next(kinds)

    monkeypatch.setattr(ev, "compute_coordination", odd)
    with pytest.warns(UserWarning, match=f"failed on {n_runs}/{n_runs} runs"):
        res = analyze_runs(runs)
    assert res.coordination is None and len(res.coordination_errors) == n_runs
    assert len(res.runs) == n_runs and res.performance.n_runs == n_runs


def test_coordination_failure_on_every_run_raises(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]  # the package re-exports evaluate() under the same name
    runs = _demo_runs()

    def missing_dep(traj, **kw):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(ev, "compute_coordination", missing_dep)
    with pytest.raises(ImportError, match="sentence-transformers"):
        analyze_runs(runs)


def test_coordination_config_error_in_evaluate_keeps_the_runs(monkeypatch):
    """A failure after every (paid) run has executed must not throw the runs away."""
    from swarmeval import AnalysisFailed, evaluate
    from swarmeval.examples.research_swarm import build_benchmark, make_swarm
    ev = sys.modules["swarmeval.evaluate"]

    def missing_dep(traj, **kw):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(ev, "compute_coordination", missing_dep)
    with pytest.raises(AnalysisFailed, match="ImportError: sentence-transformers") as info:
        evaluate(make_swarm, build_benchmark(3), trials=1, clock_factory=SimClock)
    exc = info.value
    assert len(exc.runs) == 3
    assert exc.result.coordination is None and exc.result.performance.n_runs == 3
    assert exc.result.coordination_errors == {t.run_id: "ImportError: sentence-transformers is not installed"
                                              for t in exc.runs.trajectories}
    assert isinstance(exc.__cause__, ImportError)
