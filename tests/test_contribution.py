"""Utilization, final-output lineage and observed contribution."""
import json

import pytest

from swarmeval import RunContext, SimClock, Task, compute_coordination, evaluate
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics import compute_contribution, final_lineage, summarize_contribution


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def llm(span, clock, seconds, tokens=(10, 5), **kw):
    with span.llm_call(**kw) as call:
        clock.advance(seconds)
        call.input_tokens, call.output_tokens = tokens


@pytest.fixture(scope="module")
def demo():
    return evaluate(make_swarm, build_benchmark(2), trials=3, clock_factory=SimClock)


def test_golden_demo_contribution(demo):
    """researcher_c's notes reach the coordinator, which drops them; everything
    else is consumed, and the final brief derives from a, b and the critique."""
    assert len(demo.per_run_coordination) == 6
    for cm in demo.per_run_coordination:
        c = cm.contribution
        assert c["researcher_c"].dead is True
        assert c["researcher_c"].status == "delivered_only" and c["researcher_c"].provenance_coverage == 1.0
        assert set(cm.lineage_agents) == {"researcher_a", "researcher_b", "critic", "synthesizer"}
        assert cm.lineage.inferred == []
        assert c["coordinator"].status == "delivered_only" and c["coordinator"].dead is False
        assert c["coordinator"].messages_used > 0
        assert c["synthesizer"].status == "in_final_lineage" and c["synthesizer"].dead is False
        assert c["researcher_a"].consumers == ["coordinator", "critic", "synthesizer"]
        assert cm.dead_agents == ["researcher_c"]
    summary = demo.coordination.contribution
    assert summary["researcher_c"]["dead_rate"] == 1.0 and summary["researcher_c"]["dead_known_runs"] == 6
    assert summary["researcher_a"]["lineage_rate"] == 1.0
    assert summary["coordinator"]["dead_rate"] == 0.0
    assert demo.coordination.dead_agents_per_run.mean == 1.0
    d = demo.to_dict()
    json.dumps(d)
    assert all(r["dead_agents"] == ["researcher_c"] for r in d["per_run"])


def test_utilization_shares(demo):
    cm = demo.per_run_coordination[0]
    util = cm.utilization
    assert sum(u.token_share for u in util.values()) == pytest.approx(1.0)
    assert sum(u.busy_share for u in util.values()) == pytest.approx(1.0)
    assert util["synthesizer"].tool_call_share == 0.0
    assert sum(u.tool_call_share for u in util.values()) == pytest.approx(1.0)
    assert util["researcher_a"].downstream_refs == 3 and util["researcher_c"].downstream_refs == 0
    assert util["coordinator"].messages_sent == 5 and util["coordinator"].artifacts_produced == 0


def _handoff(receiver_records_use):
    """a sends b notes that b never uses: when instrumented, b reads them and then records a call on other
    inputs; otherwise all of b's unrecorded work happens before it reads them."""
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        llm(a, clock, 1.0)
        art = a.produce("notes", "facts")
        a.send("b", "notes", artifacts=[art])
    with ctx.agent("b") as b:
        if receiver_records_use:
            b.receive()
            other = b.produce("own", "b's own work", derived_from=[])
            llm(b, clock, 1.0, input_artifacts=[other])
        else:
            llm(b, clock, 1.0)
            b.receive()
        final = b.produce("final", "answer")
    ctx.set_output("answer", artifacts=[final])
    return ctx.finish()


def test_partial_instrumentation_never_says_dead():
    traj = _handoff(receiver_records_use=False)
    cm = compute_coordination(traj)
    a = cm.contribution["a"]
    assert cm.coverage.consumers_instrumented == {"a": False, "b": False}
    assert a.status == "delivered_only" and a.provenance_coverage == 0.0
    assert a.dead is None
    assert cm.contribution["b"].dead is False


def test_complete_instrumentation_and_no_use_is_dead():
    cm = compute_coordination(_handoff(receiver_records_use=True))
    a = cm.contribution["a"]
    assert a.provenance_coverage == 1.0 and a.dead is True and cm.dead_agents == ["a"]


def _self_send(self_send):
    """c produces notes (optionally mailing them to itself); w, active
    afterwards, records no inputs, so it could have used them unseen."""
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("c") as c:
        art = c.produce("notes", "facts", derived_from=[])
        if self_send:
            c.send("c", "reminder", artifacts=[art])
    with ctx.agent("w") as w:
        llm(w, clock, 1.0)
    return compute_coordination(ctx.finish()).contribution["c"]


def test_message_to_self_does_not_make_provenance_complete():
    plain, mailed = _self_send(False), _self_send(True)
    assert (plain.provenance_coverage, plain.dead) == (0.0, None)
    assert (mailed.provenance_coverage, mailed.dead) == (plain.provenance_coverage, plain.dead)


def test_empty_derived_from_does_not_make_an_agent_instrumented():
    """An artifact made "from nothing" says nothing about what b's calls used, so a (whose message b read
    in a call listing no inputs) cannot be judged dead."""
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        art = a.produce("notes", "facts")
        a.send("b", "notes", artifacts=[art])
    with ctx.agent("b") as b:
        b.receive()
        llm(b, clock, 1.0)
        final = b.produce("final", "answer", derived_from=[])
    ctx.set_output("answer", artifacts=[final])
    cm = compute_coordination(ctx.finish())
    assert cm.coverage.consumers_instrumented["b"] is False
    assert cm.contribution["a"].provenance_coverage == 0.0 and cm.contribution["a"].dead is not True


def test_reading_an_artifact_message_in_a_call_without_recorded_inputs_counts_as_use():
    """Reading a's message in a call that lists no inputs may be use, so a is not dead; a call listing
    *other* inputs is recorded non-use (see _handoff)."""
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        art = a.produce("notes", "facts")
        a.send("b", "notes", artifacts=[art])
    with ctx.agent("b") as b:
        own = b.produce("own", "b's own work", derived_from=[])
        llm(b, clock, 1.0, input_artifacts=[own])
        b.receive()
        llm(b, clock, 1.0)
        final = b.produce("final", "answer")
    ctx.set_output("answer", artifacts=[final])
    cm = compute_coordination(ctx.finish())
    (link,) = cm.graph.messages
    assert link.received and link.used and link.used_inferred
    a_c = cm.contribution["a"]
    assert a_c.provenance_coverage == 1.0 and a_c.messages_used == 1 and a_c.dead is False


def test_consumption_and_lineage_fallback_without_derived_from():
    ctx = make_ctx()
    clock = ctx.clock
    with ctx.agent("a") as a:
        notes = a.produce("notes", "facts")
    with ctx.agent("b") as b:
        b.consume(notes)
        draft = b.produce("draft", "d")
    with ctx.agent("c") as c:
        llm(c, clock, 1.0, input_artifacts=[draft])
        final = c.produce("final", "f", derived_from=[draft])
    ctx.set_output("f", artifacts=[final])
    traj = ctx.finish()
    lin = final_lineage(traj)
    assert set(lin.artifacts) == {final.artifact_id, draft.artifact_id, notes.artifact_id}
    assert lin.inferred == [notes.artifact_id]
    contrib = compute_contribution(traj)
    assert contrib["a"].in_final_lineage and contrib["a"].lineage_inferred
    assert contrib["b"].in_final_lineage and not contrib["b"].lineage_inferred
    assert contrib["a"].consumers == ["b"] and contrib["a"].dead is False


def test_agent_without_outputs_is_unknown_not_dead():
    ctx = make_ctx()
    with ctx.agent("thinker") as t:
        llm(t, ctx.clock, 1.0)
    cm = compute_coordination(ctx.finish())
    c = cm.contribution["thinker"]
    assert c.status == "unknown" and c.provenance_coverage is None and c.dead is None
    assert cm.coverage.message_receive_coverage is None and cm.coverage.derivation_coverage is None


def test_summary_dead_rate_ignores_unknown_runs():
    known = compute_coordination(_handoff(receiver_records_use=True)).contribution
    unknown = compute_coordination(_handoff(receiver_records_use=False)).contribution
    s = summarize_contribution([known, unknown])
    assert s["a"]["dead_rate"] == 1.0 and s["a"]["dead_known_runs"] == 1
    assert s["a"]["provenance_coverage"]["mean"] == pytest.approx(0.5)
    assert s["a"]["status_counts"] == {"delivered_only": 2}


def test_contribution_is_never_reported_as_a_percentage(demo):
    text = json.dumps(demo.coordination.to_dict()) + json.dumps(demo.per_run_coordination[0].to_dict())
    assert "contributed" not in text
    cm, cs = demo.per_run_coordination[0], demo.coordination
    contrib = json.dumps([{k: v.to_dict() for k, v in cm.contribution.items()}, cm.lineage.to_dict(),
                          cs.contribution])
    assert "%" not in contrib
