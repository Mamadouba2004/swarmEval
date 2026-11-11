"""Context bloat: received vs traceable prompt tokens per LLM call (metrics/context.py)."""
import pytest

from swarmeval import RunContext, SimClock, Task, compute_coordination, evaluate
from swarmeval.evaluate import analyze_runs, load_runset
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics.context import compute_context
from swarmeval.schema import estimate_tokens

TASK = "q" * 40


def make_ctx():
    ctx = RunContext(Task("t", TASK), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def call(span, prompt_tokens, out=10, **kw):
    with span.llm_call(**kw) as c:
        span.ctx.clock.advance(1.0)
        c.input_tokens, c.output_tokens = prompt_tokens, out
    return c


def bloat(cm):
    return [d for d in cm.diagnostics if d.smell == "context_bloat"]


def test_demo_synthesizer_is_flagged_high():
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    for cm in res.per_run_coordination:
        worst = cm.context.worst
        assert worst.agent_id == "synthesizer" and worst.basis == "referenced"
        assert worst.received > 9000 and worst.ratio > 10
        assert set(worst.parts) == {"task", "artifacts", "messages", "prior_outputs", "system"}
        assert worst.parts["messages"] < 50 and worst.parts["artifacts"] > 0
        d = next(d for d in bloat(cm) if d.agents == ["synthesizer"])
        assert d.severity == "high" and not d.inferred
    s = res.coordination.context
    assert s.worst_agents == {"synthesizer": 2} and s.ratio.n == 2


def test_call_receiving_exactly_what_it_references_is_not_flagged():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("doc", "x" * 40_000)
        call(a, 10 + 10_000, input_artifacts=[art])
    cm = compute_coordination(ctx.finish())
    c = cm.context.calls[0]
    assert c.received == c.relevant == 10_010 and c.excess == 0 and c.ratio == pytest.approx(1.0)
    assert not bloat(cm)


def test_received_messages_and_earlier_outputs_count_as_relevant():
    ctx = make_ctx()
    with ctx.agent("boss") as b:
        art = b.produce("brief", "y" * 400)
        b.send("w", "z" * 80, kind="request", artifacts=[art])
    with ctx.agent("w") as w:
        w.receive()
        w.consume(art)
        w.call_tool("lookup", lambda: "r" * 200)
        call(w, 100, out=30)
        c2 = call(w, 1000)
    ctx_m = compute_context(ctx.finish())
    first, second = [c for c in ctx_m.calls]
    assert first.parts == {"task": 10, "artifacts": 100, "messages": 20, "prior_outputs": 50, "system": 0}
    assert second.parts["prior_outputs"] == 50 + 30
    assert second.relevant == 10 + 100 + 20 + 80 and second.excess == 1000 - 210
    assert ctx_m.worst.event_id == c2.event_id and ctx_m.basis == "referenced"
    assert ctx_m.ratio == pytest.approx((100 + 1000) / (first.relevant + second.relevant))


def test_messages_read_without_a_receive_are_attributed_by_timing():
    ctx = make_ctx()
    with ctx.agent("boss") as b:
        b.send("w", "z" * 80, kind="request")
        b.send("v", "z" * 40, kind="request")
    with ctx.agent("w") as w:
        w.receive()
        call(w, 100)
    with ctx.agent("v") as v:
        v.inbox()
        call(v, 100)
    by_agent = {c.agent_id: c for c in compute_context(ctx.finish()).calls}
    assert by_agent["w"].parts["messages"] == 20 and by_agent["w"].basis == "referenced"
    assert by_agent["v"].parts["messages"] == 10 and by_agent["v"].basis == "inferred"


@pytest.mark.parametrize("in_other_span", [False, True])
def test_messages_received_outside_the_receivers_span_count_as_relevant(in_other_span):
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "m" * 40_000, kind="result")
        if in_other_span:
            ctx.receive("b")
    if not in_other_span:
        ctx.receive("b")
    with ctx.agent("b") as b:
        call(b, 10 + 10_000)
    cm = compute_coordination(ctx.finish())
    c = next(c for c in cm.context.calls if c.agent_id == "b")
    assert c.parts["messages"] == 10_000 and c.basis == "referenced"
    assert not bloat(cm)


def test_message_read_in_an_earlier_span_counts_for_a_later_call():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "m" * 40_000, kind="result")
    with ctx.agent("b") as b:
        b.receive()
    with ctx.agent("b") as b:
        call(b, 10 + 10_000)
    cm = compute_coordination(ctx.finish())
    c = next(c for c in cm.context.calls if c.agent_id == "b")
    assert c.parts["messages"] == 10_000 and c.basis == "referenced"
    assert not bloat(cm)


def test_system_tokens_are_counted_relevant():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        call(a, 5010, system_tokens=5000)
        call(a, 5010)
    cm = compute_coordination(ctx.finish())
    with_sys, without = cm.context.calls
    assert with_sys.relevant == 5010 and with_sys.excess == 0 and with_sys.basis == "referenced"
    assert without.parts["prior_outputs"] == 10 and without.excess == 5010 - 20
    d = bloat(cm)
    assert len(d) == 1 and d[0].evidence["worst_event_id"] == without.event_id


def test_task_only_basis_caps_severity():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        call(a, 50_000)
    cm = compute_coordination(ctx.finish())
    assert cm.context.basis == "task_only" and cm.context.calls[0].relevant == estimate_tokens(TASK)
    d = bloat(cm)
    assert len(d) == 1 and d[0].severity == "low" and d[0].inferred


def test_no_llm_calls_is_unknown_not_zero():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.call_tool("t", lambda: "x")
    c = compute_context(ctx.finish())
    assert c.ratio is None and c.worst is None and c.basis is None and c.llm_calls == 0


def test_phase1_messages_are_attributed_by_timing_and_marked_inferred():
    res = analyze_runs(load_runset("tests/fixtures/phase1_runs.json"))
    for cm in res.per_run_coordination:
        assert cm.context.basis == "inferred"
        for d in bloat(cm):
            assert d.inferred and d.severity in ("low", "medium")
