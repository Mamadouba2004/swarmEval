"""Context bloat: received vs traceable prompt tokens per LLM call (metrics/context.py)."""
import pytest

from swarmeval import RunContext, SimClock, Task, compute_coordination, evaluate
from swarmeval.evaluate import analyze_runs, load_runset
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics.context import compute_context
from swarmeval.schema import estimate_tokens

TASK = "q" * 40          # 10 tokens


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
        # task + notes_a + notes_b + critique + the message body (artifacts counted once, as artifacts)
        assert set(worst.parts) == {"task", "artifacts", "messages", "prior_outputs", "system"}
        assert worst.parts["messages"] < 50 and worst.parts["artifacts"] > 0
        d = next(d for d in bloat(cm) if d.agents == ["synthesizer"])
        assert d.severity == "high" and not d.inferred
    s = res.coordination.context
    assert s.worst_agents == {"synthesizer": 2} and s.ratio.n == 2


def test_call_receiving_exactly_what_it_references_is_not_flagged():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("doc", "x" * 40_000)           # 10,000 tokens
        call(a, 10 + 10_000, input_artifacts=[art])
    cm = compute_coordination(ctx.finish())
    c = cm.context.calls[0]
    assert c.received == c.relevant == 10_010 and c.excess == 0 and c.ratio == pytest.approx(1.0)
    assert not bloat(cm)


def test_received_messages_and_earlier_outputs_count_as_relevant():
    ctx = make_ctx()
    with ctx.agent("boss") as b:
        art = b.produce("brief", "y" * 400)           # 100 tokens
        b.send("w", "z" * 80, kind="request", artifacts=[art])   # body 20 tokens + artifact 100
    with ctx.agent("w") as w:
        w.receive()
        w.consume(art)
        w.call_tool("lookup", lambda: "r" * 200)      # result: 50 tokens
        call(w, 100, out=30)
        c2 = call(w, 1000)
    ctx_m = compute_context(ctx.finish())
    first, second = [c for c in ctx_m.calls]
    assert first.parts == {"task": 10, "artifacts": 100, "messages": 20, "prior_outputs": 50, "system": 0}
    assert second.parts["prior_outputs"] == 50 + 30
    assert second.relevant == 10 + 100 + 20 + 80 and second.excess == 1000 - 210
    assert ctx_m.worst.event_id == c2.event_id and ctx_m.basis == "referenced"
    assert ctx_m.ratio == pytest.approx((100 + 1000) / (first.relevant + second.relevant))


def test_system_tokens_are_counted_relevant():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        call(a, 5010, system_tokens=5000)
        call(a, 5010)
    cm = compute_coordination(ctx.finish())
    with_sys, without = cm.context.calls
    assert with_sys.relevant == 5010 and with_sys.excess == 0 and with_sys.basis == "referenced"
    assert without.parts["prior_outputs"] == 10 and without.excess == 5010 - 20
    # only the call without a recorded system prompt is flagged
    d = bloat(cm)
    assert len(d) == 1 and d[0].evidence["worst_event_id"] == without.event_id


def test_task_only_basis_caps_severity():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        call(a, 50_000)                              # nothing recorded beyond the task
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
