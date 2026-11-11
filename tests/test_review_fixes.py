"""Regression tests for the PR #2 review findings."""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from swarmeval import (
    Benchmark,
    LLMResult,
    RunContext,
    SimClock,
    Swarm,
    Task,
    analyze_runs,
    compute_efficiency,
    evaluate,
)
from swarmeval import evaluators as E
from swarmeval.cli import main
from swarmeval.integrations.anthropic import anthropic_llm, instrument_client
from swarmeval.pricing import PricingTable
from swarmeval.schema import EventType


def one_task(**kw):
    return Benchmark("b", [Task("q", "what?", **kw)])


def ctx_for(**kw):
    ctx = RunContext(Task("t", "q"), clock=SimClock(), **kw)
    ctx.start()
    return ctx


def test_unknown_cost_is_never_reported_as_zero():
    def swarm(task, ctx):
        with ctx.agent("local", model="llama-3-70b") as a:
            a.llm(lambda: LLMResult("x", 1000, 100))
        return "ok"

    res = evaluate(swarm, one_task())
    assert res.cost["total"] is None and res.cost["per_run"] is None and not res.cost["complete"]
    assert res.efficiency.per_agent["local"]["cost_usd"] is None
    assert res.performance.per_task["q"]["cost_usd"] is None
    assert "cost=unknown" in repr(res)
    text = res.report()
    assert "$0.0000" not in text and "unknown" in text


def test_partial_cost_is_marked_in_every_row():
    def swarm(task, ctx):
        with ctx.agent("local", model="llama-3-70b") as a:
            a.llm(lambda: LLMResult("x", 1000, 100))
        with ctx.agent("claude", model="claude-sonnet-5") as b:
            b.llm(lambda: LLMResult("y", 1000, 100))
        return "ok"

    res = evaluate(swarm, one_task())
    assert res.cost["total"] == pytest.approx((1000 * 2 + 100 * 10) / 1e6) and not res.cost["complete"]
    agent_rows = [ln for ln in res.report().splitlines() if ln.strip().startswith(("local", "q "))]
    assert agent_rows and all("unknown" in ln or "*" in ln for ln in agent_rows)


def test_saved_runs_can_be_repriced(tmp_path):
    def swarm(task, ctx):
        with ctx.agent("a", model="acme-llm-1") as a:
            a.llm(lambda: LLMResult("x", 200_000, 20_000))
        return "ok"

    res = evaluate(swarm, one_task())
    assert res.cost["total"] is None
    path = tmp_path / "r.json"
    res.save(str(path))
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"acme-llm-1": [3.0, 15.0]}))
    repriced = analyze_runs(res.runs, pricing=PricingTable.load(str(prices)))
    assert repriced.cost["total"] == pytest.approx(0.9) and repriced.cost["complete"]
    assert main(["report", str(path), "--pricing", str(prices)]) == 0


def test_pricing_lookup_does_not_borrow_another_models_price():
    p = PricingTable()
    assert p.lookup("claude-opus-5-5") == (4.0, 20.0, 0.2)
    assert p.lookup("claude-opus-5-9") is None
    assert p.lookup("claude-opus-5-20260101") == p.lookup("claude-opus-5")
    assert p.lookup("anthropic/claude-opus-5") == p.lookup("claude-opus-5")
    assert p.lookup("us.anthropic.claude-haiku-4-5-v1:0") == p.lookup("claude-haiku-4-5")
    assert p.lookup("my-claude-opus-5-finetune") is None


def test_prompt_cache_tokens_are_priced():
    p = PricingTable()
    assert p.cost("claude-sonnet-5", 100, 10, cache_read_tokens=1000, cache_write_tokens=1000) == pytest.approx(
        (100 * 2 + 10 * 10 + 1000 * 0.2 + 1000 * 2.5) / 1e6)
    assert p.cost("claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.20)
    assert p.cost("claude-sonnet-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.10)
    assert p.cost("claude-fable-5-1", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.25)
    assert p.lookup("claude-opus-4-1-20250805") == (15.0, 75.0)
    assert p.lookup("claude-3-5-haiku-20241022") == (0.8, 4.0)


@pytest.mark.parametrize("spec", [{"type": "contains"}, {"type": "regex"}, {"type": "contains_all", "keyword": ["a"]},
                                  {"type": "contains_all"}, {"type": "exact_match"}, {"type": "regex", "pattern": "("}])
def test_bad_evaluator_specs_fail_before_any_run(spec):
    runs = []

    def swarm(task, ctx):
        runs.append(task.task_id)
        return "x"

    bench = Benchmark("b", [Task("ok", "a", evaluator={"type": "exact_match", "expected": "x"}),
                            Task("bad", "b", evaluator=spec)])
    with pytest.raises(ValueError):
        evaluate(swarm, bench)
    assert runs == []


def test_task_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="evaluater"):
        Task.from_dict({"task_id": "t", "input": "q", "evaluater": {"type": "exact_match"}})


def test_caller_evaluator_overrides_benchmark_default():
    bench = Benchmark("b", [Task("q", "what?")], default_evaluator={"type": "exact_match", "expected": "nope"})
    res = evaluate(lambda task, ctx: "anything", bench, evaluator=lambda t, o, tr: True)
    assert res.success_rate == 1.0


def test_unserializable_evaluators_fail_clearly(tmp_path):
    bench = Benchmark("b", [Task("q", "what?", evaluator=lambda t, o, tr: True)])
    target = tmp_path / "bench.json"
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        bench.save(str(target))
    assert not target.exists()
    reloaded = Benchmark.from_dict(json.loads(json.dumps(bench.to_dict())))
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        reloaded.validate()


def test_task_to_dict_does_not_copy_the_evaluator():
    class Judge(E.Evaluator):
        name = "judge"

        def __init__(self):
            self.lock = threading.Lock()

        def evaluate(self, task, output, trajectory=None):
            return E.Evaluation(True, 1.0, self.name)

    res = evaluate(lambda task, ctx: "x", Benchmark("b", [Task("q", "?", evaluator=Judge())]))
    assert res.runs.trajectories[0].to_dict()["task"]["evaluator"] == {"type": "judge"}


def test_regex_flags_survive_a_roundtrip():
    ev = E.Regex("Foo", flags=0)
    back = E.from_spec(json.loads(json.dumps(ev.to_spec())))
    assert not ev.evaluate(Task("t", "q"), "foo").success
    assert not back.evaluate(Task("t", "q"), "foo").success


def test_numeric_reads_thousands_separators():
    ev = E.NumericTolerance(expected=1000)
    assert ev.evaluate(Task("t", "q"), "The answer is 1,000").success
    assert E.NumericTolerance(expected=12000).evaluate(Task("t", "q"), "about $12,000/s").success


def test_tool_use_flags_forbidden_tool_even_when_it_failed():
    ctx = ctx_for()
    with ctx.agent("a") as a:
        with pytest.raises(RuntimeError):
            a.call_tool("rm_rf", lambda: (_ for _ in ()).throw(RuntimeError("denied")))
    traj = ctx.finish()
    r = E.ToolUseEvaluator(forbidden_tools=["rm_rf"], allow_errors=True).evaluate(Task("t", "q"), "out", traj)
    assert not r.success and r.details["forbidden"] == ["rm_rf"]


def test_tool_use_counts_an_exception_through_nested_calls_once():
    ctx = ctx_for()
    with ctx.agent("a") as a:
        with pytest.raises(ZeroDivisionError):
            a.call_tool("outer", lambda: a.call_tool("inner", lambda: 1 / 0))
    traj = ctx.finish()
    assert compute_efficiency(traj).errors == 1
    r = E.ToolUseEvaluator().evaluate(Task("t", "q"), "out", traj)
    assert not r.success and r.details["errors"] == 1

def test_swarm_class_is_accepted_as_factory():
    class S(Swarm):
        def run(self, task, ctx):
            return "42"

    res = evaluate(S, one_task(evaluator={"type": "exact_match", "expected": "42"}))
    assert res.success_rate == 1.0 and res.performance.run_errors == 0


def test_async_swarm_is_awaited_and_recorded():
    async def model(prompt):
        await asyncio.sleep(0)
        return LLMResult("42", 1200, 300)

    async def search(q):
        await asyncio.sleep(0)
        return ["hit"]

    class AsyncSwarm(Swarm):
        async def run(self, task, ctx):
            async with ctx.agent("a", model="claude-sonnet-5") as a:
                await a.acall_tool("search", search, "q")
                return await a.allm(model, task.input)

    res = evaluate(AsyncSwarm(), one_task(evaluator={"type": "exact_match", "expected": "42"}))
    traj = res.runs.trajectories[0]
    assert res.success_rate == 1.0 and traj.final_output == "42"
    llm = traj.llm_calls()[0]
    assert (llm.input_tokens, llm.output_tokens, llm.agent_id) == (1200, 300, "a")
    assert traj.tool_calls()[0].content == ["hit"]


def test_async_swarm_inside_running_loop_raises_instead_of_recording_a_crash():
    from swarmeval.runner import UsageError

    class AsyncSwarm(Swarm):
        async def run(self, task, ctx):
            return "42"

    async def outer():
        return evaluate(AsyncSwarm(), one_task())

    with pytest.raises(UsageError, match="running event loop"):
        asyncio.run(outer())

    async def body():
        return "42"

    async def outer2():
        return evaluate(lambda task, ctx: body(), one_task())

    with pytest.raises(UsageError, match="running event loop"):
        asyncio.run(outer2())


def test_sync_wrappers_reject_async_functions():
    async def model(prompt):
        return "x"

    ctx = ctx_for()
    with ctx.agent("a") as a:
        with pytest.raises(TypeError, match="allm"):
            a.llm(model, "p")
        with pytest.raises(TypeError, match="acall_tool"):
            a.call_tool("t", model, "p")


def test_events_from_plain_threads_are_attributed_to_their_span():
    ctx = ctx_for()
    with ctx.agent("coord") as coord:
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda q: coord.call_tool("search", lambda: q), ["a", "b"]))
        with ThreadPoolExecutor(1) as ex:
            ex.submit(coord.llm, lambda: LLMResult("x", 10, 1)).result()
    traj = ctx.finish()
    calls = traj.tool_calls() + traj.llm_calls()
    assert all(e.agent_id == "coord" and e.span_id for e in calls)
    assert compute_efficiency(traj).per_agent["coord"].tool_calls == 2


def test_simclock_parallel_branches_overlap():
    ctx = ctx_for()

    def worker(name):
        with ctx.agent(name) as w:
            w.call_tool("search", lambda: ctx.clock.advance(0.5))
            with w.llm_call(model="claude-sonnet-5") as call:
                ctx.clock.advance(2.0)
                call.input_tokens = 10

    with ctx.agent("coord"):
        ctx.parallel(*(lambda n=n: worker(n) for n in ("w1", "w2", "w3")))
    traj = ctx.finish()
    assert traj.duration == pytest.approx(2.5)
    assert all(e.latency_ms == pytest.approx(2000.0) for e in traj.llm_calls())
    assert compute_efficiency(traj).agent_time_s == pytest.approx(7.5)


def test_parallel_branch_rng_is_reproducible():
    def draws(seed):
        ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=seed)
        return ctx.parallel(*(lambda: [ctx.rng.random() for _ in range(3)] for _ in range(4)))

    assert draws(5) == draws(5) and draws(5) != draws(6)


def test_asyncio_gather_branches_overlap():
    ctx = ctx_for()

    async def worker(name):
        async with ctx.agent(name):
            await asyncio.sleep(0)
            ctx.clock.advance(2.0)

    async def main_():
        async with ctx.agent("coord"):
            await ctx.gather(worker("w1"), worker("w2"))

    asyncio.run(main_())
    assert ctx.finish().duration == pytest.approx(2.0)


def test_one_exception_counts_as_one_error():
    def swarm(task, ctx):
        with ctx.agent("coordinator") as c:
            with c.sub_agent("worker") as w:
                with w.sub_agent("leaf") as leaf:
                    leaf.call_tool("flaky", lambda: 1 / 0)

    res = evaluate(swarm, one_task())
    eff = res.per_run[0]
    assert eff.errors == 1 and sum(a.errors for a in eff.per_agent.values()) == 1
    errors = res.runs.trajectories[0].events_of(EventType.ERROR)
    assert len(errors) == 4 and sum(1 for e in errors if not e.metadata.get("propagated")) == 0


def test_wrapped_exception_still_counts_as_one_error():
    def swarm(task, ctx):
        with ctx.agent("outer") as o:
            try:
                with o.sub_agent("inner") as i:
                    i.call_tool("flaky", lambda: 1 / 0)
            except ZeroDivisionError as e:
                raise RuntimeError("wrapped") from e

    res = evaluate(swarm, one_task())
    eff = res.per_run[0]
    assert eff.errors == 1 and {a: x.errors for a, x in eff.per_agent.items()} == {"outer": 0, "inner": 1}
    errors = res.runs.trajectories[0].events_of(EventType.ERROR)
    assert errors and all(e.metadata.get("propagated") for e in errors)
    origin = {e.metadata["origin_event_id"] for e in errors}
    assert origin == {res.runs.trajectories[0].tool_calls()[0].event_id}


def test_exception_through_nested_tool_calls_counts_once():
    """A tool that calls a tool: one ZeroDivisionError passes through both
    wrappers.  Both calls stay visibly failed; the error is counted once."""
    def swarm(task, ctx):
        with ctx.agent("a") as a:
            def inner():
                return a.call_tool("inner", lambda: 1 / 0)
            a.call_tool("outer", inner)

    res = evaluate(swarm, one_task())
    eff = res.per_run[0]
    assert eff.errors == 1 and {k: v.errors for k, v in eff.per_agent.items()} == {"a": 1}
    inner_ev, outer_ev = res.runs.trajectories[0].tool_calls()
    assert (inner_ev.tool_name, outer_ev.tool_name) == ("inner", "outer")
    assert inner_ev.status == outer_ev.status == "error"
    assert "propagated" not in inner_ev.metadata
    assert outer_ev.metadata.items() >= {"propagated": True, "origin_event_id": inner_ev.event_id}.items()


def test_exception_through_llm_call_wrapping_a_tool_call_counts_once():
    def swarm(task, ctx):
        with ctx.agent("a") as a:
            with a.llm_call(model="claude-sonnet-5"):
                a.call_tool("t", lambda: 1 / 0)

    res = evaluate(swarm, one_task())
    eff = res.per_run[0]
    assert eff.errors == 1 and eff.per_agent["a"].errors == 1
    traj = res.runs.trajectories[0]
    assert traj.llm_calls()[0].status == "error" and traj.llm_calls()[0].metadata["propagated"]


def test_llm_records_prompt_without_prompt_kwarg():
    long = "word " * 9000
    ctx = ctx_for()
    with ctx.agent("a", model="claude-opus-5") as a:
        a.llm(lambda p: "short answer", long)
        seen = []
        a.llm(lambda p: seen.append(p) or "ok", prompt="hello")
    first, second = ctx.finish().llm_calls()
    assert first.input_tokens > 10_000 and first.metadata["input_tokens_estimated"]
    assert "prompt" in first.metadata
    assert seen == ["hello"] and second.status == "success"


def test_capture_text_off_applies_to_everything_saved():
    secret = "SECRET-" + "x" * 6000

    def swarm(task, ctx):
        with ctx.agent("a") as a:
            a.call_tool("t", lambda s: s, secret)
            a.produce("art", secret)
            a.note(secret)
        return secret

    res = evaluate(swarm, one_task(), capture_text=False)
    saved = json.dumps(res.runs.to_dict())
    assert secret not in saved
    assert res.runs.trajectories[0].final_output == secret


def test_max_text_chars_applies_to_non_strings():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), max_text_chars=100)
    ctx.start()
    with ctx.agent("a") as a:
        a.call_tool("t", lambda: ["item"] * 1000)
    assert len(ctx.finish().tool_calls()[0].content) < 200


def fake_response(tin=1000, tout=200, cr=0, cw=0):
    usage = SimpleNamespace(input_tokens=tin, output_tokens=tout, cache_read_input_tokens=cr,
                            cache_creation_input_tokens=cw)
    return SimpleNamespace(usage=usage, model="claude-sonnet-5", stop_reason="end_turn",
                           content=[SimpleNamespace(type="text", text="hi")])


def fake_client(n_calls):
    def create(**kw):
        n_calls.append(kw.get("model"))
        return fake_response(cr=500, cw=100)

    return SimpleNamespace(messages=SimpleNamespace(create=create),
                           beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


@pytest.mark.parametrize("fallbacks", [False, True])
def test_instrumented_client_inside_span_llm_records_once(fallbacks):
    calls = []
    client = instrument_client(fake_client(calls))
    ctx = ctx_for()
    with ctx.agent("a") as a:
        a.llm(anthropic_llm(client, model="claude-sonnet-5", fallbacks=fallbacks), "hi")
        with a.llm_call(model="claude-sonnet-5"):
            client.messages.create(model="claude-sonnet-5", messages=[])
        client.beta.messages.create(model="claude-sonnet-5", messages=[])
    llm = ctx.finish().llm_calls()
    assert len(calls) == 3 and len(llm) == 3
    assert all((e.input_tokens, e.cache_read_tokens, e.cache_write_tokens) == (1000, 500, 100) for e in llm)


def test_instrumented_async_client_records_real_usage():
    async def create(**kw):
        return fake_response()

    client = instrument_client(SimpleNamespace(messages=SimpleNamespace(create=create)))
    ctx = ctx_for()

    async def go():
        async with ctx.agent("a"):
            await client.messages.create(model="claude-sonnet-5", messages=[])

    asyncio.run(go())
    ev = ctx.finish().llm_calls()[0]
    assert (ev.input_tokens, ev.output_tokens) == (1000, 200) and ev.cost_usd > 0


def test_report_on_summary_only_result_is_an_error(tmp_path):
    res = evaluate(lambda task, ctx: "x", one_task())
    path = tmp_path / "summary.json"
    res.save(str(path), include_runs=False)
    with pytest.raises(SystemExit, match="no trajectories"):
        main(["report", str(path)])


def test_cli_export_jsonl_and_parquet(tmp_path):
    res = evaluate(lambda task, ctx: "x", one_task())
    path = tmp_path / "r.json"
    res.save(str(path))
    assert main(["export", str(path), str(tmp_path / "e.jsonl")]) == 0
    assert len((tmp_path / "e.jsonl").read_text().splitlines()) == len(res.runs.trajectories[0].events)
    pytest.importorskip("pyarrow")
    from swarmeval.export import read_events_parquet
    assert main(["export", str(path), str(tmp_path / "e.parquet")]) == 0
    events = read_events_parquet(str(tmp_path / "e.parquet"))
    assert [e.event_type for e in events] == [e.event_type for e in res.runs.trajectories[0].events]
