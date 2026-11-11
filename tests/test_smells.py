"""Swarm smells (diagnostics/smells.py) and their prevalence over runs (diagnostics/engine.py)."""
import copy
import json
import re
import sys
from functools import partial

import pytest
from test_metrics import fan_out_trajectory

from swarmeval import (
    Diagnostic,
    RunContext,
    RunSet,
    SimClock,
    SmellConfig,
    Task,
    analyze_runs,
    compute_coordination,
    evaluate,
    load_smell_config,
    summarize_diagnostics,
)
from swarmeval.diagnostics.smells import SMELLS, detect_smells, resolve_smell_config
from swarmeval.examples.research_swarm import ResearchSwarm, build_benchmark, make_swarm
from swarmeval.report.text import render_coordination

BANNED = [re.compile(p, re.IGNORECASE)
          for p in (r"\bcaus", r"contributed \d+%", r"responsible for", r"because of")]


def make_ctx(difficulty=None):
    ctx = RunContext(Task("t", "q", difficulty=difficulty), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


def llm(span, seconds=1.0, tokens=100):
    with span.llm_call() as c:
        span.ctx.clock.advance(seconds)
        c.input_tokens, c.output_tokens = tokens, tokens


def names(cm):
    return {d.smell for d in cm.diagnostics}


def of(cm, smell):
    return [d for d in cm.diagnostics if d.smell == smell]


@pytest.fixture(scope="module")
def demo():
    return evaluate(make_swarm, build_benchmark(), trials=2, clock_factory=SimClock)


def test_golden_demo_smells(demo):
    expected = {"coordinator_bottleneck", "dead_agent", "under_specialization", "redundant_delegation",
                "context_bloat"}
    for cm in demo.per_run_coordination:
        assert names(cm) == expected
        assert [d.agents for d in of(cm, "dead_agent")] == [["researcher_c"]]
        assert of(cm, "under_specialization")[0].agents == ["researcher_a", "researcher_c"]
        assert of(cm, "redundant_delegation")[0].agents == ["coordinator", "researcher_a", "researcher_c"]
        assert of(cm, "coordinator_bottleneck")[0].agents == ["coordinator"]
        assert {tuple(d.agents) for d in of(cm, "context_bloat")} == {("coordinator",), ("synthesizer",)}
        assert not any(d.inferred for d in cm.diagnostics)
    rep = demo.coordination.diagnostics
    assert set(rep.names) == expected and all(s.prevalence == 1.0 for s in rep.smells)
    assert rep.get("dead_agent").agents == {"researcher_c": len(demo.runs)}
    assert rep.get("context_bloat").agents == {"coordinator": len(demo.runs), "synthesizer": len(demo.runs)}
    assert demo.coordination.smells_per_run.mean == 5.0


def test_bottleneck_relay_and_critical_path_evidence(demo):
    d = of(demo.per_run_coordination[0], "coordinator_bottleneck")[0]
    ev = d.evidence
    assert ev["hub_share"] == 1.0 and len(ev["peers"]) == 5
    assert ev["relay_fraction"] == pytest.approx(0.4)
    assert ev["critical_path_share"] >= 0.25 and d.severity == "medium"


def test_review_rounds_default_keeps_the_original_flow():
    a = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    b = evaluate(ResearchSwarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    assert a.efficiency.tokens.total == b.efficiency.tokens.total and a.efficiency.messages.mean == 9
    with pytest.raises(ValueError):
        ResearchSwarm(review_rounds=0)


def test_review_rounds_show_ping_pong_without_a_cycle():
    res = evaluate(partial(make_swarm, review_rounds=3), build_benchmark(2), trials=1, clock_factory=SimClock)
    for cm in res.per_run_coordination:
        pp = of(cm, "agent_ping_pong")
        assert len(pp) == 1 and pp[0].agents == ["coordinator", "critic"]
        assert pp[0].evidence["alternations"] == 5 and pp[0].severity == "medium"
        assert cm.dependency.cycles == []
        assert "premature_synthesis" not in names(cm) and "late_information" not in names(cm)


def exchange_traj(n_messages):
    """a and b take turns, each answering the last message: n messages, n - 1 changes of direction."""
    ctx = make_ctx()
    last = None
    for i in range(n_messages):
        src, dst = ("a", "b") if i % 2 == 0 else ("b", "a")
        with ctx.agent(src) as s:
            s.receive()
            llm(s)
            last = s.send(dst, f"round {i}", kind="feedback", reply_to=last)
    with ctx.agent("b" if n_messages % 2 else "a") as s:
        s.receive()
    return ctx.finish()


def exchange(n_messages, **kw):
    return compute_coordination(exchange_traj(n_messages), **kw)


def test_ping_pong_thresholds():
    assert not of(exchange(4), "agent_ping_pong")
    six = of(exchange(7), "agent_ping_pong")
    assert len(six) == 1 and six[0].severity == "medium" and six[0].evidence["alternations"] == 6
    assert of(exchange(9), "agent_ping_pong")[0].severity == "high"
    assert not of(exchange(7, smell_config={"pingpong_min_alternations": 7}), "agent_ping_pong")


def test_sequential_delegation_is_not_ping_pong():
    ctx = make_ctx()
    for i in range(6):
        with ctx.agent("boss") as b:
            b.receive()
            b.send("worker", f"task {i}", kind="task_assignment")
        with ctx.agent("worker") as w:
            w.receive()
            llm(w)
            w.send("boss", f"done {i}", kind="result")
    assert not of(compute_coordination(ctx.finish()), "agent_ping_pong")


@pytest.mark.parametrize("reply_to", [False, True])
def test_delegated_tasks_with_default_kinds_are_not_ping_pong(reply_to):
    ctx = make_ctx()
    for i in range(3):
        with ctx.agent("coordinator") as b:
            b.receive()
            task = b.send("worker", f"task {i}")
        with ctx.agent("worker") as w:
            w.receive()
            llm(w)
            w.send("coordinator", f"done {i}", reply_to=task if reply_to else None)
    assert not of(compute_coordination(ctx.finish()), "agent_ping_pong")


@pytest.mark.parametrize("reply_to", [False, True])
def test_review_loop_of_plain_text_drafts_with_a_reviewer_is_ping_pong(reply_to):
    ctx = make_ctx()
    for i in range(6):
        with ctx.agent("writer") as w:
            w.receive()
            draft = w.send("rev", f"draft v{i}")
        with ctx.agent("rev", role="reviewer") as r:
            r.receive()
            llm(r)
            r.send("writer", f"fix issue {i}", reply_to=draft if reply_to else None)
    pp = of(compute_coordination(ctx.finish()), "agent_ping_pong")
    assert [(d.severity, d.evidence["alternations"]) for d in pp] == [("high", 11)]


def test_same_artifact_attached_to_unrelated_tasks_is_not_ping_pong():
    ctx = make_ctx()
    with ctx.agent("coordinator") as b:
        guide = b.produce("style_guide", "write plainly", derived_from=[])
    for i in range(3):
        with ctx.agent("coordinator") as b:
            b.receive()
            b.send("worker", f"task {i}", artifacts=[guide])
        with ctx.agent("worker") as w:
            w.receive()
            llm(w)
            w.send("coordinator", f"done {i}")
    assert not of(compute_coordination(ctx.finish()), "agent_ping_pong")


@pytest.mark.parametrize("revision", ["same_name", "derived_from"])
def test_review_loop_of_redrafts_sent_as_requests_is_ping_pong(revision):
    ctx = make_ctx()
    draft = None
    for i in range(6):
        with ctx.agent("writer") as w:
            w.receive()
            llm(w)
            if revision == "same_name":
                draft = w.produce("draft", f"draft {i}", derived_from=[])
            else:
                draft = w.produce(f"draft_v{i}", f"draft {i}", derived_from=[draft] if draft else [])
            w.send("critic", f"please review draft {i}", kind="request", artifacts=[draft])
        with ctx.agent("critic") as c:
            c.receive()
            llm(c)
            c.send("writer", f"critique {i}", kind="feedback")
    pp = of(compute_coordination(ctx.finish()), "agent_ping_pong")
    assert len(pp) == 1 and pp[0].agents == ["writer", "critic"] and pp[0].evidence["alternations"] == 11


def test_review_loop_chained_by_reply_to_with_default_kinds_is_ping_pong():
    ctx = make_ctx()
    last = None
    for i in range(6):
        with ctx.agent("writer") as w:
            w.receive()
            llm(w)
            ask = w.send("critic", f"draft {i}", reply_to=last)
        with ctx.agent("critic") as c:
            c.receive()
            llm(c)
            last = c.send("writer", f"critique {i}", reply_to=ask)
    pp = of(compute_coordination(ctx.finish()), "agent_ping_pong")
    assert len(pp) == 1 and pp[0].evidence["alternations"] == 11


def test_redundant_delegation_from_several_senders():
    ctx = make_ctx()
    for boss in ("lead_1", "lead_2"):
        with ctx.agent(boss) as b:
            b.send("worker", "Please summarise the quarterly sales report for the board", kind="request")
    with ctx.agent("worker") as w:
        w.receive()
        llm(w)
    cm = compute_coordination(ctx.finish())
    d = of(cm, "redundant_delegation")
    assert len(d) == 1 and d[0].agents == ["lead_1", "lead_2", "worker"]
    assert d[0].evidence["similarity"] == 1.0


def test_verifier_duplicates_are_not_redundant_delegation():
    ctx = make_ctx()
    with ctx.agent("boss") as b:
        b.send("worker", "check s1", kind="task_assignment")
        b.send("checker", "check s1", kind="task_assignment")
    with ctx.agent("worker") as w:
        w.receive()
        w.call_tool("search", lambda q: q, q="s1")
    with ctx.agent("checker", role="verifier") as v:
        v.receive()
        v.call_tool("search", lambda q: q, q="s1")
    cm = compute_coordination(ctx.finish())
    assert "redundant_delegation" not in names(cm) and "under_specialization" not in names(cm)


def test_over_agentization_on_a_simple_task():
    ctx = make_ctx(difficulty="simple")
    for a in ("a", "b", "c"):
        with ctx.agent(a) as s:
            llm(s)
    cm = compute_coordination(ctx.finish())
    d = of(cm, "over_agentization")
    assert len(d) == 1 and d[0].severity == "low" and d[0].evidence["agents"] == 3
    many = compute_coordination(fan_out_trajectory())
    assert "over_agentization" not in names(many)


def test_over_agentization_keeps_agent_ids_case():
    ctx = make_ctx()
    for a in ("Lead", "Writer", "ResearcherC", "QA_Bot"):
        with ctx.agent(a) as s:
            llm(s)
            if a in ("ResearcherC", "QA_Bot"):
                s.produce("notes", "unused notes")
    d = of(compute_coordination(ctx.finish()), "over_agentization")
    assert len(d) == 1 and "(ResearcherC and QA_Bot)" in d[0].observation


def test_late_information_and_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        w.produce("draft", "the draft", derived_from=[])
    with ctx.agent("researcher") as r:
        llm(r)
        r.send("writer", "an important late finding", kind="result")
    traj = ctx.finish()
    cm = compute_coordination(traj)
    late = of(cm, "late_information")
    assert len(late) == 1 and late[0].agents == ["writer", "researcher"] and late[0].severity == "medium"

    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
    with ctx.agent("researcher") as r:
        llm(r)
        r.send("writer", "an important late finding", kind="result")
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    cm = compute_coordination(ctx.finish())
    pre = of(cm, "premature_synthesis")
    assert len(pre) == 1 and pre[0].severity == "high" and pre[0].evidence["final_producer"]
    assert not of(cm, "late_information")


@pytest.mark.parametrize("kind", ["result", "information"])
def test_peeked_message_followed_by_work_is_not_late_information(kind):
    ctx = make_ctx()
    with ctx.agent("b") as b:
        llm(b)
    with ctx.agent("a") as a:
        llm(a)
        a.send("b", "use source X, not Y", kind=kind)
    with ctx.agent("b") as b:
        assert b.inbox()
        llm(b)
    assert not of(compute_coordination(ctx.finish()), "late_information")


def test_message_arriving_during_the_receivers_last_call_is_late_information():
    ctx = make_ctx()
    with ctx.agent("b") as b:
        llm(b)
        with b.llm_call() as k:
            ctx.clock.advance(5.0)
            with ctx.agent("a") as a:
                a.send("b", "use source X, not Y", kind="result")
            ctx.clock.advance(5.0)
            k.input_tokens, k.output_tokens = 10, 5
    assert [d.severity for d in of(compute_coordination(ctx.finish()), "late_information")] == ["medium"]


def asked_then_answered(produce_first, reply_to):
    """writer asks reviewer something, before or after producing the final draft; the answer comes last."""
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        if produce_first:
            llm(w)
            final = w.produce("draft", "the draft", derived_from=[])
        req = w.send("reviewer", "please check this", kind="request")
        if not produce_first:
            llm(w)
            final = w.produce("draft", "the draft", derived_from=[])
    with ctx.agent("reviewer") as r:
        r.receive()
        llm(r)
        r.send("writer", "approved, no changes", kind="feedback", reply_to=req if reply_to else None)
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    return compute_coordination(ctx.finish())


@pytest.mark.parametrize("reply_to", [True, False])
def test_answer_to_a_request_sent_after_producing_is_not_premature_synthesis(reply_to):
    assert not of(asked_then_answered(True, reply_to), "premature_synthesis")
    pre = of(asked_then_answered(False, reply_to), "premature_synthesis")
    assert len(pre) == 1 and pre[0].severity == "high"


def test_review_approval_with_default_kinds_is_not_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
        w.send("reviewer", "final draft attached", artifacts=[final])
    with ctx.agent("reviewer") as r:
        r.receive()
        llm(r)
        r.send("writer", "approved, no changes")
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    assert not of(compute_coordination(ctx.finish()), "premature_synthesis")


def test_unrelated_note_after_producing_does_not_excuse_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
        w.send("reviewer", "fyi I'm offline after 5")
    with ctx.agent("reviewer") as r:
        llm(r)
        r.send("writer", "here are the 3 facts you were missing")
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    assert [d.severity for d in of(compute_coordination(ctx.finish()), "premature_synthesis")] == ["high"]


def test_reply_to_a_review_request_sent_to_another_agent_is_not_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
        ask = w.send("editor", "please review the draft", kind="request", artifacts=[final])
    with ctx.agent("reviewer") as r:
        llm(r)
        r.send("writer", "approved", kind="feedback", reply_to=ask)
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    assert not of(compute_coordination(ctx.finish()), "premature_synthesis")


def test_question_asked_before_producing_with_default_kinds_is_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        w.send("reviewer", "what should the draft cover?")
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
    with ctx.agent("reviewer") as r:
        r.receive()
        llm(r)
        r.send("writer", "cover the risks too")
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    pre = of(compute_coordination(ctx.finish()), "premature_synthesis")
    assert len(pre) == 1 and pre[0].severity == "high"


def test_unsolicited_late_information_is_premature_synthesis():
    ctx = make_ctx()
    with ctx.agent("writer") as w:
        llm(w)
        final = w.produce("draft", "the draft", derived_from=[])
    with ctx.agent("researcher") as r:
        llm(r)
        r.send("writer", "an important late finding")
    with ctx.agent("writer") as w:
        w.receive()
    ctx.set_output("the draft", artifacts=[final])
    pre = of(compute_coordination(ctx.finish()), "premature_synthesis")
    assert len(pre) == 1 and pre[0].agents == ["writer", "researcher"] and pre[0].severity == "high"


def test_communication_overhead():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        llm(a, tokens=50)
        for i in range(4):
            a.send("b", f"long status update number {i} " + "blah " * 60, kind="coordination")
    with ctx.agent("b") as b:
        b.receive()
        llm(b, tokens=50)
    cm = compute_coordination(ctx.finish())
    d = of(cm, "communication_overhead")
    assert cm.communication.comm_work_ratio > 0.5
    assert len(d) == 1 and d[0].severity == "high" and "not automatically waste" in d[0].hypothesis


def test_dead_agent_needs_complete_provenance():
    demo_cm = compute_coordination(evaluate(make_swarm, build_benchmark(1), trials=1,
                                            clock_factory=SimClock).runs.trajectories[0])
    assert [d.agents for d in of(demo_cm, "dead_agent")] == [["researcher_c"]]
    ctx = make_ctx()
    with ctx.agent("a") as a:
        art = a.produce("notes", "n")
        a.send("b", "notes", kind="result", artifacts=[art])
    with ctx.agent("b") as b:
        b.receive()
        llm(b)
    cm = compute_coordination(ctx.finish())
    assert cm.contribution["a"].dead is False and not of(cm, "dead_agent")


def test_smell_config_overrides_and_unknown_keys(tmp_path, demo):
    p = tmp_path / "smells.json"
    p.write_text(json.dumps({"bloat_ratio": 100.0, "hub_share": 0.99}))
    cfg = load_smell_config(str(p))
    assert cfg.bloat_ratio == 100.0 and cfg.hub_share == 0.99 and cfg.dead_min_coverage == 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bloat_ratoi": 2}))
    with pytest.raises(ValueError, match="bloat_ratoi"):
        load_smell_config(str(bad))
    with pytest.raises(ValueError):
        resolve_smell_config({"nope": 1})
    assert resolve_smell_config(None) == SmellConfig()
    traj = demo.runs.trajectories[0]
    assert "context_bloat" not in names(compute_coordination(traj, smell_config=cfg))
    for spec in (cfg, {"bloat_ratio": 100.0}, str(p)):
        res = analyze_runs(demo.runs.__class__("x", "b", [traj]), smell_config=spec)
        assert "context_bloat" not in res.coordination.diagnostics.names


def test_detect_smells_runs_on_precomputed_metrics(demo):
    cm = demo.per_run_coordination[0]
    again = detect_smells(demo.runs.trajectories[0], cm, SmellConfig())
    assert [d.to_dict() for d in again] == [d.to_dict() for d in cm.diagnostics]
    assert all(d.smell in SMELLS for d in again)


@pytest.fixture
def smell_registry():
    from swarmeval.diagnostics import smells as S
    saved, saved_smells = dict(S._REGISTRY), list(SMELLS)
    yield S
    S._REGISTRY.clear()
    S._REGISTRY.update(saved)
    SMELLS[:] = saved_smells


def test_register_smell_runs_custom_detector(demo, smell_registry):
    from swarmeval import SmellContext, register_smell

    def custom(c):
        assert isinstance(c, SmellContext)
        c.add("custom_smell", "low", c.agents[:1], "observed", "may", "re-run", {"n": 1})

    traj, cm = demo.runs.trajectories[0], demo.per_run_coordination[0]
    builtin = [d.to_dict() for d in detect_smells(traj, cm)]
    register_smell("custom_smell", custom)
    assert SMELLS[-1] == "custom_smell"
    out = detect_smells(traj, cm)
    assert [d.to_dict() for d in out[:-1]] == builtin
    assert out[-1].smell == "custom_smell" and out[-1].evidence == {"n": 1} and out[-1].run_id == traj.run_id
    register_smell("custom_smell", lambda c: None)
    assert SMELLS.count("custom_smell") == 1
    assert [d.to_dict() for d in detect_smells(traj, cm)] == builtin


def test_late_and_premature_registers_two_smells(smell_registry):
    S = smell_registry
    assert list(S._REGISTRY) == ["redundant_delegation", "agent_ping_pong", "over_agentization",
                                 "under_specialization", "coordinator_bottleneck", "late_and_premature",
                                 "context_bloat", "dead_agent", "serialized_execution", "communication_overhead"]
    assert S._REGISTRY["late_and_premature"] is S._late_and_premature
    assert SMELLS == ["redundant_delegation", "agent_ping_pong", "over_agentization", "under_specialization",
                      "coordinator_bottleneck", "late_information", "premature_synthesis", "context_bloat",
                      "dead_agent", "serialized_execution", "communication_overhead"]


def D(smell, sev, run, agents=(), inferred=False, obs=None):
    return Diagnostic(smell, sev, run, list(agents), obs or f"{smell} {sev} in {run}", "may", "test", {}, inferred)


def test_engine_counts_each_run_once_and_orders_by_severity_then_prevalence():
    per_run = {
        "r1": [D("dead_agent", "medium", "r1", ["x"]), D("dead_agent", "medium", "r1", ["y"]),
               D("context_bloat", "low", "r1", ["x"], inferred=True)],
        "r2": [D("dead_agent", "medium", "r2", ["x"]), D("dead_agent", "high", "r2", ["x"]),
               D("context_bloat", "low", "r2", ["z"])],
        "r3": [D("agent_ping_pong", "high", "r3", ["a", "b"])],
        "r4": [],
    }
    rep = summarize_diagnostics(per_run)
    assert rep.n_runs == 4 and rep.total == 7
    assert rep.names == ["dead_agent", "agent_ping_pong", "context_bloat"]
    dead = rep.get("dead_agent")
    assert dead.runs_affected == 2 and dead.prevalence == 0.5
    assert dead.severity_counts == {"medium": 1, "high": 1} and dead.max_severity == "high"
    assert dead.agents == {"x": 2, "y": 1}
    assert dead.observation == "dead_agent high in r2" and dead.example_runs == ["r2", "r1"]
    bloat = rep.get("context_bloat")
    assert bloat.inferred_runs == 1 and bloat.observation == "context_bloat low in r1"
    as_list = summarize_diagnostics([per_run[k] for k in ("r1", "r2", "r3", "r4")])
    assert [s.runs_affected for s in as_list.smells] == [2, 1, 2]
    assert summarize_diagnostics({}).smells == [] and summarize_diagnostics([[]], n_runs=5).n_runs == 5
    json.dumps(rep.to_dict())


def test_runs_sharing_a_run_id_are_all_counted():
    ctx = make_ctx(difficulty="simple")
    for a in ("a", "b", "c"):
        with ctx.agent(a) as s:
            llm(s)
    traj = ctx.finish()
    res = analyze_runs(RunSet("l", "b", [traj, copy.deepcopy(traj)]))
    over = res.coordination.diagnostics.get("over_agentization")
    assert over.runs_affected == 2 and over.prevalence == 1.0


def test_engine_keeps_at_most_three_examples():
    rep = summarize_diagnostics({f"r{i}": [D("dead_agent", "medium", f"r{i}")] for i in range(6)})
    assert rep.get("dead_agent").example_runs == ["r0", "r1", "r2"]


def test_no_causal_or_credit_language_anywhere(demo):
    texts = []
    rounds = evaluate(partial(make_swarm, review_rounds=3), build_benchmark(2), trials=1, clock_factory=SimClock)
    serial = analyze_runs(demo.runs.__class__("serial", "b", [fan_out_trajectory(parallel=False)]))
    chatty = exchange(9)
    for res in (demo, rounds, serial):
        texts.append(res.report())
        texts.append(render_coordination(res))
        for cm in res.per_run_coordination:
            texts += [d.observation + " " + d.hypothesis + " " + d.intervention for d in cm.diagnostics]
    texts += [d.observation + " " + d.hypothesis + " " + d.intervention for d in chatty.diagnostics]
    seen = {d.smell for res in (demo, rounds, serial) for cm in res.per_run_coordination for d in cm.diagnostics}
    assert {"agent_ping_pong", "serialized_execution", "dead_agent", "context_bloat"} <= seen
    for t in texts:
        for pat in BANNED:
            assert not pat.search(t), (pat.pattern, t)
    assert "hypotheses to test, not verdicts" in demo.report()
    assert "provenance, not marginal effect" in demo.report()


HUB = "research_coordinator_agent_07"


def hub_run():
    ctx = make_ctx()
    with ctx.agent(HUB) as h:
        for p in ("p1", "p2", "p3"):
            h.send(p, "do x")
    for p in ("p1", "p2", "p3"):
        with ctx.agent(p) as s:
            s.receive()
            s.send(HUB, "done")
    return analyze_runs(RunSet("l", "b", [ctx.finish()]))


def test_hub_share_row_keeps_long_agent_id():
    (line,) = [ln for ln in render_coordination(hub_run()).splitlines() if "Hub Share" in ln]
    assert HUB in line


def test_contribution_row_keeps_long_agent_id():
    text = render_coordination(hub_run())
    section = text.split("OBSERVED CONTRIBUTION", 1)[1]
    assert any(ln.startswith("  " + HUB + " ") for ln in section.splitlines())


def test_coordination_box_shows_a_run_whose_analysis_failed_alone(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]

    def odd(traj, **kw):
        raise KeyError("odd trace")

    monkeypatch.setattr(ev, "compute_coordination", odd)
    with pytest.warns(UserWarning, match="failed on 1/1 runs"):
        res = analyze_runs(RunSet("l", "b", [fan_out_trajectory(parallel=True)]))
    assert res.coordination is None
    assert "Analysis Failed" in render_coordination(res) and "1/1 runs: KeyError" in render_coordination(res)


def test_coordination_box_shows_failed_runs(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]
    real, calls = ev.compute_coordination, []

    def flaky(traj, **kw):
        calls.append(traj.run_id)
        if len(calls) % 2 == 0:
            raise KeyError("odd trace " + "x" * 200)
        return real(traj, **kw)

    monkeypatch.setattr(ev, "compute_coordination", flaky)
    traj = fan_out_trajectory(parallel=True)
    runs = [copy.deepcopy(traj) for _ in range(4)]
    for i, t in enumerate(runs):
        t.run_id = f"r{i}"
    with pytest.warns(UserWarning, match="failed on 2/4 runs"):
        res = analyze_runs(RunSet("l", "b", runs))
    assert len(res.coordination_errors) == 2
    box = render_coordination(res).split("OBSERVED CONTRIBUTION", 1)[0]
    (line,) = [ln for ln in box.splitlines() if "Analysis Failed" in ln]
    assert "2/4 runs: KeyError: 'odd trace" in line and line.endswith("│")
