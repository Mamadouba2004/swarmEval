"""Swarm smells (diagnostics/smells.py) and their prevalence over runs (diagnostics/engine.py)."""
import copy
import json
import re
from functools import partial

import pytest

from swarmeval import (Diagnostic, RunContext, RunSet, SimClock, SmellConfig, Task, analyze_runs, compute_coordination,
                       evaluate, load_smell_config, summarize_diagnostics)
from swarmeval.diagnostics.smells import SMELLS, detect_smells, resolve_smell_config
from swarmeval.examples.research_swarm import ResearchSwarm, build_benchmark, make_swarm
from swarmeval.report.text import render_coordination

from test_metrics import fan_out_trajectory

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


# --------------------------------------------------------------------------- #
# golden demo
# --------------------------------------------------------------------------- #
def test_golden_demo_smells(demo):
    expected = {"coordinator_bottleneck", "dead_agent", "under_specialization", "redundant_delegation",
                "context_bloat"}
    for cm in demo.per_run_coordination:
        assert names(cm) == expected
        assert [d.agents for d in of(cm, "dead_agent")] == [["researcher_c"]]
        assert of(cm, "under_specialization")[0].agents == ["researcher_a", "researcher_c"]
        assert of(cm, "redundant_delegation")[0].agents == ["coordinator", "researcher_a", "researcher_c"]
        assert of(cm, "coordinator_bottleneck")[0].agents == ["coordinator"]
        # the synthesizer (extra 9000 tokens) and the coordinator's hand-off call (extra 2600) are both
        # well above the threshold: a correct observation, not noise
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
    assert ev["relay_fraction"] == pytest.approx(0.4)          # below 0.5: fires on the critical-path share
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
        assert cm.dependency.cycles == []           # request/response round trips are not cycles
        assert "premature_synthesis" not in names(cm) and "late_information" not in names(cm)


# --------------------------------------------------------------------------- #
# individual smells
# --------------------------------------------------------------------------- #
def exchange_traj(n_messages):
    """a and b take turns: n messages, n - 1 changes of direction."""
    ctx = make_ctx()
    for i in range(n_messages):
        src, dst = ("a", "b") if i % 2 == 0 else ("b", "a")
        with ctx.agent(src) as s:
            s.receive()
            llm(s)
            s.send(dst, f"round {i}", kind="feedback")
    with ctx.agent("b" if n_messages % 2 else "a") as s:
        s.receive()
    return ctx.finish()


def exchange(n_messages, **kw):
    return compute_coordination(exchange_traj(n_messages), **kw)


def test_ping_pong_thresholds():
    assert not of(exchange(4), "agent_ping_pong")                     # 3 alternations
    six = of(exchange(7), "agent_ping_pong")                          # 6 alternations
    assert len(six) == 1 and six[0].severity == "medium" and six[0].evidence["alternations"] == 6
    assert of(exchange(9), "agent_ping_pong")[0].severity == "high"  # 8 alternations
    assert not of(exchange(7, smell_config={"pingpong_min_alternations": 7}), "agent_ping_pong")


def test_sequential_delegation_is_not_ping_pong():
    ctx = make_ctx()
    for i in range(6):                    # six new tasks, each answered: round trips, not back-and-forth
        with ctx.agent("boss") as b:
            b.receive()
            b.send("worker", f"task {i}", kind="task_assignment")
        with ctx.agent("worker") as w:
            w.receive()
            llm(w)
            w.send("boss", f"done {i}", kind="result")
    assert not of(compute_coordination(ctx.finish()), "agent_ping_pong")


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
    assert "over_agentization" not in names(many)     # nobody unused, task difficulty unknown


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
        w.receive()                       # read, but nothing produced afterwards
    ctx.set_output("the draft", artifacts=[final])
    cm = compute_coordination(ctx.finish())
    pre = of(cm, "premature_synthesis")
    assert len(pre) == 1 and pre[0].severity == "high" and pre[0].evidence["final_producer"]
    assert not of(cm, "late_information")


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
    assert not of(asked_then_answered(True, reply_to), "premature_synthesis")      # e.g. a review approval
    pre = of(asked_then_answered(False, reply_to), "premature_synthesis")         # produced without waiting
    assert len(pre) == 1 and pre[0].severity == "high"


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
        b.receive()                      # b's next call lists no inputs: use is inferred, not dead
        llm(b)
    cm = compute_coordination(ctx.finish())
    assert cm.contribution["a"].dead is False and not of(cm, "dead_agent")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
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
    assert rep.names == ["dead_agent", "agent_ping_pong", "context_bloat"]   # high (50%), high (25%), low
    dead = rep.get("dead_agent")
    assert dead.runs_affected == 2 and dead.prevalence == 0.5
    assert dead.severity_counts == {"medium": 1, "high": 1} and dead.max_severity == "high"
    assert dead.agents == {"x": 2, "y": 1}                     # x once per run, not three times
    assert dead.observation == "dead_agent high in r2" and dead.example_runs == ["r2", "r1"]
    bloat = rep.get("context_bloat")
    assert bloat.inferred_runs == 1 and bloat.observation == "context_bloat low in r1"   # tie: earliest run
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
    res = analyze_runs(RunSet("l", "b", [traj, copy.deepcopy(traj)]))       # e.g. merged run files
    over = res.coordination.diagnostics.get("over_agentization")
    assert over.runs_affected == 2 and over.prevalence == 1.0


def test_engine_keeps_at_most_three_examples():
    rep = summarize_diagnostics({f"r{i}": [D("dead_agent", "medium", f"r{i}")] for i in range(6)})
    assert rep.get("dead_agent").example_runs == ["r0", "r1", "r2"]


# --------------------------------------------------------------------------- #
# language
# --------------------------------------------------------------------------- #
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
