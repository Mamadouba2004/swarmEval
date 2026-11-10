"""Redundancy: tool-call keys and duplicate classes, artifact redundancy."""
import json

import pytest

from swarmeval import (EmbeddingSimilarity, LexicalSimilarity, RunContext, SimClock, Task, Trajectory,
                       compute_redundancy, evaluate)
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.metrics import canonical_args, summarize_redundancy


def make_ctx(**kw):
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0, **kw)
    ctx.start()
    return ctx


def search(ctx, span, query, fail=False):
    def fn(query):
        ctx.clock.advance(1.0)
        if fail:
            raise RuntimeError("timeout")
        return [query]
    try:
        span.call_tool("web_search", fn, query=query)
    except RuntimeError:
        pass


def test_canonical_args():
    assert canonical_args({"query": "S1 "}) == canonical_args({"query": "s1"})
    assert canonical_args({"q": "a   b", "n": 3}) == canonical_args({"n": 3, "q": " A b"})
    assert canonical_args({"query": "s1"}) != canonical_args({"query": "s2"})
    assert canonical_args({"n": 1}) != canonical_args({"n": "1"})          # numbers stay numbers
    assert canonical_args({"x": [" A ", {"B": "C"}]}) == canonical_args({"x": ["a", {"B": "c"}]})
    obj = object()
    assert canonical_args({"x": obj}) == repr({"x": obj})                   # unserializable -> repr


def test_normalized_args_are_duplicates():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        search(ctx, a, "S1 ")
        search(ctx, a, "other")
    with ctx.agent("b") as b:
        search(ctx, b, "s1")
        search(ctx, b, "s2")
    r = compute_redundancy(ctx.finish())
    assert r.tool_calls == 4 and r.duplicate_tool_calls == 1 and r.accidental_duplicates == 1
    assert r.tool_redundancy == pytest.approx(1 / 4)
    g = r.duplicate_groups[0]
    assert g["original_agent"] == "a" and g["duplicates"][0]["agent"] == "b"
    assert r.pairwise_overlap == [{"agents": ["a", "b"], "shared": 1, "jaccard": pytest.approx(1 / 3)}]


def test_duplicate_classes():
    ctx = make_ctx()
    with ctx.agent("researcher", role="research") as r:
        search(ctx, r, "q1")
        search(ctx, r, "q2", fail=True)
        search(ctx, r, "q2")              # retry: its own previous q2 call failed
        search(ctx, r, "q3")
        search(ctx, r, "q3")              # self repeat, not a retry -> accidental
    with ctx.agent("critic", role="critic") as c:
        search(ctx, c, "q1")              # checking role -> verification
    with ctx.agent("auditor", intent="verification") as au:
        search(ctx, au, "q3")             # span marked intent=verification
    with ctx.agent("other", role="research") as o:
        search(ctx, o, "q1")              # plain repeat by another agent -> accidental
    r = compute_redundancy(ctx.finish())
    assert r.tool_calls == 8 and r.duplicate_tool_calls == 5
    assert (r.accidental_duplicates, r.verification_duplicates, r.retry_duplicates) == (2, 2, 1)
    assert r.self_repeats == 1
    assert r.tool_redundancy == pytest.approx(2 / 8)      # intentional duplicates left out
    classes = {(d["agent"], d["class"]) for g in r.duplicate_groups for d in g["duplicates"]}
    assert classes == {("researcher", "retry"), ("researcher", "accidental"), ("critic", "verification"),
                       ("auditor", "verification"), ("other", "accidental")}


def test_call_intent_verification_and_no_tools():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        search(ctx, a, "q")
        with a.tool_call("web_search", {"query": "q"}) as call:
            call.metadata["intent"] = "verification"
            call.result = ["q"]
    r = compute_redundancy(ctx.finish())
    assert r.verification_duplicates == 1 and r.tool_redundancy == 0.0
    ctx = make_ctx()
    with ctx.agent("a") as a:
        a.send("b", "hi")
    r = compute_redundancy(ctx.finish())
    assert r.tool_redundancy is None and r.artifact_redundancy is None


def test_long_args_compared_in_full_with_capture_off():
    """capture_text=False keeps a 120-char preview of the args; calls sharing a
    long prefix are still distinct, and dict key order does not matter."""
    prefix = "x" * 200
    for traj in (lambda t: t, lambda t: Trajectory.from_dict(t.to_dict())):
        ctx = make_ctx(capture_text=False)
        with ctx.agent("a") as a:
            search(ctx, a, prefix + "one")
            a.call_tool("lookup", lambda **kw: 1, x=1, y=2)
        with ctx.agent("b") as b:
            search(ctx, b, prefix + "two")
            b.call_tool("lookup", lambda **kw: 1, y=2, x=1)       # same args, other key order
        r = compute_redundancy(traj(ctx.finish()))
        assert r.duplicate_tool_calls == 1 and r.duplicate_groups[0]["tool"] == "lookup"
        assert not r.tool_degraded


def test_old_data_without_args_hash_is_degraded():
    ctx = make_ctx(capture_text=False)
    with ctx.agent("a") as a:
        search(ctx, a, "q")
    d = ctx.finish().to_dict()
    for ev in d["events"]:
        ev["metadata"].pop("args_hash", None)
    r = compute_redundancy(Trajectory.from_dict(d))
    assert r.tool_calls == 1 and r.tool_degraded


def _artifacts(**ctx_kw):
    ctx = make_ctx(**ctx_kw)
    with ctx.agent("a") as a:
        orig = a.produce("notes", "Aurora was first released in 2015 by Tern", derived_from=[])
        a.produce("notes2", "Aurora was first released in 2015 by Tern", derived_from=[])   # same producer
    ctx.clock.advance(1.0)
    with ctx.agent("b") as b:
        b.produce("dup", "Aurora was first released in 2015 by Tern", derived_from=[])      # redundant
        b.produce("copy", "Aurora was first released in 2015 by Tern", derived_from=[orig])  # derived copy
    with ctx.agent("c") as c:
        c.consume(orig)
        c.produce("used", "Aurora was first released in 2015 by Tern")                    # consumed copy
    with ctx.agent("d") as d:
        d.produce("near", "aurora was first released in 2015 by Tern labs", derived_from=[])  # similar
    with ctx.agent("v", role="verifier") as v:
        v.produce("check", "Aurora was first released in 2015 by Tern", derived_from=[])    # intentional
    with ctx.agent("e") as e:
        e.produce("fresh", "Something else entirely about Helix", derived_from=[])
    return ctx.finish()


def test_artifact_redundancy():
    r = compute_redundancy(_artifacts())
    names = {p["name"]: p for p in r.redundant_artifact_pairs}
    assert set(names) == {"dup", "near", "check"}
    assert names["dup"]["exact"] and names["dup"]["matches_producer"] == "a"
    assert not names["near"]["exact"] and names["near"]["score"] >= 0.6
    assert names["check"]["class"] == "verification"
    assert r.redundant_artifacts == 2 and r.intentional_artifacts == 1
    assert r.artifact_redundancy == pytest.approx(2 / 8)
    assert r.method == "lexical" and not r.degraded and r.similarity_threshold == 0.6
    json.dumps(r.to_dict())


def test_artifact_redundancy_degraded_without_text():
    """Without captured text only exact hashes are compared: the near copy is missed.
    The text is dropped when the run is saved, so this goes through a round trip."""
    r = compute_redundancy(Trajectory.from_dict(_artifacts(capture_text=False).to_dict()))
    assert r.degraded and r.method == "hash" and r.similarity_threshold is None
    assert {p["name"] for p in r.redundant_artifact_pairs} == {"dup", "check"}
    assert r.redundant_artifacts == 1


def test_forwarded_artifact_is_not_redundant():
    """A coordinator re-producing the answer a worker sent it is passing it on."""
    ctx = make_ctx()
    with ctx.agent("worker") as w:
        ans = w.produce("answer", "Aurora was first released in 2015 by Tern", derived_from=[])
        w.send("lead", "done", kind="result", artifacts=[ans])
    ctx.clock.advance(1.0)
    with ctx.agent("lead") as lead:
        lead.receive()
        lead.produce("final", "Aurora was first released in 2015 by Tern")
    r = compute_redundancy(ctx.finish())
    assert r.artifacts == 2 and r.redundant_artifacts == 0


def test_artifact_redundancy_with_embeddings():
    """Any backend plugs in; this fake embedding calls every 'Aurora' text the same."""
    emb = EmbeddingSimilarity(embed_fn=lambda texts: [[1.0, 0.0] if "aurora" in t.lower() else [0.0, 1.0]
                                                      for t in texts])
    r = compute_redundancy(_artifacts(), similarity=emb)
    assert r.method == "embeddings" and r.similarity_threshold == 0.85
    assert {p["name"] for p in r.redundant_artifact_pairs} == {"dup", "near", "check"}


def test_summary_counts_similarity_truncated_runs():
    runs = [compute_redundancy(_artifacts(), similarity=LexicalSimilarity(max_pairs=1)),
            compute_redundancy(_artifacts())]
    assert runs[0].similarity_truncated and not runs[1].similarity_truncated
    s = summarize_redundancy(runs)
    assert s.similarity_truncated_runs == 1 and s.to_dict()["similarity_truncated_runs"] == 1


def test_golden_demo_redundancy():
    """researcher_c is given researcher_a's scope (s1, s2), so its two
    web_search calls repeat a's: 2 accidental duplicates of 6 tool calls.
    Their notes come from the same sources but each researcher keeps each
    fact with probability 0.96, so the notes are identical in some runs, close
    in others (lexical >= 0.6) and further apart when one dropped a fact.
    Whichever finished first is the original; the other is redundant."""
    res = evaluate(make_swarm, build_benchmark(2), trials=3, clock_factory=SimClock)
    lex = LexicalSimilarity()
    flagged = 0
    for traj, cm in zip(res.runs.trajectories, res.per_run_coordination):
        r = cm.redundancy
        assert r.tool_calls == 6 and r.accidental_duplicates == 2 and r.tool_redundancy == pytest.approx(2 / 6)
        assert r.verification_duplicates == 0 and r.retry_duplicates == 0 and r.self_repeats == 0
        assert all(g["original_agent"] == "researcher_a" and g["duplicates"][0]["agent"] == "researcher_c"
                   for g in r.duplicate_groups)
        assert r.pairwise_overlap == [{"agents": ["researcher_a", "researcher_c"], "shared": 2, "jaccard": 1.0}]
        # artifact redundancy: exactly the a/c notes pair, when they are similar enough
        notes = {a.name: str(a.content) for a in traj.artifacts.values()}
        similar = lex.similarity(notes["notes_a"], notes["notes_c"]) >= lex.threshold
        assert r.artifacts == 5 and r.method == "lexical"
        assert r.redundant_artifacts == int(similar) and r.artifact_redundancy == pytest.approx(int(similar) / 5)
        for p in r.redundant_artifact_pairs:
            assert {p["producer"], p["matches_producer"]} == {"researcher_a", "researcher_c"}
        flagged += r.redundant_artifacts
    assert flagged == 4        # 4 of these 6 seeded runs (2 identical, 2 near-identical notes)
    s = summarize_redundancy([cm.redundancy for cm in res.per_run_coordination])
    assert s.tool_redundancy.mean == pytest.approx(1 / 3) and s.methods == {"lexical": 6}
    assert res.coordination.redundancy.artifact_redundancy.n == 6
    json.dumps(res.to_dict())
