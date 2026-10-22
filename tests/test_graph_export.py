"""Graph export (graph/export.py), the `swarmeval graph` CLI, compare's coordination rows."""
import json

import pytest

from swarmeval import (EvaluationResult, RunContext, SimClock, Task, compare, compute_coordination, evaluate,
                       export_graph, to_dot, to_json, to_mermaid)
from swarmeval.cli import main
from swarmeval.compare import METRICS
from swarmeval.evaluate import analyze_runs, load_runset
from swarmeval.examples.research_swarm import build_benchmark, make_swarm

FIXTURE = "tests/fixtures/phase1_runs.json"
HOSTILE = 'evil"] --> x[<script>alert(1)</script>\nnext|{}'


@pytest.fixture(scope="module")
def demo():
    return evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)


def test_mermaid_agents_view(demo):
    cm = demo.per_run_coordination[0]
    text = to_mermaid(cm.graph, coordination=cm)
    assert text == to_mermaid(cm.graph, coordination=cm)              # deterministic
    lines = text.splitlines()
    assert lines[0] == "flowchart LR"
    assert '    a3["researcher_c<br/>3% tokens"]:::dead' in lines
    assert any(ln.startswith("    classDef dead") and "stroke-dasharray" in ln for ln in lines)
    # receives were recorded, so no edge is dashed even though use of task assignments is judged by timing
    assert "-.->" not in text
    assert '    a0 -->|"message ×1 · 186 tok"| a4' in lines
    # researcher_a's notes went to the coordinator, the critic and the synthesizer
    assert sum(1 for ln in lines if ln.startswith("    a1 -->|\"artifact ×1 · ")) == 3


def test_inferred_edges_are_dashed_on_phase1_data():
    res = analyze_runs(load_runset(FIXTURE))
    cm = res.per_run_coordination[0]
    mm = to_mermaid(cm.graph, coordination=cm)
    assert '    a0 -.->|"message ×1 · 16 tok · 1 inferred"| a1' in mm.splitlines()
    dot = to_dot(cm.graph, coordination=cm)
    assert 'a0 -> a1 [label="message ×1 · 16 tok · 1 inferred", style=dashed];' in dot
    assert 'a3 [label="researcher_c\\n3% tokens", style=dashed, color=red, fontcolor=red];' in dot
    data = json.loads(to_json(cm.graph, coordination=cm))
    assert data["coverage"]["message_receive_coverage"] == 0.0
    assert any(e["delivery_inferred"] for e in data["edges"])
    assert next(n for n in data["nodes"] if n["id"] == "researcher_c")["dead"] is True


def test_dag_view_groups_by_agent_and_marks_the_critical_path(demo):
    cm = demo.per_run_coordination[0]
    mm = to_mermaid(cm.graph, view="dag", coordination=cm)
    assert mm.startswith("flowchart TB\n") and '    subgraph g0["coordinator"]' in mm
    assert mm.count(":::cp") == len(cm.dependency.critical_path.node_ids)
    dot = to_dot(cm.graph, view="dag", coordination=cm)
    assert "subgraph cluster_0 {" in dot and 'label="coordinator";' in dot
    assert dot.count('fillcolor="#fde68a"') == len(cm.dependency.critical_path.node_ids)
    data = json.loads(to_json(cm.graph, view="dag", coordination=cm))
    assert data["critical_path"] == cm.dependency.critical_path.node_ids
    assert sum(n["critical"] for n in data["nodes"]) == len(data["critical_path"])
    with pytest.raises(ValueError, match="view"):
        to_mermaid(cm.graph, view="nope")
    with pytest.raises(ValueError, match="format"):
        export_graph(cm.graph, "svg")


def test_hostile_agent_ids_are_escaped():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    with ctx.agent(HOSTILE) as a:
        with a.llm_call() as c:
            ctx.clock.advance(1.0)
            c.input_tokens = c.output_tokens = 10
        a.send("ok", "hi")
    with ctx.agent("ok") as b:
        b.receive()
    traj = ctx.finish()
    cm = compute_coordination(traj)
    for view in ("agents", "dag"):
        mm = to_mermaid(cm.graph, view=view, coordination=cm)
        assert "<script>" not in mm and "\nnext" not in mm
        assert "evil#quot;#93; --&gt; x#91;&lt;script&gt;" in mm
        dot = to_dot(cm.graph, view=view, coordination=cm)
        assert "<script>" not in dot and "\nnext" not in dot
        assert 'evil\\"] --&gt; x[&lt;script&gt;alert(1)&lt;/script&gt;\\nnext|{}' in dot
        assert json.loads(to_json(cm.graph, view=view, coordination=cm))


def test_graph_cli_writes_each_format(tmp_path, demo, capsys):
    saved = tmp_path / "res.json"
    demo.save(str(saved))
    out = {}
    for fmt in ("mermaid", "dot", "json"):
        p = tmp_path / f"g.{fmt}"
        assert main(["graph", str(saved), "--format", fmt, "--out", str(p)]) == 0
        out[fmt] = p.read_text()
    assert out["mermaid"].startswith("flowchart LR") and ":::dead" in out["mermaid"]
    assert out["dot"].startswith("digraph") and "style=dashed, color=red" in out["dot"]
    assert json.loads(out["json"])["view"] == "agents"
    capsys.readouterr()
    assert main(["graph", str(saved), "--run", "1", "--view", "dag"]) == 0
    assert "subgraph" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["graph", str(saved), "--run", "9"])


def test_cli_coordination_flags(tmp_path, demo, capsys):
    saved = tmp_path / "res.json"
    demo.save(str(saved))
    assert main(["report", str(saved)]) == 0
    assert "DIAGNOSTICS" in capsys.readouterr().out
    assert main(["report", str(saved), "--no-coordination"]) == 0
    assert "COORDINATION" not in capsys.readouterr().out
    cfg = tmp_path / "smells.json"
    cfg.write_text(json.dumps({"bloat_ratio": 1000.0}))
    assert main(["report", str(saved), "--smells-config", str(cfg), "--similarity", "lexical"]) == 0
    text = capsys.readouterr().out
    assert "DIAGNOSTICS" in text and "context_bloat" not in text
    cfg.write_text(json.dumps({"typo": 1}))
    with pytest.raises(SystemExit, match="typo"):
        main(["report", str(saved), "--smells-config", str(cfg)])
    assert main(["compare", str(saved), str(saved), "--no-coordination"]) == 0
    assert "Critical Path" in capsys.readouterr().out    # row present, value unknown


def test_compare_shows_coordination_metrics_with_neutral_direction(demo):
    neutral = {k for k, _, hib, _ in METRICS if hib is None}
    assert neutral == {"potential_parallelism", "comm_work_ratio"}
    rounds = evaluate(lambda: make_swarm(review_rounds=3), build_benchmark(2), trials=1, clock_factory=SimClock,
                      label="rounds")
    cmp = compare(demo, rounds)
    comm = cmp.metric("comm_work_ratio")
    assert comm.verdict == "changed" and comm.higher_is_better is None
    assert "comm_work_ratio" not in cmp.improved + cmp.regressed
    assert cmp.metric("smells_per_run").verdict == "regressed"           # ping-pong added
    assert cmp.metric("critical_path_s").verdict == "regressed"
    text = cmp.report()
    assert "Communication / Work" in text and "changed (no better direction): " in text
    off = compare(demo, rounds, coordination=False)
    assert off.metric("critical_path_s").verdict == "unknown" and off.metric("success_rate").verdict == "unchanged"
    json.dumps(cmp.to_dict())


def test_result_to_dict_is_json_serializable(demo, tmp_path):
    d = demo.to_dict()
    text = json.dumps(d)
    assert '"diagnostics"' in text and '"dependency"' in text and '"context"' in text
    assert d["coordination"]["diagnostics"]["n_runs"] == len(demo.runs)
    p = tmp_path / "r.json"
    demo.save(str(p))
    back = EvaluationResult.load(str(p))
    assert back.coordination.diagnostics.names == demo.coordination.diagnostics.names
