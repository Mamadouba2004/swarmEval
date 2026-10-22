"""Standalone HTML report: content, escaping, self-containment, CLI."""
import html as htmllib
import re

import pytest

from swarmeval import FunctionSwarm, evaluate, render_html
from swarmeval.cli import main
from swarmeval.evaluate import analyze_runs, load_runset
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.recorder import SimClock
from swarmeval.schema import Benchmark, Task

FIXTURE = "tests/fixtures/phase1_runs.json"
HOSTILE = "<script>alert(1)</script>"
QUOTED_TASK = 'task "quoted" \'single\' & <b>'


@pytest.fixture(scope="module")
def demo():
    return evaluate(make_swarm, build_benchmark(), trials=2, clock_factory=SimClock)


@pytest.fixture(scope="module")
def demo_html(demo):
    return render_html(demo)


def visible_text(doc: str) -> str:
    """Text a reader sees: no style block, no tags, entities decoded."""
    doc = re.sub(r"<style>.*?</style>", " ", doc, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", doc))


def assert_self_contained(doc: str) -> None:
    assert "http://" not in doc and "https://" not in doc
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:|//)""", doc, re.I)
    assert not re.search(r"""url\(\s*["']?\s*(?!#)""", doc), "only same-document url(#id) references allowed"
    assert not re.search(r"<script", doc, re.I)
    assert not re.search(r"<link\b|@import", doc, re.I)


def test_demo_renders_graph_timeline_and_sections(demo_html):
    doc = demo_html
    assert doc.startswith("<!doctype html>")
    assert doc.count("<svg") == 2
    assert 'aria-label="agent graph"' in doc and 'aria-label="timeline"' in doc
    assert re.search(r'class="[^"]*\bcp\b', doc)
    assert "on the critical path" in doc
    assert "Observed contribution (provenance, not marginal effect)" in doc
    assert "Diagnostics — hypotheses to test, not verdicts" in doc
    for smell in ("dead_agent", "coordinator_bottleneck", "context_bloat"):
        assert smell in doc
    assert "Observed:" in doc and "Hypothesis:" in doc and "Test:" in doc
    # coordination tiles reuse the text-report labels, with their definitions as captions
    assert "Actual Parallelism" in doc and "work ÷ wall" in doc
    assert "Communication / Work" in doc and "message ÷ LLM tokens" in doc
    assert "no overall score" in doc
    assert "prefers-color-scheme: dark" in doc
    assert_self_contained(doc)


def test_dead_agent_and_hub_marked_in_graph(demo_html):
    graph = re.search(r'<svg class="fit".*?</svg>', demo_html, re.S).group()
    dead = re.search(r'<g><title>(researcher_c[^<]*)</title><rect class="n dead', graph)
    assert dead, "researcher_c should be drawn as dead"
    assert graph.count('class="n dead') == 1
    assert re.search(r'<title>coordinator[^<]*hub', graph)
    assert "Hub agent: <b>coordinator</b>" in demo_html


def test_run_index_selects_run_and_is_checked(demo):
    doc = render_html(demo, run_index=1)
    assert "run 1: task" in doc
    with pytest.raises(ValueError):
        render_html(demo, run_index=len(demo.runs))


def _hostile_result():
    def run(task, ctx):
        with ctx.agent(HOSTILE, role='r"ole') as a:
            with a.llm_call(model="m", input_tokens=10, output_tokens=5):
                ctx.clock.sleep(1.0)
            art = a.produce('note "x"', "content")
            with ctx.agent("worker") as w:
                w.consume(art)
                w.send(HOSTILE, "done", kind="result")
        return "out"
    bench = Benchmark("bench <i>", [Task(task_id=QUOTED_TASK, input="q")])
    return evaluate(FunctionSwarm(run), bench, trials=1, clock_factory=SimClock, label='lab"el<x>')


def test_user_strings_are_escaped():
    doc = render_html(_hostile_result())
    assert HOSTILE not in doc and "<script" not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc
    assert QUOTED_TASK not in doc
    assert "task &quot;quoted&quot; &#x27;single&#x27; &amp; &lt;b&gt;" in doc
    assert 'lab"el<x>' not in doc and "<i>" not in doc
    assert_self_contained(doc)


def test_phase1_fixture_renders_with_inferred_caveat(tmp_path):
    res = analyze_runs(load_runset(FIXTURE))
    doc = render_html(res, path=str(tmp_path / "p1.html"))
    assert (tmp_path / "p1.html").read_text(encoding="utf-8") == doc
    assert "inferred from timing" in doc
    assert 'class="e message inf"' in doc
    assert doc.count("<svg") == 2
    assert_self_contained(doc)


def test_without_coordination_renders_phase1_and_skip_note():
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock, coordination=False)
    assert res.coordination is None
    doc = render_html(res)
    assert "Coordination analysis was skipped" in doc
    assert "Task Success" in doc and "Cost / Run" in doc
    assert "Observed contribution" not in doc and "Diagnostics" not in doc
    assert 'aria-label="timeline"' in doc and 'aria-label="agent graph"' not in doc
    assert_self_contained(doc)


def test_cli_report_and_demo_write_html(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main(["report", FIXTURE, "--html", str(out), "--run", "1"]) == 0
    doc = out.read_text(encoding="utf-8")
    assert "run 1: task" in doc and "<svg" in doc
    with pytest.raises(SystemExit):
        main(["report", FIXTURE, "--html", str(tmp_path / "bad.html"), "--run", "999"])
    demo_out = tmp_path / "demo.html"
    assert main(["demo", "--trials", "1", "--html", str(demo_out)]) == 0
    assert "Observed contribution" in demo_out.read_text(encoding="utf-8")
    assert "wrote HTML report" in capsys.readouterr().out


def test_language_lint(demo_html):
    text = visible_text(demo_html)
    for pattern in (r"\bcaus", r"contributed \d+%", r"responsible for"):
        assert not re.search(pattern, text, re.I), pattern
