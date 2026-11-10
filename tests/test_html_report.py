"""Standalone HTML report: content, escaping, self-containment, CLI."""
import html as htmllib
import json
import re
import sys

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


def test_cli_run_checked_before_work_and_requires_html(tmp_path, capsys):
    with pytest.raises(SystemExit, match="out of range"):
        main(["report", FIXTURE, "--html", str(tmp_path / "bad.html"), "--run", "999"])
    assert "Task Success" not in capsys.readouterr().out, "--run must be checked before the report is printed"
    with pytest.raises(SystemExit, match="out of range"):
        main(["demo", "--trials", "1", "--out", str(tmp_path / "out"), "--html", str(tmp_path / "d.html"),
              "--run", "999"])
    assert not (tmp_path / "out").exists(), "--run must be checked before outputs are written"
    for cmd in (["report", FIXTURE], ["demo", "--trials", "1"]):
        with pytest.raises(SystemExit) as exc:
            main(cmd + ["--run", "1"])
        assert exc.value.code == 2
        assert "--run requires --html" in capsys.readouterr().err


def test_trial_label_is_escaped():
    res = _hostile_result()
    res.runs.trajectories[0].trial = HOSTILE  # a hand-edited run file can carry any JSON value here
    doc = render_html(res)
    assert HOSTILE not in doc and "trial &lt;script&gt;" in doc


def hostile_trial_file(tmp_path):
    d = json.loads(open(FIXTURE).read())
    d["trajectories"][0]["trial"] = HOSTILE        # a hand-edited run file
    p = tmp_path / "hostile.json"
    p.write_text(json.dumps(d))
    return str(p), d["trajectories"][0]["run_id"]


def test_non_integer_trial_is_rejected_on_load(tmp_path):
    path, run_id = hostile_trial_file(tmp_path)
    with pytest.raises(ValueError, match=re.escape(f"run {run_id!r}: trial must be an integer")):
        load_runset(path)


def test_cli_report_rejects_non_integer_trial_cleanly(tmp_path):
    path, run_id = hostile_trial_file(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["report", path, "--html", str(tmp_path / "r.html")])
    assert str(exc.value.code).startswith(f"swarmeval: run {run_id!r}: trial must be an integer")


def test_timeline_tooltips_escape_recorded_strings():
    # Coverage-only regression test: these already pass; they pin the escaping
    # of model, tool name, sender and message kind in timeline tooltips.
    def run(task, ctx):
        with ctx.agent("worker") as w:
            with w.llm_call(model=HOSTILE + "m", input_tokens=1, output_tokens=1):
                ctx.clock.sleep(1.0)
            with w.tool_call(HOSTILE + "t"):
                ctx.clock.sleep(1.0)
        with ctx.agent(HOSTILE + "s") as s:
            s.send("worker", "hi", kind=HOSTILE + "k")
        return "out"
    res = evaluate(FunctionSwarm(run), Benchmark("b", [Task(task_id="t", input="q")]), trials=1,
                   clock_factory=SimClock)
    timeline = re.search(r'<svg[^>]*aria-label="timeline".*?</svg>', render_html(res), re.S).group()
    assert "<script" not in timeline
    tips = [htmllib.unescape(t) for t in re.findall(r"<title>([^<]*)</title>", timeline)]
    for s in ("m", "t", "s", "k"):
        assert any(HOSTILE + s in t for t in tips), s


def test_edge_tooltip_does_not_double_count_undelivered_messages():
    def run(task, ctx):
        with ctx.agent("worker") as w:
            with w.llm_call(model="m", input_tokens=1, output_tokens=1):
                ctx.clock.sleep(1.0)
        with ctx.agent("boss") as b:
            ctx.clock.sleep(1.0)
            b.send("worker", "late 1")  # worker never runs again: neither message is delivered
            b.send("worker", "late 2")
        return "out"
    res = evaluate(FunctionSwarm(run), Benchmark("b", [Task(task_id="t", input="q")]), trials=1,
                   clock_factory=SimClock)
    doc = render_html(res)
    tip = htmllib.unescape(re.search(r'class="e message inf"[^>]*><title>([^<]*)</title>', doc).group(1))
    assert "2 never delivered" in tip
    assert "inferred" not in tip, tip


def test_open_span_tooltip_matches_drawn_bar():
    def run(task, ctx):
        with ctx.agent("crashed") as a:
            with a.llm_call(model="m", input_tokens=1, output_tokens=1):
                ctx.clock.sleep(2.0)
        with ctx.agent("other"):
            ctx.clock.sleep(1.0)
        return "out"
    res = evaluate(FunctionSwarm(run), Benchmark("b", [Task(task_id="t", input="q")]), trials=1,
                   clock_factory=SimClock, coordination=False)
    traj = res.runs.trajectories[0]
    span = next(s for s in traj.spans.values() if s.agent_id == "crashed")
    span.end = None  # e.g. the process died before the span closed; the bar runs to the end of the run
    doc = render_html(res)
    assert re.search(r"<title>crashed span 3\.00s \(no end\)", doc), re.findall(r"<title>crashed span[^<]*", doc)


def test_failed_coordination_is_reported_not_called_skipped(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]   # the package attribute is the evaluate() function
    real, calls = ev.compute_coordination, []

    def flaky(traj, **kw):
        calls.append(traj.run_id)
        if len(calls) % 2 == 0:
            raise KeyError(HOSTILE)
        return real(traj, **kw)

    monkeypatch.setattr(ev, "compute_coordination", flaky)
    with pytest.warns(UserWarning, match="failed on 1/2 runs"):
        res = analyze_runs(load_runset(FIXTURE))
    doc = render_html(res)
    assert "coordination analysis failed on 1/2 runs: KeyError: " in visible_text(doc)
    assert "was skipped" not in doc and HOSTILE not in doc
    # no summary but recorded errors (every run failing raises in analyze_runs; a
    # result built by hand can still look like this): the note, not "skipped"
    res.coordination = None
    doc = render_html(res)
    assert "coordination analysis failed on 1/2 runs: KeyError: " in visible_text(doc)
    assert "was skipped" not in doc and HOSTILE not in doc


def test_every_run_failing_coordination_raises(monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]

    def broken(traj, **kw):
        raise KeyError(HOSTILE)

    monkeypatch.setattr(ev, "compute_coordination", broken)
    with pytest.raises(KeyError):
        analyze_runs(load_runset(FIXTURE))
