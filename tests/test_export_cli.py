import json
import sys

import pytest

from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval import (Benchmark, EvaluationResult, FunctionSwarm, SimClock, Task, evaluate, read_events_jsonl,
                       render_trajectory, write_events_jsonl)
from swarmeval.cli import main
from swarmeval.runner import RunSet, derive_seed


def test_evaluate_runs_benchmark_and_is_reproducible():
    a = evaluate(make_swarm, build_benchmark(3), trials=2, clock_factory=SimClock)
    b = evaluate(make_swarm, build_benchmark(3), trials=2, clock_factory=SimClock)
    assert len(a.runs) == 6 and a.performance.n_tasks == 3 and a.performance.trials == 2
    assert [t.succeeded for t in a.runs] == [t.succeeded for t in b.runs]
    assert a.efficiency.tokens.total == b.efficiency.tokens.total
    assert derive_seed(0, "task_01", 0) != derive_seed(0, "task_01", 1)
    assert a.agents == ["coordinator", "researcher_a", "researcher_b", "researcher_c", "critic", "synthesizer"]
    assert a.efficiency.cost_complete and a.cost["total"] > 0


def test_runset_and_result_roundtrip(tmp_path):
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    p = tmp_path / "r.json"
    res.save(str(p))
    back = EvaluationResult.load(str(p))
    assert back.success_rate == res.success_rate
    assert back.summary() == res.summary()
    rs = RunSet.from_dict(json.load(open(p))["runs"])
    assert len(rs) == 2 and rs.label == "baseline" and rs.benchmark_name == "research_briefs"
    d = res.to_dict()
    assert "runs" not in d and len(d["per_run"]) == 2


def test_jsonl_export(tmp_path):
    res = evaluate(make_swarm, build_benchmark(1), trials=1, clock_factory=SimClock)
    traj = res.runs.trajectories[0]
    p = tmp_path / "e.jsonl"
    n = write_events_jsonl(traj, str(p))
    events = read_events_jsonl(str(p))
    assert n == len(traj.events) == len(events)
    assert events[0].event_type == "run_start" and events[0].run_id == traj.run_id
    with open(p) as f:
        first = json.loads(f.readline())
    assert set(first) >= {"event_id", "run_id", "task_id", "event_type", "timestamp"}


def test_trajectory_text_rendering():
    res = evaluate(make_swarm, build_benchmark(1), trials=1, clock_factory=SimClock)
    text = render_trajectory(res.runs.trajectories[0])
    assert "llm_call" in text and "web_search" in text and "coordinator" in text and "critic" in text


def test_function_swarm_and_crash_is_a_result():
    def good(task, ctx):
        with ctx.agent("solo") as s:
            s.llm(lambda: "the answer is 42", prompt=task.input)
        return "42"

    def bad(task, ctx):
        with ctx.agent("solo"):
            raise RuntimeError("kaboom")

    bench = Benchmark("b", [Task("q", "what?", evaluator={"type": "exact_match", "expected": "42"})])
    ok = evaluate(good, bench, clock_factory=SimClock)
    assert ok.success_rate == 1.0 and isinstance(ok.runs.trajectories[0].metadata["swarm"], str)
    crashed = evaluate(FunctionSwarm(bad), bench, clock_factory=SimClock)
    t = crashed.runs.trajectories[0]
    assert crashed.success_rate == 0.0 and t.status == "error" and "kaboom" in t.error
    assert crashed.performance.run_errors == 1


def test_benchmark_file_roundtrip_and_validation(tmp_path):
    bench = build_benchmark(2)
    p = tmp_path / "bench.json"
    bench.save(str(p))
    back = Benchmark.load(str(p))
    assert [t.task_id for t in back] == [t.task_id for t in bench]
    assert back.get("task_01").evaluator["type"] == "contains_all"
    import pytest
    with pytest.raises(ValueError):
        Benchmark("dup", [Task("x", "a"), Task("x", "b")]).validate()
    with pytest.raises(ValueError):
        Task("x", "a", difficulty="impossible").validate()


def test_cli_demo_run_report(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--trials", "1", "--out", str(out), "--trace"]) == 0
    assert (out / "baseline.json").exists() and (out / "baseline.events.jsonl").exists()
    bench_path = tmp_path / "bench.json"
    build_benchmark(2).save(str(bench_path))
    assert main(["run", "--swarm", "swarmeval.examples.research_swarm:make_swarm", "--benchmark", str(bench_path),
                 "--trials", "1", "--clock", "sim", "--label", "cli", "--out", str(tmp_path / "run")]) == 0
    assert main(["report", str(tmp_path / "run" / "cli.json"), "--trace", "0"]) == 0
    captured = capsys.readouterr()
    assert "SWARM EVALUATION" in captured.out and "cli  ·  research_briefs" in captured.out
    assert "run_start" in captured.out


def test_cli_writes_the_plain_evaluation_when_coordination_analysis_fails(tmp_path, capsys, monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]

    def missing_dep(traj, **kw):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(ev, "compute_coordination", missing_dep)
    assert main(["demo", "--trials", "1", "--out", str(tmp_path)]) == 1
    assert (tmp_path / "baseline.json").exists() and (tmp_path / "baseline.events.jsonl").exists()
    err = capsys.readouterr().err
    assert ("swarmeval: coordination analysis failed: ImportError: sentence-transformers is not installed; "
            f"plain evaluation written to {tmp_path}") in err


def test_cli_says_the_plain_evaluation_is_only_shown_without_out(capsys, monkeypatch):
    ev = sys.modules["swarmeval.evaluate"]

    def missing_dep(traj, **kw):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(ev, "compute_coordination", missing_dep)
    assert main(["demo", "--trials", "1"]) == 1
    err = capsys.readouterr().err
    assert "; plain evaluation shown" in err and "written" not in err


def test_report_shows_the_plain_evaluation_when_coordination_analysis_fails(tmp_path, capsys, monkeypatch):
    from swarmeval import SimClock, evaluate
    from swarmeval.examples.research_swarm import build_benchmark, make_swarm
    saved = tmp_path / "runs.json"
    evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock).runs.save(str(saved))

    def odd(traj, **kw):
        raise KeyError("span_x")

    monkeypatch.setattr(sys.modules["swarmeval.evaluate"], "compute_coordination", odd)
    assert main(["report", str(saved)]) == 1
    out, err = capsys.readouterr()
    assert "SWARM EVALUATION" in out
    assert ("swarmeval: coordination analysis failed: KeyError: 'span_x'; showing the plain evaluation "
            "(rerun with --no-coordination to skip it)") in err


def test_report_names_the_failure_when_every_run_fails_coordination_analysis(tmp_path, capsys, monkeypatch):
    saved = tmp_path / "runs.json"
    evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock).runs.save(str(saved))

    def odd(traj, **kw):
        raise KeyError("span_x")

    monkeypatch.setattr(sys.modules["swarmeval.evaluate"], "compute_coordination", odd)
    html = tmp_path / "r.html"
    assert main(["report", str(saved), "--html", str(html)]) == 1
    out = capsys.readouterr().out
    assert "Analysis Failed" in out and "2/2 runs: KeyError" in out
    doc = html.read_text()
    assert "coordination analysis failed on 2/2 runs: KeyError" in doc and "was skipped" not in doc


def test_report_html_keeps_a_partial_coordination_failure(tmp_path, capsys, monkeypatch):
    saved = tmp_path / "runs.json"
    evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock).runs.save(str(saved))
    ev = sys.modules["swarmeval.evaluate"]
    real, calls = ev.compute_coordination, []

    def flaky(traj, **kw):
        calls.append(traj.run_id)
        if len(calls) == 2:
            raise KeyError("span_x")
        return real(traj, **kw)

    monkeypatch.setattr(ev, "compute_coordination", flaky)
    html = tmp_path / "r.html"
    with pytest.warns(UserWarning, match="failed on 1/2 runs"):
        assert main(["report", str(saved), "--html", str(html)]) == 0
    assert "1/2 runs: KeyError" in capsys.readouterr().out
    doc = html.read_text()
    assert "coordination analysis failed on 1/2 runs: KeyError" in doc and "was skipped" not in doc


def test_demo_html_names_the_failure_when_every_run_fails_coordination_analysis(tmp_path, capsys, monkeypatch):
    def odd(traj, **kw):
        raise KeyError("span_x")

    monkeypatch.setattr(sys.modules["swarmeval.evaluate"], "compute_coordination", odd)
    html = tmp_path / "d.html"
    assert main(["demo", "--trials", "1", "--html", str(html), "--out", str(tmp_path / "D")]) == 1
    n = len(build_benchmark())
    doc = html.read_text()
    assert f"coordination analysis failed on {n}/{n} runs: KeyError" in doc and "was skipped" not in doc


def test_compare_shows_the_plain_comparison_when_coordination_analysis_fails(tmp_path, capsys, monkeypatch):
    from swarmeval import SimClock, evaluate
    from swarmeval.examples.research_swarm import build_benchmark, make_swarm
    saved = tmp_path / "runs.json"
    evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock).runs.save(str(saved))

    def odd(traj, **kw):
        raise KeyError("span_x")

    monkeypatch.setattr(sys.modules["swarmeval.evaluate"], "compute_coordination", odd)
    assert main(["compare", str(saved), str(saved)]) == 1
    out, err = capsys.readouterr()
    assert out.strip()
    assert ("swarmeval: coordination analysis failed: KeyError: 'span_x'; showing the plain evaluation "
            "(rerun with --no-coordination to skip it)") in err
