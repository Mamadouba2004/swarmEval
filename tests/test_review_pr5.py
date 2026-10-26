"""Regression tests for rohitp30's review nits on PR #5."""
import json

import pytest

from swarmeval import LexicalSimilarity, RunContext, RunSet, SimClock, Task, analyze_runs, cli, evaluate, \
    render_coordination
from swarmeval.examples.research_swarm import build_benchmark, make_swarm

HUB = "research_coordinator_agent_07"   # long enough that "100% (<id>)" passes the old 30-char value slice


def hub_run():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    with ctx.agent(HUB) as h:
        for p in ("p1", "p2", "p3"):
            h.send(p, "do x")
    for p in ("p1", "p2", "p3"):
        with ctx.agent(p) as s:
            s.receive()
            s.send(HUB, "done")
    return analyze_runs(RunSet("l", "b", [ctx.finish()]))


# -- long agent ids are not cut off -------------------------------------------- #
def test_hub_share_row_keeps_long_agent_id():
    (line,) = [l for l in render_coordination(hub_run()).splitlines() if "Hub Share" in l]
    assert HUB in line


def test_contribution_row_keeps_long_agent_id():
    text = render_coordination(hub_run())
    section = text.split("OBSERVED CONTRIBUTION", 1)[1]
    assert any(l.startswith("  " + HUB + " ") for l in section.splitlines())


# -- `swarmeval graph` honours the same analysis options as `report` ------------ #
@pytest.fixture
def saved_demo(tmp_path):
    path = str(tmp_path / "r.json")
    evaluate(make_swarm, build_benchmark(1), clock_factory=SimClock).save(path)
    return path


@pytest.fixture
def coordination_calls(monkeypatch):
    calls = []
    real = cli.compute_coordination

    def spy(traj, **kw):
        calls.append(kw)
        return real(traj, **kw)

    monkeypatch.setattr(cli, "compute_coordination", spy)
    return calls


def test_graph_uses_report_analysis_options(saved_demo, coordination_calls, tmp_path):
    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"hub_share": 0.9}))
    cli.main(["graph", saved_demo, "--similarity", "lexical", "--smells-config", str(cfg)])
    (kw,) = coordination_calls
    assert isinstance(kw["similarity"], LexicalSimilarity) and kw["smell_config"].hub_share == 0.9


def test_graph_no_coordination_skips_the_analysis(saved_demo, coordination_calls):
    cli.main(["graph", saved_demo, "--no-coordination"])
    assert coordination_calls == []
