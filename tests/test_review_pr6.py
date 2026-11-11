"""Regression test for rohitp30's review nit on PR #6: an edge that skips a
column must not be drawn through the box of an agent it does not touch."""
import html
import re

from swarmeval import SimClock, build_graph, evaluate
from swarmeval.examples.research_swarm import build_benchmark, make_swarm
from swarmeval.report.html import _NODE_H, _NODE_W, _first_start, _layout, graph_svg

EDGE = re.compile(r'd="M([\d.-]+),([\d.-]+) Q([\d.-]+),([\d.-]+) ([\d.-]+),([\d.-]+)"[^>]*><title>([^<]+?) → ([^:<]+):')


def test_edges_do_not_pass_through_other_agents():
    traj = evaluate(make_swarm, build_benchmark(1), clock_factory=SimClock).runs.trajectories[0]
    g = build_graph(traj)
    pos = _layout(g, _first_start(traj, list(g.agents)))
    edges = list(EDGE.finditer(graph_svg(traj, g)))
    assert edges
    crossings = set()
    for m in edges:
        x1, y1, qx, qy, x2, y2 = map(float, m.groups()[:6])
        src, dst = html.unescape(m.group(7)), html.unescape(m.group(8))
        for t in (i / 50 for i in range(1, 50)):
            px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * qx + t * t * x2
            py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * qy + t * t * y2
            for a, (nx, ny) in pos.items():
                if a not in (src, dst) and nx < px < nx + _NODE_W and ny < py < ny + _NODE_H:
                    crossings.add(f"{src}→{dst} crosses {a}")
    assert not crossings, sorted(crossings)
