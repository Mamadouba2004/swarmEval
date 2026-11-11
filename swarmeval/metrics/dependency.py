"""Layer 3 -- dependencies, critical path and parallelism of one run (Phase 2).

Built on the work-unit DAG of ``graph.builder``.  Definitions:

* **critical path** = the heaviest path through the DAG, each unit weighted by
  the time it spends working (LLM/tool duration; sends, receives, artifacts
  and waits weigh 0).  No schedule of the recorded work, however parallel,
  could finish faster than this -- *if* the DAG holds every real dependency.
* **work** (``work_s``) = Σ durations of top-level LLM/tool calls (calls nested
  in another call are already inside their parent's time, and so are the calls
  of an agent run inside another agent's call, e.g. an agent used as a tool).
* **Actual parallelism** = work ÷ wall clock: how many calls ran at once on
  average.  **Potential parallelism** = work ÷ critical path: the most the
  recorded dependencies allow.
* **Parallel headroom** = wall − critical path (seconds) and wall ÷ critical
  path (ratio).  Part of that gap can be time in which nothing was recorded
  at all (``unrecorded_s``: wall minus the union of all call and wait
  intervals) -- idle time or untracked work, not work that could have run in
  parallel, so it is reported separately rather than read as headroom.

The DAG is only as complete as the record: on Phase 1 data message edges are
inferred from timing (``inferred_edges_on_path`` counts them on the critical
path), and missing edges make the critical path too short and the headroom
too large.

``cycles`` are strongly connected groups of three or more agents in the
message graph once replies are set aside -- a request answered by a result is
a round trip, never a cycle.  ``isolated_agents`` exchanged nothing with any
other agent (no message, artifact or invocation edge either way).

Classes:
- CriticalPath: The heaviest DAG path. ``node_ids`` in path order; ``per_agent_s`` is each agent's working time on
  it and ``wait_s`` the duration of waits that lie on it (they weigh 0 in the length).
- DependencyMetrics: Dependency structure of one run (see the module docstring). Ratios are ``None`` when undefined
  (no wall clock, empty critical path).
- DependencySummary: Dependency metrics over runs; ratio summaries cover the runs where the ratio is defined
  (``n``).
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph.builder import EPS, ExecutionGraph, build_graph
from ..schema import MessageKind, Trajectory
from ..stats import Summary, known_summary, ranged, summarize, summary_dict
from .efficiency import EfficiencyMetrics, union_length

_CALL_KINDS = ("llm", "tool")
REQUEST_KINDS = frozenset({MessageKind.TASK_ASSIGNMENT.value, MessageKind.REQUEST.value,
                           MessageKind.COORDINATION.value})
DEPENDENCY_METRICS = (
    ("critical_path_s", "Critical Path / Run", False, "s", "critical_path_s"),
    ("actual_parallelism", "Actual Parallelism", True, "num", "actual_parallelism"),
    ("potential_parallelism", "Potential Parallelism", None, "num", "potential_parallelism"),
)
ANSWER_KINDS = frozenset({MessageKind.RESULT.value, MessageKind.FEEDBACK.value})


@dataclass
class CriticalPath:
    length_s: float = 0.0
    node_ids: list[str] = field(default_factory=list)
    per_agent_s: dict[str, float] = field(default_factory=dict)
    wait_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The path as a JSON-ready dict."""
        return asdict(self)


@dataclass
class DependencyMetrics:
    critical_path: CriticalPath = field(default_factory=CriticalPath)
    work_s: float = 0.0
    wall_s: float = 0.0
    actual_parallelism: float | None = None
    potential_parallelism: float | None = None
    headroom_s: float | None = None
    headroom_ratio: float | None = None
    unrecorded_s: float | None = None
    depth: int = 0
    inferred_edges_on_path: int = 0
    cycles: list[list[str]] = field(default_factory=list)
    isolated_agents: list[str] = field(default_factory=list)

    @property
    def critical_path_s(self) -> float:
        """Length of the critical path in seconds."""
        return self.critical_path.length_s

    def to_dict(self) -> dict[str, Any]:
        """The metrics as a JSON-ready dict."""
        d = asdict(self)
        d["critical_path_s"] = self.critical_path_s
        return d


def critical_path(graph: ExecutionGraph) -> tuple[CriticalPath, int]:
    """Heaviest DAG path by Kahn-order DP, ties going to the earliest node so results are deterministic.
    Returns the path and the number of inferred edges on it."""
    units = graph.dag_units
    order = {u.node_id: i for i, u in enumerate(units)}
    indeg: dict[str, int] = {u.node_id: 0 for u in units}
    preds: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    succs: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        if e.src in indeg and e.dst in indeg:
            indeg[e.dst] += 1
            preds[e.dst].append((e.src, e.inferred))
            succs[e.src].append(e.dst)
    ready = [order[n] for n, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    dist: dict[str, float] = {}
    best: dict[str, tuple[str, bool] | None] = {}
    while ready:
        n = units[heapq.heappop(ready)].node_id
        cand = sorted(preds.get(n, []), key=lambda p: order[p[0]])
        top: tuple[str, bool] | None = None
        for p in cand:
            if p[0] in dist and (top is None or dist[p[0]] > dist[top[0]] + EPS):
                top = p
        dist[n] = graph.unit(n).weight + (dist[top[0]] if top else 0.0)
        best[n] = top
        for s in succs.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready, order[s])
    if not dist:
        return CriticalPath(), 0
    end = None
    for u in units:
        if u.node_id in dist and (end is None or dist[u.node_id] > dist[end] + EPS):
            end = u.node_id
    path: list[str] = []
    inferred = 0
    cur: str | None = end
    while cur is not None:
        path.append(cur)
        step = best.get(cur)
        if step is not None and step[1]:
            inferred += 1
        cur = step[0] if step else None
    path.reverse()
    per_agent: dict[str, float] = defaultdict(float)
    waits: dict[str, float] = defaultdict(float)
    for n in path:
        u = graph.unit(n)
        key = u.agent_id or "-"
        if u.kind in _CALL_KINDS:
            per_agent[key] += u.weight
        elif u.kind == "wait":
            waits[key] += u.end - u.start
    return CriticalPath(dist[end], path, dict(per_agent), dict(waits)), inferred


def reply_flags(traj: Trajectory) -> list[tuple[Any, bool]]:
    """Messages in start order, each flagged as a reply when it names ``reply_to`` or is a result/feedback
    to an agent that earlier sent its sender a request or task assignment."""
    msgs = sorted(traj.messages(), key=lambda e: e.timestamp)
    asked: set[tuple[str, str]] = set()
    out: list[tuple[Any, bool]] = []
    for ev in msgs:
        is_reply = bool(ev.reply_to) or (ev.kind in ANSWER_KINDS and (ev.receiver, ev.sender) in asked)
        if ev.kind in REQUEST_KINDS and ev.sender and ev.receiver:
            asked.add((ev.sender, ev.receiver))
        out.append((ev, is_reply))
    return out


def tarjan_scc(nodes: Sequence[str], adj: dict[str, list[str]]) -> list[list[str]]:
    """Strongly connected components (iterative Tarjan), each in ``nodes`` order."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    rank = {n: i for i, n in enumerate(nodes)}
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on.add(v)
            nbrs = adj.get(v, [])
            if i < len(nbrs):
                work.append((v, i + 1))
                w = nbrs[i]
                if w not in index:
                    work.append((w, 0))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp, key=lambda n: rank.get(n, len(rank))))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return out


def message_cycles(traj: Trajectory, agents: Sequence[str]) -> list[list[str]]:
    """Cycles in the agent graph of non-reply messages."""
    adj: dict[str, list[str]] = defaultdict(list)
    for ev, is_reply in reply_flags(traj):
        if is_reply or not ev.sender or not ev.receiver or ev.sender == ev.receiver:
            continue
        if ev.receiver not in adj[ev.sender]:
            adj[ev.sender].append(ev.receiver)
    nodes = list(agents) + sorted({n for n in adj if n not in agents})
    return [c for c in tarjan_scc(nodes, adj) if len(c) >= 3]


def _inside_calls(traj: Trajectory, graph: ExecutionGraph) -> set[str]:
    """Spans run inside another span's LLM/tool call, directly or via an ancestor: recorded since schema v2,
    inferred on Phase 1 data from a span lying inside a call of one of its ancestors."""
    spans = traj.spans
    calls: dict[str, list[Any]] = defaultdict(list)
    for u in graph.units:
        if u.kind in _CALL_KINDS:
            calls[u.span_id].append(u)
    out: set[str] = set()
    for sid, s in spans.items():
        end = s.end if s.end is not None else s.start
        cur: str | None = sid
        seen: set[str] = set()
        while cur in spans and cur not in seen:
            seen.add(cur)
            if graph.schema_version is not None:
                hit = spans[cur].metadata.get("within_call") is not None
            else:
                hit = cur != sid and any(u.start <= s.start + EPS and end <= u.end + EPS for u in calls[cur])
            if hit:
                out.add(sid)
                break
            cur = spans[cur].parent_span_id
    return out


def compute_dependency(traj: Trajectory, graph: ExecutionGraph | None = None,
                       efficiency: EfficiencyMetrics | None = None) -> DependencyMetrics:
    """Critical path, parallelism and agent-level dependency structure of one run."""
    graph = graph or build_graph(traj)
    cp, inferred = critical_path(graph)
    out = DependencyMetrics(critical_path=cp, inferred_edges_on_path=inferred)
    inside = _inside_calls(traj, graph)
    out.work_s = sum(u.weight for u in graph.dag_units if u.kind in _CALL_KINDS and u.span_id not in inside)
    out.wall_s = efficiency.wall_clock_s if efficiency is not None else traj.duration
    out.depth = sum(1 for n in cp.node_ids if graph.unit(n).kind in _CALL_KINDS)
    if out.wall_s > EPS:
        out.actual_parallelism = out.work_s / out.wall_s
        intervals = [(u.start, u.end) for u in graph.units if u.kind in _CALL_KINDS + ("wait",)]
        out.unrecorded_s = max(0.0, out.wall_s - union_length(intervals))
        out.headroom_s = max(0.0, out.wall_s - cp.length_s)
        if cp.length_s > EPS:
            out.headroom_ratio = out.wall_s / cp.length_s
    if cp.length_s > EPS:
        out.potential_parallelism = out.work_s / cp.length_s
    out.cycles = message_cycles(traj, graph.agents)
    connected = {a for e in graph.agent_edges if e.source != e.target for a in (e.source, e.target)}
    out.isolated_agents = [a for a in graph.agents if a not in connected] if len(graph.agents) > 1 else []
    return out


@dataclass
class DependencySummary:
    n_runs: int
    critical_path_s: Summary
    work_s: Summary
    actual_parallelism: Summary | None
    potential_parallelism: Summary | None
    headroom_s: Summary | None
    headroom_ratio: Summary | None
    unrecorded_s: Summary | None
    depth: Summary
    runs_with_cycles: int = 0
    isolated_agents: dict[str, int] = field(default_factory=dict)
    critical_path_share: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dict."""
        return summary_dict(self)



def summarize_dependency(per_run: Sequence[DependencyMetrics]) -> DependencySummary:
    """``critical_path_share[a]`` = mean over runs of agent ``a``'s working
    time on the critical path ÷ its length (runs with an empty path skipped)."""
    iso: dict[str, int] = defaultdict(int)
    share: dict[str, list[float]] = defaultdict(list)
    for r in per_run:
        for a in r.isolated_agents:
            iso[a] += 1
        if r.critical_path_s > EPS:
            for a, s in r.critical_path.per_agent_s.items():
                share[a].append(s / r.critical_path_s)
    n_cp = sum(1 for r in per_run if r.critical_path_s > EPS)
    return DependencySummary(
        n_runs=len(per_run), critical_path_s=summarize([r.critical_path_s for r in per_run]),
        work_s=summarize([r.work_s for r in per_run]),
        actual_parallelism=known_summary([r.actual_parallelism for r in per_run]),
        potential_parallelism=known_summary([r.potential_parallelism for r in per_run]),
        headroom_s=known_summary([r.headroom_s for r in per_run]),
        headroom_ratio=known_summary([r.headroom_ratio for r in per_run]),
        unrecorded_s=known_summary([r.unrecorded_s for r in per_run]),
        depth=summarize([r.depth for r in per_run]),
        runs_with_cycles=sum(1 for r in per_run if r.cycles), isolated_agents=dict(iso),
        critical_path_share={a: sum(v) / n_cp for a, v in share.items()} if n_cp else {})


def dependency_rows(dep: DependencySummary) -> list[tuple[str, str] | None]:
    """The dependency rows of the COORDINATION box, ending with a group separator."""
    sec = lambda x: f"{x:.1f}s"         # noqa: E731
    times = lambda x: f"{x:.2f}×"       # noqa: E731
    headroom = ranged(dep.headroom_s, sec)
    if dep.headroom_ratio is not None:
        headroom += f"  {dep.headroom_ratio.mean:.2f}×"
    return [
        ("Critical Path / Run", ranged(dep.critical_path_s, sec)),
        ("Actual Parallelism (work ÷ wall)", ranged(dep.actual_parallelism, times)),
        ("Potential Parallelism (work ÷ critical path)", ranged(dep.potential_parallelism, times)),
        ("Parallel Headroom (wall − critical path)", headroom),
        ("Unrecorded Time (nothing running)", ranged(dep.unrecorded_s, sec)),
        None,
    ]
