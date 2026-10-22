"""Layer 3 -- dependencies, critical path and parallelism of one run (Phase 2).

Built on the work-unit DAG of ``graph.builder``.  Definitions:

* **critical path** = the heaviest path through the DAG, each unit weighted by
  the time it spends working (LLM/tool duration; sends, receives, artifacts
  and waits weigh 0).  No schedule of the recorded work, however parallel,
  could finish faster than this -- *if* the DAG holds every real dependency.
* **work** (``work_s``) = Σ durations of top-level LLM/tool calls (calls nested
  in another call are already inside their parent's time).
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
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..graph.builder import EPS, ExecutionGraph, build_graph
from ..schema import MessageKind, Trajectory
from ..stats import Summary, summarize
from .efficiency import EfficiencyMetrics, union_length

_CALL_KINDS = ("llm", "tool")
# Kinds that ask for something; a result/feedback going back the other way answers them.
REQUEST_KINDS = frozenset({MessageKind.TASK_ASSIGNMENT.value, MessageKind.REQUEST.value,
                           MessageKind.COORDINATION.value})
ANSWER_KINDS = frozenset({MessageKind.RESULT.value, MessageKind.FEEDBACK.value})


@dataclass
class CriticalPath:
    """The heaviest DAG path.  ``node_ids`` in path order; ``per_agent_s``
    is each agent's working time on it and ``wait_s`` the duration of waits
    that lie on it (they weigh 0 in the length)."""
    length_s: float = 0.0
    node_ids: List[str] = field(default_factory=list)
    per_agent_s: Dict[str, float] = field(default_factory=dict)
    wait_s: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyMetrics:
    """Dependency structure of one run (see the module docstring).  Ratios
    are ``None`` when undefined (no wall clock, empty critical path)."""
    critical_path: CriticalPath = field(default_factory=CriticalPath)
    work_s: float = 0.0
    wall_s: float = 0.0
    actual_parallelism: Optional[float] = None
    potential_parallelism: Optional[float] = None
    headroom_s: Optional[float] = None
    headroom_ratio: Optional[float] = None
    unrecorded_s: Optional[float] = None
    depth: int = 0
    inferred_edges_on_path: int = 0
    cycles: List[List[str]] = field(default_factory=list)
    isolated_agents: List[str] = field(default_factory=list)

    @property
    def critical_path_s(self) -> float:
        return self.critical_path.length_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["critical_path_s"] = self.critical_path_s
        return d


# --------------------------------------------------------------------------- #
# Critical path
# --------------------------------------------------------------------------- #
def critical_path(graph: ExecutionGraph) -> Tuple[CriticalPath, int]:
    """Heaviest path by topological DP (Kahn's algorithm, ready nodes taken
    in start order so ties resolve the same way every time: the earliest
    predecessor wins, and the earliest-finishing end node).  Returns the path
    and the number of inferred edges on it."""
    units = graph.dag_units
    order = {u.node_id: i for i, u in enumerate(units)}
    indeg: Dict[str, int] = {u.node_id: 0 for u in units}
    preds: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    succs: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges:
        if e.src in indeg and e.dst in indeg:
            indeg[e.dst] += 1
            preds[e.dst].append((e.src, e.inferred))
            succs[e.src].append(e.dst)
    ready = [order[n] for n, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    dist: Dict[str, float] = {}
    best: Dict[str, Optional[Tuple[str, bool]]] = {}
    while ready:
        n = units[heapq.heappop(ready)].node_id
        cand = sorted(preds.get(n, []), key=lambda p: order[p[0]])
        top: Optional[Tuple[str, bool]] = None
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
    for u in units:   # start order; strict > keeps the earliest on ties
        if u.node_id in dist and (end is None or dist[u.node_id] > dist[end] + EPS):
            end = u.node_id
    path: List[str] = []
    inferred = 0
    cur: Optional[str] = end
    while cur is not None:
        path.append(cur)
        step = best.get(cur)
        if step is not None and step[1]:
            inferred += 1
        cur = step[0] if step else None
    path.reverse()
    # zero-weight nodes (sends, receives, waits) stay on the path: they cost
    # nothing but show how the work is chained across agents
    per_agent: Dict[str, float] = defaultdict(float)
    waits: Dict[str, float] = defaultdict(float)
    for n in path:
        u = graph.unit(n)
        key = u.agent_id or "-"
        if u.kind in _CALL_KINDS:
            per_agent[key] += u.weight
        elif u.kind == "wait":
            waits[key] += u.end - u.start
    return CriticalPath(dist[end], path, dict(per_agent), dict(waits)), inferred


# --------------------------------------------------------------------------- #
# Agent-level structure
# --------------------------------------------------------------------------- #
def reply_flags(traj: Trajectory) -> List[Tuple[Any, bool]]:
    """Messages in start order, each with whether it is a reply: it names
    the message it answers (``reply_to``), or it is a result/feedback to an
    agent that earlier sent its sender a request or task assignment."""
    msgs = sorted(traj.messages(), key=lambda e: e.timestamp)
    asked: Set[Tuple[str, str]] = set()      # (asker, asked)
    out: List[Tuple[Any, bool]] = []
    for ev in msgs:
        is_reply = bool(ev.reply_to) or (ev.kind in ANSWER_KINDS and (ev.receiver, ev.sender) in asked)
        if ev.kind in REQUEST_KINDS and ev.sender and ev.receiver:
            asked.add((ev.sender, ev.receiver))
        out.append((ev, is_reply))
    return out


def tarjan_scc(nodes: Sequence[str], adj: Dict[str, List[str]]) -> List[List[str]]:
    """Strongly connected components (iterative Tarjan), each in ``nodes`` order."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on: Set[str] = set()
    stack: List[str] = []
    out: List[List[str]] = []
    counter = 0
    rank = {n: i for i, n in enumerate(nodes)}
    for root in nodes:
        if root in index:
            continue
        work: List[Tuple[str, int]] = [(root, 0)]
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


def message_cycles(traj: Trajectory, agents: Sequence[str]) -> List[List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for ev, is_reply in reply_flags(traj):
        if is_reply or not ev.sender or not ev.receiver or ev.sender == ev.receiver:
            continue
        if ev.receiver not in adj[ev.sender]:
            adj[ev.sender].append(ev.receiver)
    nodes = list(agents) + sorted({n for n in adj if n not in agents})
    return [c for c in tarjan_scc(nodes, adj) if len(c) >= 3]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def compute_dependency(traj: Trajectory, graph: Optional[ExecutionGraph] = None,
                       efficiency: Optional[EfficiencyMetrics] = None) -> DependencyMetrics:
    graph = graph or build_graph(traj)
    cp, inferred = critical_path(graph)
    out = DependencyMetrics(critical_path=cp, inferred_edges_on_path=inferred)
    out.work_s = sum(u.weight for u in graph.dag_units if u.kind in _CALL_KINDS)
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
    """Dependency metrics over runs; ratio summaries cover the runs where
    the ratio is defined (``n``)."""
    n_runs: int
    critical_path_s: Summary
    work_s: Summary
    actual_parallelism: Optional[Summary]
    potential_parallelism: Optional[Summary]
    headroom_s: Optional[Summary]
    headroom_ratio: Optional[Summary]
    unrecorded_s: Optional[Summary]
    depth: Summary
    runs_with_cycles: int = 0
    isolated_agents: Dict[str, int] = field(default_factory=dict)
    critical_path_share: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}


def _known(values: Sequence[Optional[float]]) -> Optional[Summary]:
    known = [v for v in values if v is not None]
    return summarize(known) if known else None


def summarize_dependency(per_run: Sequence[DependencyMetrics]) -> DependencySummary:
    """``critical_path_share[a]`` = mean over runs of agent ``a``'s working
    time on the critical path ÷ its length (runs with an empty path skipped)."""
    iso: Dict[str, int] = defaultdict(int)
    share: Dict[str, List[float]] = defaultdict(list)
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
        actual_parallelism=_known([r.actual_parallelism for r in per_run]),
        potential_parallelism=_known([r.potential_parallelism for r in per_run]),
        headroom_s=_known([r.headroom_s for r in per_run]),
        headroom_ratio=_known([r.headroom_ratio for r in per_run]),
        unrecorded_s=_known([r.unrecorded_s for r in per_run]),
        depth=summarize([r.depth for r in per_run]),
        runs_with_cycles=sum(1 for r in per_run if r.cycles), isolated_agents=dict(iso),
        critical_path_share={a: sum(v) / n_cp for a, v in share.items()} if n_cp else {})
