"""Layer 3 -- execution graph of one trajectory (Phase 2: coordination analysis).

Two views, built only from what was recorded:

* the **agent graph**: who messaged whom, whose artifacts were consumed by
  whom, and who invoked whom;
* the **work-unit DAG**: LLM and tool calls plus point events (send, receive,
  produce, use, wait) with typed dependency edges, the input to critical path
  and parallelism analysis.

Where the record is silent (Phase 1 data has no receive events, message ids or
``derived_from``) an edge is either inferred from timing and flagged
``inferred=True``, or missing.  ``Coverage`` says how much was recorded, so a
reader knows how far the DAG can be trusted: missing edges under-count
dependencies and over-state possible parallelism.

Events are appended when they complete but timestamped when they start, so
everything here works on start-ordered units, never on event order.  Ties in
start time are broken by agent id and then by recording order, which for
point events is the order in which they happened.
"""
from __future__ import annotations

import heapq
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..schema import Event, EventType, Trajectory

EPS = 1e-9   # timestamps closer than this count as simultaneous

_UNIT_KINDS = {
    EventType.LLM_CALL.value: "llm",
    EventType.TOOL_CALL.value: "tool",
    EventType.MESSAGE.value: "send",
    EventType.MESSAGE_RECEIVE.value: "receive",
    EventType.ARTIFACT.value: "produce",
    EventType.ARTIFACT_USE.value: "use",
    EventType.WAIT.value: "wait",
}
_CALL_KINDS = ("llm", "tool")
_COMM_KINDS = ("message", "artifact")


# --------------------------------------------------------------------------- #
# Graph elements
# --------------------------------------------------------------------------- #
@dataclass
class WorkUnit:
    """One node of the DAG.  ``weight`` is the time the unit itself spends
    working (LLM/tool duration; 0 for point events and waits).  A call made
    inside another call of the same span has ``nested_in`` set and takes no
    part in the DAG: its time is already inside its parent's."""
    node_id: str
    kind: str                       # llm | tool | send | receive | produce | use | wait
    event_id: str
    agent_id: Optional[str]
    span_id: Optional[str]
    start: float
    end: float
    weight: float = 0.0
    nested_in: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DagEdge:
    """``dst`` could not start before ``src`` finished.  ``ref`` names what
    links them: a message id, an artifact id, a child span id or an agent id."""
    src: str
    dst: str
    type: str                       # sequence | invocation | message | artifact | wait
    inferred: bool = False
    ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentEdge:
    """Aggregated interaction between two agents.  For messages ``count`` is
    the number of messages, ``tokens`` their recorded tokens, ``used_count``
    how many were used by the receiver and ``inferred_count`` how many had
    their delivery or use inferred rather than recorded.  For artifacts
    ``count`` is the number of distinct artifacts the target consumed."""
    source: str
    target: str
    kind: str                       # message | artifact | invocation
    count: int = 0
    tokens: int = 0
    inferred_count: int = 0
    used_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MessageLink:
    """What became of one message.  ``receive_node`` is the recorded receive
    or, for Phase 1 data without one, the receiver's first unit after the send
    (``received`` False); ``None`` means it never reached the receiver.  A
    message is ``used`` when the receiver consumed one of its artifacts or,
    failing that, did LLM/tool work after receiving it -- a timing heuristic,
    flagged ``used_inferred``.  For a message carrying artifacts only calls
    that record no ``input_artifacts`` count: one listing other inputs shows
    the artifacts were not used."""
    send_node: str
    receive_node: Optional[str]
    message_id: Optional[str]
    sender: Optional[str]
    receiver: Optional[str]
    kind: Optional[str]
    tokens: int
    artifacts: List[str]
    received: bool
    used: bool
    used_inferred: bool

    @property
    def inferred(self) -> bool:
        return not self.received or self.used_inferred

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inferred"] = self.inferred
        return d


@dataclass
class Coverage:
    """How much provenance was recorded.  Ratios are ``None`` when there is
    nothing to measure (no messages / no artifacts), never a silent 0 or 1.
    ``consumers_instrumented[a]`` is True when agent ``a`` records what it
    uses (an ARTIFACT_USE, an LLM/tool call with ``input_artifacts``, or an
    artifact with a non-empty ``derived_from``); without that, "a never used X" cannot be
    told apart from "a's use of X was not recorded"."""
    messages: int = 0
    messages_received: int = 0
    artifacts: int = 0
    artifacts_with_derived_from: int = 0
    consumers_instrumented: Dict[str, bool] = field(default_factory=dict)

    @property
    def message_receive_coverage(self) -> Optional[float]:
        return self.messages_received / self.messages if self.messages else None

    @property
    def derivation_coverage(self) -> Optional[float]:
        return self.artifacts_with_derived_from / self.artifacts if self.artifacts else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["message_receive_coverage"] = self.message_receive_coverage
        d["derivation_coverage"] = self.derivation_coverage
        return d


class ExecutionGraph:
    """Agent graph + work-unit DAG of one run (see the module docstring).

    ``artifact_consumers`` maps each artifact id to the agents that genuinely
    consumed it (in first-use order, producer included if it re-used its own
    artifact).  ``dropped_edges`` counts candidate edges discarded because
    they pointed backwards in time (inconsistent records); keeping only
    forward edges guarantees the DAG is acyclic."""

    def __init__(self, run_id: str, agents: List[str], agent_edges: List[AgentEdge], units: List[WorkUnit],
                 edges: List[DagEdge], coverage: Coverage, messages: List[MessageLink],
                 artifact_consumers: Dict[str, List[str]], schema_version: Optional[int] = None,
                 dropped_edges: int = 0) -> None:
        self.run_id = run_id
        self.agents = agents
        self.agent_edges = agent_edges
        self.units = units
        self.edges = edges
        self.coverage = coverage
        self.messages = messages
        self.artifact_consumers = artifact_consumers
        self.schema_version = schema_version
        self.dropped_edges = dropped_edges
        self._by_id = {u.node_id: u for u in units}
        self._preds: Dict[str, List[str]] = defaultdict(list)
        self._succs: Dict[str, List[str]] = defaultdict(list)
        for e in edges:
            self._preds[e.dst].append(e.src)
            self._succs[e.src].append(e.dst)

    @property
    def dag_units(self) -> List[WorkUnit]:
        """Units that take part in the DAG (nested calls excluded), in start order."""
        return [u for u in self.units if u.nested_in is None]

    def unit(self, node_id: str) -> WorkUnit:
        return self._by_id[node_id]

    def preds(self, node_id: str) -> List[str]:
        return list(self._preds.get(node_id, []))

    def succs(self, node_id: str) -> List[str]:
        return list(self._succs.get(node_id, []))

    def edges_of(self, *types: str) -> List[DagEdge]:
        return [e for e in self.edges if e.type in types]

    def fan_in(self, agent_id: str) -> int:
        """Distinct other agents with a message or artifact edge into ``agent_id``.
        Invocation edges are control flow, not communication."""
        return len({e.source for e in self.agent_edges if e.kind in _COMM_KINDS
                    and e.target == agent_id and e.source != agent_id})

    def fan_out(self, agent_id: str) -> int:
        """Distinct other agents ``agent_id`` reaches by message or artifact edges."""
        return len({e.target for e in self.agent_edges if e.kind in _COMM_KINDS
                    and e.source == agent_id and e.target != agent_id})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "schema_version": self.schema_version, "agents": list(self.agents),
            "agent_edges": [e.to_dict() for e in self.agent_edges],
            "units": [u.to_dict() for u in self.units], "edges": [e.to_dict() for e in self.edges],
            "messages": [m.to_dict() for m in self.messages],
            "artifact_consumers": {k: list(v) for k, v in self.artifact_consumers.items()},
            "coverage": self.coverage.to_dict(), "dropped_edges": self.dropped_edges,
        }

    def __repr__(self) -> str:
        return (f"ExecutionGraph(run_id={self.run_id!r}, agents={len(self.agents)}, units={len(self.units)}, "
                f"edges={len(self.edges)})")


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class _Item:
    """An entry of one span's sequence sweep: a real unit, or a child span
    standing in for all of its work (``child`` set)."""
    __slots__ = ("start", "end", "seq", "unit", "child")

    def __init__(self, start: float, end: float, seq: int, unit: Optional[WorkUnit] = None,
                 child: Optional[str] = None) -> None:
        self.start, self.end, self.seq, self.unit, self.child = start, end, seq, unit, child


class _Builder:
    def __init__(self, traj: Trajectory) -> None:
        self.traj = traj
        self.version = traj.metadata.get("schema_version")
        self.seq: Dict[str, int] = {}
        self.ev: Dict[str, Event] = {}
        self.units: List[WorkUnit] = []
        self.edges: List[DagEdge] = []
        self._edge_keys: Set[Tuple[str, str, str]] = set()
        self.dropped = 0

    # -- units ----------------------------------------------------------------
    def order(self, u: WorkUnit) -> Tuple[float, str, int]:
        return (round(u.start, 9), u.agent_id or "", self.seq[u.event_id])

    def key(self, u: WorkUnit) -> Tuple[float, int]:
        """Position used to keep edges forward in time.  A wait counts at its
        end: what it waited for may arrive while it is in progress."""
        t = u.end if u.kind == "wait" else u.start
        return (round(t, 9), self.seq[u.event_id])

    def make_units(self) -> None:
        for i, ev in enumerate(self.traj.events):
            self.seq[ev.event_id] = i
            kind = _UNIT_KINDS.get(ev.event_type)
            if kind is None:
                continue
            self.ev[ev.event_id] = ev
            start = ev.timestamp
            end = max(start, ev.end_timestamp) if kind in ("llm", "tool", "wait") else start
            self.units.append(WorkUnit(ev.event_id, kind, ev.event_id, ev.agent_id, ev.span_id, start, end,
                                       weight=(end - start) if kind in _CALL_KINDS else 0.0))
        self.mark_nesting()
        self.units.sort(key=self.order)
        self.by_id = {u.node_id: u for u in self.units}
        self.dag = [u for u in self.units if u.nested_in is None]

    def mark_nesting(self) -> None:
        calls = [u for u in self.units if u.kind in _CALL_KINDS]
        if self.version is not None:
            # Recorded explicitly since schema v2: the interval test below
            # cannot tell a tool run inside an LLM call from gathered calls.
            ids = {u.node_id for u in calls}
            for u in calls:
                parent = self.ev[u.event_id].metadata.get("within_call")
                if parent in ids and parent != u.node_id:
                    u.nested_in = parent
            return
        # Phase 1 data: a call lying inside another call of the same span is
        # taken as nested.  Identical intervals are treated as concurrent.
        by_span: Dict[str, List[WorkUnit]] = defaultdict(list)
        for u in calls:
            if u.span_id:
                by_span[u.span_id].append(u)
        for group in by_span.values():
            group.sort(key=lambda u: (u.start, -u.end, self.seq[u.event_id]))
            stack: List[WorkUnit] = []
            for u in group:
                while stack and not _contains(stack[-1], u):
                    stack.pop()
                if stack:
                    u.nested_in = stack[-1].node_id
                stack.append(u)

    def top(self, u: WorkUnit) -> WorkUnit:
        seen = set()
        while u.nested_in is not None and u.node_id not in seen:
            seen.add(u.node_id)
            u = self.by_id[u.nested_in]
        return u

    # -- edges ----------------------------------------------------------------
    def add_edge(self, src: WorkUnit, dst: WorkUnit, type_: str, inferred: bool = False,
                 ref: Optional[str] = None) -> bool:
        if src.node_id == dst.node_id:
            return False
        k = (src.node_id, dst.node_id, type_)
        if k in self._edge_keys:
            return True
        if not self.key(src) < self.key(dst):
            self.dropped += 1
            return False
        self._edge_keys.add(k)
        self.edges.append(DagEdge(src.node_id, dst.node_id, type_, inferred, ref))
        return True

    def sequence_and_invocation(self) -> None:
        """Rules 2 and 3: overlap-aware order within a span, with each child
        span standing in its parent's sequence as one interval."""
        spans = self.traj.spans
        own: Dict[str, List[WorkUnit]] = defaultdict(list)
        for u in self.dag:
            if u.span_id in spans:
                own[u.span_id].append(u)
        children: Dict[str, List[str]] = defaultdict(list)
        for s in spans.values():
            if s.parent_span_id in spans and s.parent_span_id != s.span_id:
                children[s.parent_span_id].append(s.span_id)

        def depth(sid: str) -> int:
            d, seen = 0, {sid}
            p = spans[sid].parent_span_id
            while p in spans and p not in seen:
                seen.add(p)
                d += 1
                p = spans[p].parent_span_id
            return d

        self.entry: Dict[str, List[WorkUnit]] = {}
        self.exit: Dict[str, List[WorkUnit]] = {}
        span_end: Dict[str, float] = {}
        # deepest spans first, so children are summarised before their parents
        for sid in sorted(spans, key=lambda s: -depth(s)):
            s = spans[sid]
            items = [_Item(u.start, u.end, self.seq[u.event_id], unit=u) for u in own.get(sid, [])]
            for c in children.get(sid, []):
                if self.entry.get(c):   # spans without any work are left out
                    cs = spans[c]
                    end = cs.end if cs.end is not None else span_end.get(c, cs.start)
                    items.append(_Item(cs.start, max(cs.start, end),
                                       self.seq.get(cs.start_event_id or "", len(self.seq)), child=c))
            if not items:
                continue
            items.sort(key=lambda it: (round(it.start, 9), it.seq))
            entry, final = self._sweep(items)
            self.entry[sid] = _dedup(n for it in entry for n in self._entry_of(it))
            self.exit[sid] = _dedup(n for it in final for n in self._exit_of(it))
            span_end[sid] = max(it.end for it in items)

    def _entry_of(self, it: _Item) -> List[WorkUnit]:
        return [it.unit] if it.unit is not None else self.entry[it.child]

    def _exit_of(self, it: _Item) -> List[WorkUnit]:
        return [it.unit] if it.unit is not None else self.exit[it.child]

    def _sweep(self, items: List[_Item]) -> Tuple[List[_Item], List[_Item]]:
        """Link each item to the *finished frontier* before it: the items that
        ended by the time it started and were not followed by another such
        item.  Items that overlap (gathered calls) get no edge between them,
        and the next item depends on all of them.  Returns the entry items
        (nothing before them) and the final frontier."""
        heap: List[Tuple[float, int, _Item]] = []    # started items, by end
        tail: deque = deque()                        # finished items ending after m, in end order
        m = float("-inf")                            # latest start among finished items
        last: Optional[Tuple[int, _Item]] = None     # (position, item) of the finished item starting at m
        entry: List[_Item] = []

        def finish(pos: int, x: _Item) -> None:
            nonlocal m, last
            if x.start > m + EPS:
                m, last = x.start, (pos, x)
            elif x.start >= m - EPS and (last is None or pos > last[0]):
                last = (pos, x)
            while tail and tail[0].end <= m + EPS:
                tail.popleft()
            if x.end > m + EPS:
                tail.append(x)

        def frontier() -> List[_Item]:
            out = list(tail)
            if last is not None and all(x is not last[1] for x in out):
                out.append(last[1])
            return out

        for pos, it in enumerate(items):
            while heap and heap[0][0] <= it.start + EPS:
                _, p, x = heapq.heappop(heap)
                finish(p, x)
            before = frontier()
            if not before:
                entry.append(it)
            for b in before:
                pseudo = b.child or it.child
                for src in self._exit_of(b):
                    for dst in self._entry_of(it):
                        self.add_edge(src, dst, "invocation" if pseudo else "sequence", ref=pseudo)
            heapq.heappush(heap, (it.end, pos, it))
        while heap:
            _, p, x = heapq.heappop(heap)
            finish(p, x)
        return entry, frontier()


def _contains(outer: WorkUnit, inner: WorkUnit) -> bool:
    if abs(outer.start - inner.start) <= EPS and abs(outer.end - inner.end) <= EPS:
        return False
    return outer.start <= inner.start + EPS and inner.end <= outer.end + EPS


def _dedup(units: Iterable[WorkUnit]) -> List[WorkUnit]:
    out: Dict[str, WorkUnit] = {}
    for u in units:
        out.setdefault(u.node_id, u)
    return list(out.values())


def build_graph(traj: Trajectory) -> ExecutionGraph:
    """Build the agent graph and the work-unit DAG of one trajectory.

    DAG edges (all O(n log n) apart from the edges themselves):

    1. a call nested in another call of its span is excluded (``nested_in``);
    2. *sequence*: within a span, each unit depends on the finished frontier
       before it, so overlapping (gathered) calls are not chained;
    3. *invocation*: a child span's first units depend on its parent's
       frontier at the child's start, and the parent's next units on the
       child's last units;
    4. *message*: send -> recorded receive (matched by message id); without
       one, in Phase 1 data only, send -> the receiver's first unit after the
       send (``inferred``);
    5. *artifact*: produce -> genuine consumption only (ARTIFACT_USE, an
       LLM/tool call's ``input_artifacts``, another artifact's
       ``derived_from``); carrying an artifact in a message is not using it;
    6. *wait*: what a wait was ``on`` -> the wait.
    """
    b = _Builder(traj)
    b.make_units()
    b.sequence_and_invocation()

    # consumption index (rule 5) -------------------------------------------------
    # (artifact, consuming agent, time, DAG node).  Calls nested in another
    # call are attributed to their top-level call, which is the DAG node.
    consumptions: List[Tuple[str, Optional[str], float, WorkUnit]] = []
    for u in b.units:
        if u.kind in ("use", "llm", "tool", "produce"):
            node = b.top(u)
            for a in b.ev[u.event_id].input_artifacts:
                consumptions.append((a, u.agent_id, u.start, node))
    produce_node: Dict[str, WorkUnit] = {}
    for u in b.units:
        if u.kind == "produce":
            for a in b.ev[u.event_id].output_artifacts:
                produce_node.setdefault(a, u)
    artifact_consumers: Dict[str, List[str]] = {}
    for a, agent, _, node in consumptions:
        if agent is not None and agent not in artifact_consumers.setdefault(a, []):
            artifact_consumers[a].append(agent)
        p = produce_node.get(a)
        if p is not None and p.node_id != node.node_id:
            b.add_edge(p, node, "artifact", ref=a)

    # messages (rule 4) ----------------------------------------------------------
    by_agent: Dict[str, List[WorkUnit]] = defaultdict(list)
    for u in b.dag:
        if u.agent_id is not None:
            by_agent[u.agent_id].append(u)
    starts = {a: [x.start for x in us] for a, us in by_agent.items()}
    receive_of: Dict[str, WorkUnit] = {}
    send_of: Dict[str, WorkUnit] = {}
    for u in b.units:
        mid = b.ev[u.event_id].message_id
        if mid and u.kind == "receive":
            receive_of.setdefault(mid, u)
        elif mid and u.kind == "send":
            send_of.setdefault(mid, u)
    used_by: Dict[Tuple[str, Optional[str]], List[float]] = defaultdict(list)
    for a, agent, t, _ in consumptions:
        used_by[(a, agent)].append(t)

    links: List[MessageLink] = []
    for s in b.units:
        if s.kind != "send":
            continue
        ev = b.ev[s.event_id]
        rcv = receive_of.get(ev.message_id) if ev.message_id else None
        target: Optional[WorkUnit] = None
        if rcv is not None and b.add_edge(s, rcv, "message", ref=ev.message_id):
            target = rcv
        elif b.version is None:
            # Phase 1 data has no receive events: infer delivery from timing.
            # Schema v2 records every receive, so without one it never arrived.
            cands = by_agent.get(ev.receiver or "", [])
            i = bisect_left(starts.get(ev.receiver or "", []), s.start - EPS)
            for cand in cands[i:]:
                if cand.node_id != s.node_id and b.key(s) < b.key(cand):
                    target = cand
                    break
            if target is not None:
                b.add_edge(s, target, "message", inferred=True, ref=ev.message_id)
        received = target is not None and target is rcv
        arts = list(ev.input_artifacts)
        used, used_inferred = False, False
        if target is not None:
            used = any(t >= s.start - EPS for a in arts for t in used_by.get((a, ev.receiver), []))
            if not used:
                # a later call may have used the content; for a message with
                # artifacts, a call listing other inputs is recorded non-use
                t0 = target.start
                used = any(x.kind in _CALL_KINDS and x.start >= t0 - EPS
                           and not (arts and b.ev[x.event_id].input_artifacts)
                           for x in by_agent.get(ev.receiver or "", []))
                used_inferred = used
        links.append(MessageLink(s.node_id, target.node_id if target else None, ev.message_id, ev.sender,
                                 ev.receiver, ev.kind, ev.content_tokens, arts, received, used,
                                 used_inferred))

    # waits (rule 6) -------------------------------------------------------------
    for w in b.units:
        if w.kind != "wait":
            continue
        ev = b.ev[w.event_id]
        for a in ev.input_artifacts:
            if a in produce_node:
                b.add_edge(produce_node[a], w, "wait", ref=a)
        for mid in ev.metadata.get("on_messages", []) or []:
            if mid in send_of:
                b.add_edge(send_of[mid], w, "wait", ref=mid)
        for agent in ev.metadata.get("on_agents", []) or []:
            if agent == w.agent_id:
                continue
            done = [x for x in by_agent.get(agent, []) if x.end <= w.end + EPS and b.key(x) < b.key(w)]
            if done:
                b.add_edge(done[-1], w, "wait", inferred=True, ref=agent)

    # agent graph ----------------------------------------------------------------
    agents = traj.agent_ids()
    extra = set()
    for ev in traj.events:
        for a in (ev.agent_id, ev.sender, ev.receiver):
            if a is not None and a not in agents:
                extra.add(a)
    agents += sorted(extra)
    rank = {a: i for i, a in enumerate(agents)}

    agent_edges: Dict[Tuple[str, str, str], AgentEdge] = {}

    def edge(src: str, dst: str, kind: str) -> AgentEdge:
        k = (src, dst, kind)
        if k not in agent_edges:
            agent_edges[k] = AgentEdge(src, dst, kind)
        return agent_edges[k]

    for m in links:
        if m.sender is None or m.receiver is None:
            continue
        e = edge(m.sender, m.receiver, "message")
        e.count += 1
        e.tokens += m.tokens
        e.used_count += int(m.used)
        e.inferred_count += int(m.inferred)
    for a, consumers in artifact_consumers.items():
        art = traj.artifacts.get(a)
        if art is None or art.producer is None:
            continue
        for c in consumers:
            if c != art.producer:
                e = edge(art.producer, c, "artifact")
                e.count += 1
                e.used_count += 1
                e.tokens += art.tokens
    for s in traj.spans.values():
        parent = traj.spans.get(s.parent_span_id or "")
        if parent is not None and parent.agent_id != s.agent_id:
            edge(parent.agent_id, s.agent_id, "invocation").count += 1
    kind_rank = {"invocation": 0, "message": 1, "artifact": 2}
    ordered = sorted(agent_edges.values(), key=lambda e: (kind_rank[e.kind], rank.get(e.source, 0),
                                                           rank.get(e.target, 0)))

    # coverage -------------------------------------------------------------------
    instrumented = {a: False for a in agents}
    for u in b.units:
        ev = b.ev[u.event_id]
        if u.agent_id is None:
            continue
        if u.kind == "use" or (u.kind in _CALL_KINDS and ev.input_artifacts):
            instrumented[u.agent_id] = True
    for art in traj.artifacts.values():
        if art.derived_from and art.producer is not None:   # [] names no input: no evidence of recording use
            instrumented[art.producer] = True
    derived = sum(1 for a in traj.artifacts.values() if a.derived_from is not None)
    cov = Coverage(messages=len(links), messages_received=sum(1 for m in links if m.received),
                   artifacts=len(traj.artifacts), artifacts_with_derived_from=derived,
                   consumers_instrumented=instrumented)
    return ExecutionGraph(traj.run_id, agents, ordered, b.units, b.edges, cov, links, artifact_consumers,
                          schema_version=b.version, dropped_edges=b.dropped)
