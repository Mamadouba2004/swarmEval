"""Layer 3 -- execution graph of one trajectory (Phase 2: coordination analysis).

Two views, built only from what was recorded:

* the **agent graph**: who messaged whom, whose artifacts were consumed by
  whom, and who invoked whom;
* the **work-unit DAG**: LLM and tool calls plus point events (send, receive,
  produce, use, wait) with typed dependency edges, the input to critical path
  and parallelism analysis.

Where the record is silent (Phase 1 data has no receive events, message ids or
``derived_from``; ``inbox()`` reads record nothing) an edge is either inferred
from timing and flagged ``inferred=True``, or missing.  ``Coverage`` says how
much was recorded, so a reader knows how far the DAG can be trusted: missing
edges under-count dependencies and over-state possible parallelism.

Events are appended when they complete but timestamped when they start, so
everything here works on start-ordered units, never on event order.  Ties in
start time are broken by agent id and then by recording order, which for
point events is the order in which they happened.

DAG edges (all O(n log n) apart from the edges themselves):

1. a call nested in another call of its span is excluded (``nested_in``);
2. *sequence*: within a span, each unit depends on the finished frontier
   before it, so overlapping (gathered) calls are not chained;
3. *invocation*: a child span's first units depend on its parent's
   frontier at the child's start, and the parent's next units on the
   child's last units;
4. *message*: send -> recorded receive (matched by message id); without
   one (Phase 1 data, or a message only read with ``inbox()``), send ->
   the receiver's first unit after the send (``inferred``);
5. *artifact*: produce -> genuine consumption only (ARTIFACT_USE, an
   LLM/tool call's ``input_artifacts``, another artifact's
   ``derived_from``); carrying an artifact in a message is not using it;
6. *wait*: what a wait was ``on`` -> the wait.

Nesting comes from ``metadata["within_call"]`` (schema v2). On Phase 1 data a call lying inside another call of
the same span is taken as nested, and identical intervals are treated as concurrent. Calls nested in another
call are attributed to their top-level call, which is the DAG node.

Classes:
- WorkUnit: One node of the DAG. ``weight`` is the time the unit itself spends working (LLM/tool duration; 0 for
  point events and waits). A call made inside another call of the same span has ``nested_in`` set and takes no part
  in the DAG: its time is already inside its parent's.
- DagEdge: ``dst`` could not start before ``src`` finished. ``ref`` names what links them: a message id, an artifact
  id, a child span id or an agent id.
- AgentEdge: Aggregated interaction between two agents. For messages ``count`` is the number of messages, ``tokens``
  their recorded tokens, ``used_count`` how many were used by the receiver and ``inferred_count`` how many had their
  delivery or use inferred rather than recorded. For artifacts ``count`` is the number of distinct artifacts the
  target consumed.
- MessageLink: What became of one message. ``receive_node`` is the recorded receive or, without one (Phase 1 data,
  or a message only read with ``inbox()``), the receiver's first unit after the send (``received`` False); ``None``
  means the receiver did nothing after it. ``received`` is True whenever a receive was recorded, even when its DAG
  edge was dropped because the receive is stamped before the send. A message is ``used`` when the receiver consumed
  one of its artifacts or, failing that, did LLM/tool work after receiving it -- a timing heuristic, flagged
  ``used_inferred``. For a message carrying artifacts only calls that record no ``input_artifacts`` count: one
  listing other inputs shows the artifacts were not used.
- Coverage: How much provenance was recorded. Ratios are ``None`` when there is nothing to measure (no messages / no
  artifacts), never a silent 0 or 1. ``consumers_instrumented[a]`` is True when agent ``a`` records what it uses (an
  ARTIFACT_USE, an LLM/tool call with ``input_artifacts``, or an artifact with a non-empty ``derived_from``);
  without that, "a never used X" cannot be told apart from "a's use of X was not recorded".
- ExecutionGraph: Agent graph + work-unit DAG of one run (see the module docstring). ``artifact_consumers`` maps
  each artifact id to the agents that genuinely consumed it (in first-use order, producer included if it re-used its
  own artifact). ``dropped_edges`` counts candidate edges discarded because they pointed backwards in time
  (inconsistent records); keeping only forward edges guarantees the DAG is acyclic.
- _Item: An entry of one span's sequence sweep: a real unit, or a child span standing in for all of its work
  (``child`` set).
"""
from __future__ import annotations

import heapq
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..schema import Event, EventType, Trajectory

EPS = 1e-9

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


@dataclass
class WorkUnit:
    node_id: str
    kind: str
    event_id: str
    agent_id: str | None
    span_id: str | None
    start: float
    end: float
    weight: float = 0.0
    nested_in: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The unit as a JSON-ready dict."""
        return asdict(self)


@dataclass
class DagEdge:
    src: str
    dst: str
    type: str
    inferred: bool = False
    ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The edge as a JSON-ready dict."""
        return asdict(self)


@dataclass
class AgentEdge:
    source: str
    target: str
    kind: str
    count: int = 0
    tokens: int = 0
    inferred_count: int = 0
    used_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The edge as a JSON-ready dict."""
        return asdict(self)


@dataclass
class MessageLink:
    send_node: str
    receive_node: str | None
    message_id: str | None
    sender: str | None
    receiver: str | None
    kind: str | None
    tokens: int
    artifacts: list[str]
    received: bool
    used: bool
    used_inferred: bool

    @property
    def inferred(self) -> bool:
        """True if delivery or use was inferred rather than recorded."""
        return not self.received or self.used_inferred

    def to_dict(self) -> dict[str, Any]:
        """The link as a JSON-ready dict."""
        d = asdict(self)
        d["inferred"] = self.inferred
        return d


@dataclass
class Coverage:
    messages: int = 0
    messages_received: int = 0
    artifacts: int = 0
    artifacts_with_derived_from: int = 0
    consumers_instrumented: dict[str, bool] = field(default_factory=dict)

    @property
    def message_receive_coverage(self) -> float | None:
        """Share of messages with a recorded receive, None without messages."""
        return self.messages_received / self.messages if self.messages else None

    @property
    def derivation_coverage(self) -> float | None:
        """Share of artifacts with recorded ``derived_from``, None without artifacts."""
        return self.artifacts_with_derived_from / self.artifacts if self.artifacts else None

    def to_dict(self) -> dict[str, Any]:
        """Coverage as a JSON-ready dict, ratios included."""
        d = asdict(self)
        d["message_receive_coverage"] = self.message_receive_coverage
        d["derivation_coverage"] = self.derivation_coverage
        return d


class ExecutionGraph:

    def __init__(self, run_id: str, agents: list[str], agent_edges: list[AgentEdge], units: list[WorkUnit],
                 edges: list[DagEdge], coverage: Coverage, messages: list[MessageLink],
                 artifact_consumers: dict[str, list[str]], schema_version: int | None = None,
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
        self._preds: dict[str, list[str]] = defaultdict(list)
        self._succs: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            self._preds[e.dst].append(e.src)
            self._succs[e.src].append(e.dst)

    @property
    def dag_units(self) -> list[WorkUnit]:
        """Units that take part in the DAG (nested calls excluded), in start order."""
        return [u for u in self.units if u.nested_in is None]

    def unit(self, node_id: str) -> WorkUnit:
        """The unit with ``node_id``."""
        return self._by_id[node_id]

    def preds(self, node_id: str) -> list[str]:
        """Ids of the units ``node_id`` depends on."""
        return list(self._preds.get(node_id, []))

    def succs(self, node_id: str) -> list[str]:
        """Ids of the units that depend on ``node_id``."""
        return list(self._succs.get(node_id, []))

    def edges_of(self, *types: str) -> list[DagEdge]:
        """DAG edges of the given types."""
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

    def to_dict(self) -> dict[str, Any]:
        """The graph as a JSON-ready dict."""
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


class _Item:
    __slots__ = ("start", "end", "seq", "unit", "child")

    def __init__(self, start: float, end: float, seq: int, unit: WorkUnit | None = None,
                 child: str | None = None) -> None:
        self.start, self.end, self.seq, self.unit, self.child = start, end, seq, unit, child


class _Builder:
    def __init__(self, traj: Trajectory) -> None:
        self.traj = traj
        self.version = traj.metadata.get("schema_version")
        self.seq: dict[str, int] = {}
        self.ev: dict[str, Event] = {}
        self.units: list[WorkUnit] = []
        self.edges: list[DagEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.dropped = 0

    def order(self, u: WorkUnit) -> tuple[float, str, int]:
        """Sort key: start time, then agent id, then recording order."""
        return (round(u.start, 9), u.agent_id or "", self.seq[u.event_id])

    def key(self, u: WorkUnit) -> tuple[float, int]:
        """Position used to keep edges forward in time.  A wait counts at its
        end: what it waited for may arrive while it is in progress."""
        t = u.end if u.kind == "wait" else u.start
        return (round(t, 9), self.seq[u.event_id])

    def make_units(self) -> None:
        """Create a work unit for every LLM, tool and point event, then mark nesting."""
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
        """Set ``nested_in`` from recorded ``within_call``, or by interval containment on Phase 1 data."""
        calls = [u for u in self.units if u.kind in _CALL_KINDS]
        if self.version is not None:
            ids = {u.node_id for u in calls}
            for u in calls:
                parent = self.ev[u.event_id].metadata.get("within_call")
                if parent in ids and parent != u.node_id:
                    u.nested_in = parent
            return
        by_span: dict[str, list[WorkUnit]] = defaultdict(list)
        for u in calls:
            if u.span_id:
                by_span[u.span_id].append(u)
        for group in by_span.values():
            group.sort(key=lambda u: (u.start, -u.end, self.seq[u.event_id]))
            stack: list[WorkUnit] = []
            for u in group:
                while stack and not _contains(stack[-1], u):
                    stack.pop()
                if stack:
                    u.nested_in = stack[-1].node_id
                stack.append(u)

    def top(self, u: WorkUnit) -> WorkUnit:
        """The top-level call a unit is nested in, or the unit itself."""
        seen = set()
        while u.nested_in is not None and u.node_id not in seen:
            seen.add(u.node_id)
            u = self.by_id[u.nested_in]
        return u

    def add_edge(self, src: WorkUnit, dst: WorkUnit, type_: str, inferred: bool = False,
                 ref: str | None = None) -> bool:
        """Add a DAG edge if it points forward in time; count it as dropped otherwise."""
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
        own: dict[str, list[WorkUnit]] = defaultdict(list)
        for u in self.dag:
            if u.span_id in spans:
                own[u.span_id].append(u)
        children: dict[str, list[str]] = defaultdict(list)
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

        self.entry: dict[str, list[WorkUnit]] = {}
        self.exit: dict[str, list[WorkUnit]] = {}
        span_end: dict[str, float] = {}
        for sid in sorted(spans, key=lambda s: -depth(s)):
            s = spans[sid]
            items = [_Item(u.start, u.end, self.seq[u.event_id], unit=u) for u in own.get(sid, [])]
            for c in children.get(sid, []):
                if self.entry.get(c):
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

    def _entry_of(self, it: _Item) -> list[WorkUnit]:
        """First units of an item: the unit itself, or a child span's entry units."""
        return [it.unit] if it.unit is not None else self.entry[it.child]

    def _exit_of(self, it: _Item) -> list[WorkUnit]:
        """Last units of an item: the unit itself, or a child span's exit units."""
        return [it.unit] if it.unit is not None else self.exit[it.child]

    def _sweep(self, items: list[_Item]) -> tuple[list[_Item], list[_Item]]:
        """Link each item to the finished frontier before it, leaving overlapping (gathered) items unlinked;
        return the entry items and the final frontier."""
        heap: list[tuple[float, int, _Item]] = []
        tail: deque = deque()
        m = float("-inf")
        last: tuple[int, _Item] | None = None
        entry: list[_Item] = []

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

        def frontier() -> list[_Item]:
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
    """True if ``inner`` lies within ``outer``; identical intervals count as concurrent, not nested."""
    if abs(outer.start - inner.start) <= EPS and abs(outer.end - inner.end) <= EPS:
        return False
    return outer.start <= inner.start + EPS and inner.end <= outer.end + EPS


def _dedup(units: Iterable[WorkUnit]) -> list[WorkUnit]:
    """Units with duplicates removed, keeping first-seen order."""
    out: dict[str, WorkUnit] = {}
    for u in units:
        out.setdefault(u.node_id, u)
    return list(out.values())


def build_graph(traj: Trajectory) -> ExecutionGraph:
    """Build the agent graph and the work-unit DAG of one trajectory, by the edge rules in the module overview."""
    b = _Builder(traj)
    b.make_units()
    b.sequence_and_invocation()

    consumptions: list[tuple[str, str | None, float, WorkUnit]] = []
    for u in b.units:
        if u.kind in ("use", "llm", "tool", "produce"):
            node = b.top(u)
            for a in b.ev[u.event_id].input_artifacts:
                consumptions.append((a, u.agent_id, u.start, node))
    produce_node: dict[str, WorkUnit] = {}
    for u in b.units:
        if u.kind == "produce":
            for a in b.ev[u.event_id].output_artifacts:
                produce_node.setdefault(a, u)
    artifact_consumers: dict[str, list[str]] = {}
    for a, agent, _, node in consumptions:
        if agent is not None and agent not in artifact_consumers.setdefault(a, []):
            artifact_consumers[a].append(agent)
        p = produce_node.get(a)
        if p is not None and p.node_id != node.node_id:
            b.add_edge(p, node, "artifact", ref=a)

    by_agent: dict[str, list[WorkUnit]] = defaultdict(list)
    for u in b.dag:
        if u.agent_id is not None:
            by_agent[u.agent_id].append(u)
    starts = {a: [x.start for x in us] for a, us in by_agent.items()}
    receive_of: dict[str, WorkUnit] = {}
    send_of: dict[str, WorkUnit] = {}
    for u in b.units:
        mid = b.ev[u.event_id].message_id
        if mid and u.kind == "receive":
            receive_of.setdefault(mid, u)
        elif mid and u.kind == "send":
            send_of.setdefault(mid, u)
    used_by: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    for a, agent, t, _ in consumptions:
        used_by[(a, agent)].append(t)

    links: list[MessageLink] = []
    for s in b.units:
        if s.kind != "send":
            continue
        ev = b.ev[s.event_id]
        rcv = receive_of.get(ev.message_id) if ev.message_id else None
        target: WorkUnit | None = None
        if rcv is not None:
            target = rcv
            b.add_edge(s, rcv, "message", ref=ev.message_id)
        else:
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
                t0 = max(target.start, s.start)
                used = any(x.kind in _CALL_KINDS and x.start >= t0 - EPS
                           and not (arts and b.ev[x.event_id].input_artifacts)
                           for x in by_agent.get(ev.receiver or "", []))
                used_inferred = used
        links.append(MessageLink(s.node_id, target.node_id if target else None, ev.message_id, ev.sender,
                                 ev.receiver, ev.kind, ev.content_tokens, arts, received, used,
                                 used_inferred))

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
                b.add_edge(max(done, key=lambda x: (x.end, b.key(x))), w, "wait", inferred=True, ref=agent)

    agents = traj.agent_ids()
    extra = set()
    for ev in traj.events:
        for a in (ev.agent_id, ev.sender, ev.receiver):
            if a is not None and a not in agents:
                extra.add(a)
    agents += sorted(extra)
    rank = {a: i for i, a in enumerate(agents)}

    agent_edges: dict[tuple[str, str, str], AgentEdge] = {}

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

    instrumented = {a: False for a in agents}
    for u in b.units:
        ev = b.ev[u.event_id]
        if u.agent_id is None:
            continue
        if u.kind == "use" or (u.kind in _CALL_KINDS and ev.input_artifacts):
            instrumented[u.agent_id] = True
    for art in traj.artifacts.values():
        if art.derived_from and art.producer is not None:
            instrumented[art.producer] = True
    derived = sum(1 for a in traj.artifacts.values() if a.derived_from is not None)
    cov = Coverage(messages=len(links), messages_received=sum(1 for m in links if m.received),
                   artifacts=len(traj.artifacts), artifacts_with_derived_from=derived,
                   consumers_instrumented=instrumented)
    return ExecutionGraph(traj.run_id, agents, ordered, b.units, b.edges, cov, links, artifact_consumers,
                          schema_version=b.version, dropped_edges=b.dropped)
