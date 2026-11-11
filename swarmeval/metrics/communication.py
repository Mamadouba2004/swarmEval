"""Layer 3 -- communication overhead of one run, and its summary over runs (Phase 2).

How much of the run went into talking rather than working, what kind of
talking it was, and how it flowed between agents.  Communication is not
automatically waste: a task assignment or a handed-over result is often
necessary.  These numbers say how much there was and where, not whether it
was worth it.

Definitions:

* **message body tokens** = a message's ``content_tokens`` minus the tokens of
  the artifacts it carried.  The recorder includes attached artifacts in
  ``content_tokens``; an artifact's tokens are counted once, when it is
  produced, not again every time it is passed along.
* **Communication / Work** (``comm_work_ratio``) = message body tokens as a
  share of LLM tokens: Σ message body tokens ÷ Σ (prompt + output tokens) of
  all LLM calls.  ``None`` when the run made no LLM calls.
* **hub share** = the largest share of messages any one agent was party to
  (as sender or receiver).  A star topology, where every message goes to or
  from one hub, has hub share 1.0.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..graph.builder import ExecutionGraph, build_graph
from ..schema import EventType, MessageKind, Trajectory
from ..stats import Summary, mean, summarize

# Message kinds grouped by what the message does for the run.
CATEGORIES: Dict[str, str] = {
    MessageKind.TASK_ASSIGNMENT.value: "coordination",
    MessageKind.COORDINATION.value: "coordination",
    MessageKind.REQUEST.value: "coordination",
    MessageKind.INFORMATION.value: "information",
    MessageKind.RESULT.value: "work_product",
    MessageKind.FEEDBACK.value: "work_product",
}
CATEGORY_NAMES = ("coordination", "information", "work_product", "other")


def message_category(kind: Optional[str]) -> str:
    return CATEGORIES.get(kind or "", "other")


@dataclass
class AgentCommunication:
    """One agent's part in the run's messaging.  ``fan_out`` / ``fan_in``
    count distinct other agents it sent to / received from, over messages and
    artifact consumption.  ``relay_fraction`` = messages it sent carrying an
    artifact produced by another agent ÷ messages it sent (``None`` if it sent
    none): high values mark an agent passing others' work along."""
    agent_id: str
    messages_sent: int = 0
    messages_received: int = 0
    body_tokens_sent: int = 0
    fan_out: int = 0
    fan_in: int = 0
    relayed_messages: int = 0
    relay_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommunicationMetrics:
    """Communication overhead of one run (see the module docstring for the
    definitions).  ``by_category`` maps coordination / information /
    work_product / other to ``{"messages", "body_tokens"}``.
    ``repeated_messages`` counts extra copies of a (sender, receiver, content)
    already sent (messages without a content hash are never repeats or part
    of a broadcast); ``broadcasts`` counts one content sent from one span to two
    or more receivers (``broadcast_messages`` the messages involved; messages
    sent outside any span are never grouped).
    ``used_edge_ratio`` = messages the receiver used ÷ messages, of which
    ``used_inferred`` were judged used by timing alone."""
    messages: int = 0
    body_tokens: int = 0
    attached_artifact_tokens: int = 0
    llm_tokens: int = 0
    comm_work_ratio: Optional[float] = None
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    repeated_messages: int = 0
    broadcasts: int = 0
    broadcast_messages: int = 0
    per_agent: Dict[str, AgentCommunication] = field(default_factory=dict)
    hub_agent: Optional[str] = None
    hub_share: Optional[float] = None
    used_messages: int = 0
    used_inferred: int = 0
    used_edge_ratio: Optional[float] = None
    receive_delay_mean_s: Optional[float] = None
    receive_delay_max_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_agent"] = {k: v.to_dict() for k, v in self.per_agent.items()}
        return d


def body_tokens(ev: Any, traj: Trajectory) -> int:
    """A message's own tokens: ``content_tokens`` minus attached artifacts
    (which the recorder folded in).  Clamped at 0 in case the record was
    made by hand or an artifact's tokens were re-estimated."""
    attached = sum(traj.artifacts[a].tokens for a in ev.input_artifacts if a in traj.artifacts)
    return max(0, ev.content_tokens - attached)


def compute_communication(traj: Trajectory, graph: Optional[ExecutionGraph] = None) -> CommunicationMetrics:
    graph = graph or build_graph(traj)
    msgs = traj.messages()
    out = CommunicationMetrics(messages=len(msgs))
    out.by_category = {c: {"messages": 0, "body_tokens": 0} for c in CATEGORY_NAMES}

    llm_tokens = sum(e.prompt_tokens + e.output_tokens for e in traj.llm_calls())
    out.llm_tokens = llm_tokens

    per_agent: Dict[str, AgentCommunication] = {a: AgentCommunication(a) for a in graph.agents}

    def agent(a: str) -> AgentCommunication:
        if a not in per_agent:
            per_agent[a] = AgentCommunication(a)
        return per_agent[a]

    repeats: Counter = Counter()
    fanout_groups: Dict[Tuple[Any, ...], Set[str]] = defaultdict(set)
    fanout_counts: Counter = Counter()
    involvement: Counter = Counter()
    for ev in msgs:
        body = body_tokens(ev, traj)
        out.body_tokens += body
        out.attached_artifact_tokens += ev.content_tokens - body
        cat = out.by_category[message_category(ev.kind)]
        cat["messages"] += 1
        cat["body_tokens"] += body
        # an unhashed (imported) message's content is unknown, so it repeats nothing
        if ev.content_hash is not None:
            repeats[(ev.sender, ev.receiver, ev.content_hash)] += 1
            if ev.receiver is not None and ev.span_id is not None:   # no span, nothing ties sends together
                g = (ev.sender, ev.span_id, ev.content_hash)
                fanout_groups[g].add(ev.receiver)
                fanout_counts[g] += 1
        # an agent messaging itself is involved once, not twice
        for a in {ev.sender, ev.receiver} - {None}:
            involvement[a] += 1
        if ev.sender is not None:
            s = agent(ev.sender)
            s.messages_sent += 1
            s.body_tokens_sent += body
            producers = [traj.artifacts[a].producer for a in ev.input_artifacts if a in traj.artifacts]
            if any(p is not None and p != ev.sender for p in producers):
                s.relayed_messages += 1
        if ev.receiver is not None:
            agent(ev.receiver).messages_received += 1
    out.comm_work_ratio = out.body_tokens / llm_tokens if llm_tokens else None
    out.repeated_messages = sum(n - 1 for n in repeats.values() if n > 1)
    for g, receivers in fanout_groups.items():
        if len(receivers) >= 2:
            out.broadcasts += 1
            out.broadcast_messages += fanout_counts[g]

    for a, ac in per_agent.items():
        ac.fan_out, ac.fan_in = graph.fan_out(a), graph.fan_in(a)
        ac.relay_fraction = ac.relayed_messages / ac.messages_sent if ac.messages_sent else None
    out.per_agent = per_agent

    if msgs and involvement:
        rank = {a: i for i, a in enumerate(per_agent)}
        hub = max(involvement, key=lambda a: (involvement[a], -rank.get(a, len(rank))))
        out.hub_agent, out.hub_share = hub, involvement[hub] / len(msgs)

    links = graph.messages
    if links:
        out.used_messages = sum(1 for m in links if m.used)
        out.used_inferred = sum(1 for m in links if m.used and m.used_inferred)
        out.used_edge_ratio = out.used_messages / len(links)

    delays = [float(e.metadata["delay_s"]) for e in traj.events_of(EventType.MESSAGE_RECEIVE)
              if isinstance(e.metadata.get("delay_s"), (int, float))]
    if delays:
        out.receive_delay_mean_s, out.receive_delay_max_s = mean(delays), max(delays)
    return out


def _known(values: Sequence[Optional[float]]) -> Optional[Summary]:
    known = [v for v in values if v is not None]
    return summarize(known) if known else None


@dataclass
class CommunicationSummary:
    """Communication metrics over runs.  Ratio summaries cover the runs where
    the ratio is defined (``n``); ``None`` when no run defined it.
    ``hub_agents`` counts how often each agent was the hub."""
    n_runs: int
    messages: Summary
    body_tokens: Summary
    comm_work_ratio: Optional[Summary]
    hub_share: Optional[Summary]
    used_edge_ratio: Optional[Summary]
    repeated_messages: Summary
    broadcasts: Summary
    receive_delay_mean_s: Optional[Summary]
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hub_agents: Dict[str, int] = field(default_factory=dict)
    per_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}


def summarize_communication(per_run: Sequence[CommunicationMetrics]) -> CommunicationSummary:
    S = summarize
    by_cat = {c: {"messages": mean([r.by_category.get(c, {}).get("messages", 0) for r in per_run]),
                  "body_tokens": mean([r.by_category.get(c, {}).get("body_tokens", 0) for r in per_run])}
              for c in CATEGORY_NAMES} if per_run else {}
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in per_run:
        for aid, ac in r.per_agent.items():
            for k in ("messages_sent", "messages_received", "body_tokens_sent", "fan_out", "fan_in"):
                acc[aid][k].append(float(getattr(ac, k)))
            if ac.relay_fraction is not None:
                acc[aid]["relay_fraction"].append(ac.relay_fraction)
    per_agent = {aid: {k: mean(v) for k, v in d.items()} for aid, d in acc.items()}
    return CommunicationSummary(
        n_runs=len(per_run), messages=S([r.messages for r in per_run]),
        body_tokens=S([r.body_tokens for r in per_run]),
        comm_work_ratio=_known([r.comm_work_ratio for r in per_run]),
        hub_share=_known([r.hub_share for r in per_run]),
        used_edge_ratio=_known([r.used_edge_ratio for r in per_run]),
        repeated_messages=S([r.repeated_messages for r in per_run]),
        broadcasts=S([r.broadcasts for r in per_run]),
        receive_delay_mean_s=_known([r.receive_delay_mean_s for r in per_run]),
        by_category=by_cat, hub_agents=dict(Counter(r.hub_agent for r in per_run if r.hub_agent)),
        per_agent=per_agent)
