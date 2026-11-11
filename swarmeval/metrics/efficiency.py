"""Layer 2 -- efficiency: tokens, cost and latency for one trajectory.

Phase 1 reports what was directly recorded: token usage, call counts, cost,
wall clock time and per-agent time.  Critical path and parallelism need the
execution graph and live in ``metrics/dependency.py`` (Phase 2).

One exception counts as one error: ERROR events and failed calls marked ``propagated`` (an already-counted
exception passing through enclosing spans or a tool calling a tool) stay in the trajectory but are not
re-counted. Agents are ordered by first start, then id, because event order varies under concurrency.

An unknown cost has one of two causes, counted separately: ``streamed_cost_calls`` (a stream has no usage) and
``unpriced_calls`` (the model has no pricing entry). ``duration_complete`` is False and
``unknown_duration_calls`` counts the calls when a call's duration is unknown (a bare stream).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..pricing import PricingTable
from ..schema import EventType, Trajectory


def union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by a set of possibly overlapping intervals."""
    ivs = sorted((a, b) for a, b in intervals if b > a)
    total = 0.0
    cur_a, cur_b = None, None
    for a, b in ivs:
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    if cur_a is not None:
        total += cur_b - cur_a
    return total


@dataclass
class AgentEfficiency:
    agent_id: str
    role: str | None = None
    spans: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    messages_sent: int = 0
    artifacts_produced: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    cost_complete: bool = True
    self_time_s: float = 0.0
    busy_time_s: float = 0.0
    idle_time_s: float = 0.0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus output tokens."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """The record as a JSON-ready dict."""
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class EfficiencyMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    messages: int = 0
    message_tokens: int = 0
    cost_usd: float | None = None
    cost_complete: bool = True
    duration_complete: bool = True
    unknown_duration_calls: int = 0
    streamed_cost_calls: int = 0
    unpriced_calls: int = 0
    wall_clock_s: float = 0.0
    agent_time_s: float = 0.0
    llm_latency_s: float = 0.0
    tool_latency_s: float = 0.0
    waiting_time_s: float = 0.0
    errors: int = 0
    per_agent: dict[str, AgentEfficiency] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The metrics as a JSON-ready dict."""
        d = asdict(self)
        d["per_agent"] = {k: v.to_dict() for k, v in self.per_agent.items()}
        return d


def event_cost(ev, pricing: PricingTable | None) -> float | None:
    """Cost of an LLM_CALL event: re-priced from its model and tokens when a
    pricing table is given, else the cost recorded at run time."""
    if pricing is None or ev.metadata.get("cost_unknown"):
        return ev.cost_usd
    return pricing.cost(ev.model, ev.input_tokens, ev.output_tokens, ev.cache_read_tokens, ev.cache_write_tokens)


def _agent(per: dict[str, AgentEfficiency], aid: str) -> AgentEfficiency:
    """The record of agent ``aid``, created on first use."""
    if aid not in per:
        per[aid] = AgentEfficiency(aid)
    return per[aid]


def _count_call_error(ev, m: EfficiencyMetrics, per: dict[str, AgentEfficiency]) -> None:
    """Count a failed LLM or tool call, unless it is an already-counted exception passing through."""
    if ev.status == "error" and not ev.metadata.get("propagated"):
        m.errors += 1
        if ev.agent_id:
            _agent(per, ev.agent_id).errors += 1


def _tally_events(traj: Trajectory, pricing: PricingTable | None, m: EfficiencyMetrics,
                  per: dict[str, AgentEfficiency]) -> dict[str, list[tuple[float, float]]]:
    """Add every event's tokens, calls, cost, messages and errors to ``m`` and ``per``; return each span's call
    intervals."""
    busy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ev in traj.events:
        if ev.event_type == EventType.LLM_CALL.value:
            cost = event_cost(ev, pricing)
            m.llm_calls += 1
            m.input_tokens += ev.prompt_tokens
            m.output_tokens += ev.output_tokens
            m.llm_latency_s += ev.duration_s
            if ev.metadata.get("duration_unknown"):
                m.unknown_duration_calls += 1
                m.duration_complete = False
            if cost is None:
                m.cost_complete = False
                if ev.metadata.get("cost_unknown"):
                    m.streamed_cost_calls += 1
                else:
                    m.unpriced_calls += 1
            else:
                m.cost_usd = (m.cost_usd or 0.0) + cost
            if ev.span_id:
                busy[ev.span_id].append((ev.timestamp, ev.end_timestamp))
            if ev.agent_id:
                a = _agent(per, ev.agent_id)
                a.llm_calls += 1
                a.input_tokens += ev.prompt_tokens
                a.output_tokens += ev.output_tokens
                if cost is None:
                    a.cost_complete = False
                else:
                    a.cost_usd = (a.cost_usd or 0.0) + cost
            _count_call_error(ev, m, per)
        elif ev.event_type == EventType.TOOL_CALL.value:
            m.tool_calls += 1
            m.tool_latency_s += ev.duration_s
            if ev.span_id:
                busy[ev.span_id].append((ev.timestamp, ev.end_timestamp))
            if ev.agent_id:
                _agent(per, ev.agent_id).tool_calls += 1
            _count_call_error(ev, m, per)
        elif ev.event_type == EventType.MESSAGE.value:
            m.messages += 1
            m.message_tokens += ev.content_tokens
            if ev.sender:
                _agent(per, ev.sender).messages_sent += 1
        elif ev.event_type == EventType.ARTIFACT.value and ev.agent_id:
            _agent(per, ev.agent_id).artifacts_produced += 1
        elif ev.event_type == EventType.ERROR.value and not ev.metadata.get("propagated"):
            m.errors += 1
            if ev.agent_id and ev.span_id:
                _agent(per, ev.agent_id).errors += 1
    m.total_tokens = m.input_tokens + m.output_tokens
    if m.llm_calls == 0:
        m.cost_usd = 0.0
    return busy


def _span_timing(traj: Trajectory, per: dict[str, AgentEfficiency],
                 busy: dict[str, list[tuple[float, float]]]) -> None:
    """Add each finished span's self, busy and idle time to its agent."""
    children: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in traj.spans.values():
        if s.parent_span_id and s.end is not None:
            children[s.parent_span_id].append((s.start, s.end))
    for s in traj.spans.values():
        if s.end is None:
            continue
        a = _agent(per, s.agent_id)
        a.spans += 1
        a.role = a.role or s.role
        child_time = union_length([(max(s.start, c0), min(s.end, c1)) for c0, c1 in children.get(s.span_id, [])])
        self_time = max(0.0, s.duration - child_time)
        busy_time = union_length(busy.get(s.span_id, []))
        a.self_time_s += self_time
        a.busy_time_s += busy_time
        a.idle_time_s += max(0.0, self_time - busy_time)


def _order_agents(traj: Trajectory, per: dict[str, AgentEfficiency]) -> dict[str, AgentEfficiency]:
    """The agent records ordered by first span start, then id."""
    first = {aid: min((s.start for s in traj.spans.values() if s.agent_id == aid), default=float("inf"))
             for aid in per}
    return {aid: per[aid] for aid in sorted(per, key=lambda a: (first[a], a))}


def compute_efficiency(traj: Trajectory, pricing: PricingTable | None = None) -> EfficiencyMetrics:
    """Efficiency of one run; ``cost_usd`` is ``None`` when no call could be priced, never a silent zero,
    and ``cost_complete`` is False when any call is unpriced."""
    m = EfficiencyMetrics()
    per: dict[str, AgentEfficiency] = {}
    busy = _tally_events(traj, pricing, m, per)
    _span_timing(traj, per, busy)
    m.per_agent = _order_agents(traj, per)
    m.agent_time_s = sum(a.self_time_s for a in per.values())
    m.waiting_time_s = sum(a.idle_time_s for a in per.values())
    m.wall_clock_s = traj.duration
    return m
