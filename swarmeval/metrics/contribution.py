"""Layer 3 -- agent utilization and *observed* contribution (Phase 2).

Utilization is how much of the run's activity an agent accounts for.
Observed contribution is what the record shows about where its outputs went:
were its artifacts consumed by other agents, did they reach the final
output's lineage, were its messages used?  Neither is a causal or marginal
contribution -- that needs ablation (Phase 3) -- so nothing here is reported
as "agent X contributed N%".

A negative finding ("nobody used X's output") is only as good as the
recording of use.  Each agent therefore gets a ``provenance_coverage``: the
share of its outputs whose possible consumers record what they use.  ``dead``
is True only when nothing was used *and* that coverage is complete; with
partial coverage it is ``None`` (unknown), never True.

Classes:
- AgentUtilization: An agent's share of the run's activity. Shares are ``None`` when the run had none of that
  activity at all. ``busy_share`` is the agent's time inside its own LLM/tool calls over all agents' such time.
  ``downstream_refs`` counts distinct (artifact, consuming agent) pairs for the agent's artifacts, other agents
  only.
- Lineage: Artifacts the final output was (transitively) derived from. ``inferred`` holds those reached only through
  artifacts without recorded ``derived_from``, whose inputs were approximated by what the producing span consumed
  before producing them.
- ObservedContribution: Where an agent's outputs were observed to go (provenance, not marginal effect). ``status``
  is the strongest of, in order: * ``in_final_lineage`` -- produced an artifact the final output derives from; *
  ``consumed_downstream`` -- another agent consumed one of its artifacts; * ``delivered_only`` -- its
  messages/artifacts reached other agents, none of its artifacts was consumed; * ``unused`` -- it had outputs but
  none reached another agent; * ``unknown`` -- it sent no message and produced no artifact, so there is nothing to
  trace. ``dead`` is True only when nothing it sent was used, provenance coverage is complete and it did not produce
  the final output; False when some use was observed; ``None`` when that cannot be told.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph.builder import EPS, ExecutionGraph, build_graph
from ..schema import EventType, Trajectory
from ..stats import mean, summarize
from .efficiency import EfficiencyMetrics, compute_efficiency

STATUSES = ("in_final_lineage", "consumed_downstream", "delivered_only", "unused", "unknown")


@dataclass
class AgentUtilization:
    agent_id: str
    role: str | None = None
    token_share: float | None = None
    llm_call_share: float | None = None
    tool_call_share: float | None = None
    busy_share: float | None = None
    messages_sent: int = 0
    artifacts_produced: int = 0
    downstream_refs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-ready dict."""
        return asdict(self)


def _share(part: float, total: float) -> float | None:
    """``part`` over ``total``, None when the total is zero."""
    return part / total if total else None


def compute_utilization(traj: Trajectory, graph: ExecutionGraph | None = None,
                        efficiency: EfficiencyMetrics | None = None) -> dict[str, AgentUtilization]:
    """Per-agent utilization, built on the efficiency layer's per-agent
    counts and busy time (``union_length`` of each span's calls)."""
    graph = graph or build_graph(traj)
    eff = efficiency or compute_efficiency(traj)
    per = eff.per_agent
    tokens = sum(a.total_tokens for a in per.values())
    llm = sum(a.llm_calls for a in per.values())
    tools = sum(a.tool_calls for a in per.values())
    busy = sum(a.busy_time_s for a in per.values())
    refs: dict[str, int] = defaultdict(int)
    for art_id, consumers in graph.artifact_consumers.items():
        art = traj.artifacts.get(art_id)
        if art is not None and art.producer is not None:
            refs[art.producer] += sum(1 for c in consumers if c != art.producer)
    out: dict[str, AgentUtilization] = {}
    for aid in graph.agents:
        a = per.get(aid)
        if a is None:
            out[aid] = AgentUtilization(aid, downstream_refs=refs.get(aid, 0))
            continue
        out[aid] = AgentUtilization(aid, a.role, _share(a.total_tokens, tokens), _share(a.llm_calls, llm),
                                    _share(a.tool_calls, tools), _share(a.busy_time_s, busy), a.messages_sent,
                                    a.artifacts_produced, refs.get(aid, 0))
    return out


@dataclass
class Lineage:
    artifacts: list[str] = field(default_factory=list)
    inferred: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The lineage as a JSON-ready dict."""
        return asdict(self)


def _span_consumption(traj: Trajectory) -> dict[str, list[tuple[float, str]]]:
    """span id -> (time, artifact id) for everything the span genuinely used."""
    used: dict[str, list[tuple[float, str]]] = defaultdict(list)
    kinds = (EventType.ARTIFACT_USE.value, EventType.LLM_CALL.value, EventType.TOOL_CALL.value,
             EventType.ARTIFACT.value)
    for ev in traj.events:
        if ev.event_type in kinds and ev.span_id:
            for a in ev.input_artifacts:
                used[ev.span_id].append((ev.timestamp, a))
    return used


def final_lineage(traj: Trajectory) -> Lineage:
    """Walk back from ``traj.output_artifacts`` over ``derived_from``, falling back to what an artifact's
    span consumed before producing it; artifacts reached that way are marked inferred."""
    arts = traj.artifacts
    span_used: dict[str, list[tuple[float, str]]] | None = None

    def sources(a: str) -> tuple[list[str], bool]:
        nonlocal span_used
        art = arts[a]
        if art.derived_from is not None:
            return [x for x in art.derived_from if x in arts], False
        if span_used is None:
            span_used = _span_consumption(traj)
        found = [x for t, x in span_used.get(art.span_id or "", [])
                 if t <= art.produced_at + EPS and x != a and x in arts]
        return list(dict.fromkeys(found)), True

    roots = [a for a in dict.fromkeys(traj.output_artifacts) if a in arts]

    def walk(follow_inferred: bool) -> list[str]:
        seen: dict[str, None] = {}
        stack = list(roots)
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen[a] = None
            srcs, inferred = sources(a)
            if inferred and not follow_inferred:
                continue
            stack.extend(s for s in srcs if s not in seen)
        return list(seen)

    recorded = set(walk(False))
    every = walk(True)
    return Lineage(every, [a for a in every if a not in recorded])


@dataclass
class ObservedContribution:
    agent_id: str
    status: str
    artifacts_consumed_by_others: int = 0
    consumers: list[str] = field(default_factory=list)
    messages_used: int = 0
    in_final_lineage: bool = False
    lineage_inferred: bool = False
    provenance_coverage: float | None = None
    dead: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-ready dict."""
        return asdict(self)


def _artifact_receivers(graph: ExecutionGraph) -> dict[str, set[str]]:
    """artifact id -> agents it was sent to in a message."""
    receivers: dict[str, set[str]] = defaultdict(set)
    for m in graph.messages:
        for a in m.artifacts:
            if m.receiver is not None:
                receivers[a].add(m.receiver)
    return receivers


def _provenance_coverage(traj: Trajectory, graph: ExecutionGraph, agent: str, own: list[str],
                         receivers: dict[str, set[str]]) -> float | None:
    """Share of the agent's outputs whose possible consumers record what they use (an artifact's message
    receivers, else other agents active after it); an artifact nobody else could use adds no check."""
    instrumented = graph.coverage.consumers_instrumented
    checks: list[bool] = []
    for a in own:
        art = traj.artifacts[a]
        possible = receivers.get(a, set()) - {agent}
        if not possible:
            possible = {s.agent_id for s in traj.spans.values()
                        if s.agent_id != agent and (s.end is None or s.end >= art.produced_at - EPS)}
        if possible:
            checks.append(all(instrumented.get(c, False) for c in possible))
    for m in graph.messages:
        if m.sender == agent and m.artifacts and m.receiver is not None and m.receiver != agent:
            checks.append(instrumented.get(m.receiver, False))
    return sum(checks) / len(checks) if checks else None


def compute_contribution(traj: Trajectory, graph: ExecutionGraph | None = None,
                         lineage: Lineage | None = None) -> dict[str, ObservedContribution]:
    """Where each agent's outputs were observed to go in one run."""
    graph = graph or build_graph(traj)
    lineage = lineage or final_lineage(traj)
    produced: dict[str, list[str]] = defaultdict(list)
    for a in traj.artifacts.values():
        if a.producer is not None:
            produced[a.producer].append(a.artifact_id)
    in_lineage = set(lineage.artifacts)
    inferred = set(lineage.inferred)
    final_producers = {traj.artifacts[a].producer for a in traj.output_artifacts if a in traj.artifacts}
    receivers = _artifact_receivers(graph)

    out: dict[str, ObservedContribution] = {}
    for aid in graph.agents:
        own = produced.get(aid, [])
        consumers = sorted({c for a in own for c in graph.artifact_consumers.get(a, []) if c != aid},
                           key=graph.agents.index)
        consumed = sum(1 for a in own if any(c != aid for c in graph.artifact_consumers.get(a, [])))
        sent = [m for m in graph.messages if m.sender == aid and m.receiver not in (None, aid)]
        used_msgs = sum(1 for m in sent if m.used)
        delivered = any(m.receive_node is not None for m in sent)
        lin = [a for a in own if a in in_lineage]
        if lin:
            status = "in_final_lineage"
        elif consumed:
            status = "consumed_downstream"
        elif delivered:
            status = "delivered_only"
        elif own or sent:
            status = "unused"
        else:
            status = "unknown"
        coverage = _provenance_coverage(traj, graph, aid, own, receivers)
        dead: bool | None
        if aid in final_producers or consumed or used_msgs:
            dead = False
        elif status in ("unused", "delivered_only") and coverage is not None and coverage >= 1.0:
            dead = True
        else:
            dead = None
        out[aid] = ObservedContribution(aid, status, consumed, consumers, used_msgs, bool(lin),
                                        bool(lin) and all(a in inferred for a in lin), coverage, dead)
    return out


def summarize_contribution(per_run: Sequence[dict[str, ObservedContribution]]) -> dict[str, dict[str, Any]]:
    """Per agent over runs: how often it was in the final lineage, had an artifact consumed downstream,
    and was dead (over ``dead_known_runs``), plus its provenance coverage."""
    by_agent: dict[str, list[ObservedContribution]] = defaultdict(list)
    for run in per_run:
        for aid, c in run.items():
            by_agent[aid].append(c)
    out: dict[str, dict[str, Any]] = {}
    for aid, cs in by_agent.items():
        known = [c.dead for c in cs if c.dead is not None]
        cov = [c.provenance_coverage for c in cs if c.provenance_coverage is not None]
        counts = {s: sum(1 for c in cs if c.status == s) for s in STATUSES}
        out[aid] = {
            "runs": len(cs),
            "lineage_rate": mean([1.0 if c.in_final_lineage else 0.0 for c in cs]),
            "consumed_rate": mean([1.0 if c.artifacts_consumed_by_others else 0.0 for c in cs]),
            "dead_rate": mean([1.0 if d else 0.0 for d in known]) if known else None,
            "dead_known_runs": len(known),
            "provenance_coverage": summarize(cov).to_dict() if cov else None,
            "status_counts": {s: n for s, n in counts.items() if n},
        }
    return out
