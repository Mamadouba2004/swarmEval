"""Layer 3 -- context bloat: how much of what each LLM call read was traceable (Phase 2).

For every LLM call:

* **received** = the call's prompt tokens (input + cache read + cache write);
* **relevant** = what the record shows the call had reason to read:
  the task input + the distinct artifacts the call took as input or its span
  consumed before it + the bodies of messages its agent received before it,
  in any span or none (``content_tokens`` minus attached artifacts, which
  are counted as artifacts) + earlier outputs of the same span (LLM outputs, tool results)
  + ``metadata["system_tokens"]`` when the caller recorded its system prompt.
* **excess** = max(0, received − relevant); **ratio** = received ÷ max(relevant, 1).

"Relevant" is an upper bound on what was needed and a lower bound on what was
legitimately there: anything the swarm did not record (retrieved documents,
long-term memory, an unrecorded system prompt) shows up as excess.  So a
high ratio says "much of this prompt cannot be traced to recorded inputs",
not "this context was useless".  ``basis`` says what the relevant figure rests
on: ``"referenced"`` when the call had recorded inputs beyond the task,
``"task_only"`` when nothing was, ``"inferred"`` when message bodies were
attributed by timing (for a message with no receive event -- all of them on
Phase 1 data, or one read with ``inbox()`` -- the call counts it if it was
addressed to the agent and sent before the call).

Classes:
- CallContext: Context accounting of one LLM call (definitions in the module docstring).
- ContextMetrics: Context accounting of one run. ``ratio`` = Σ received ÷ Σ relevant over all LLM calls (``None``
  without LLM calls); ``worst`` is the call with the highest ratio (ties: larger excess, then earliest). ``basis``
  is ``"task_only"`` when no call had any recorded input beyond the task, ``"inferred"`` when any call relied on
  timing-attributed messages, else ``"referenced"``.
- ContextSummary: Context bloat over runs. ``worst_agents`` counts how often each agent made a run's worst call.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph.builder import EPS
from ..schema import Event, EventType, Trajectory, estimate_tokens
from ..stats import Summary, summarize
from .communication import body_tokens


@dataclass
class CallContext:
    event_id: str
    agent_id: str | None
    span_id: str | None
    received: int
    relevant: int
    excess: int
    ratio: float
    basis: str
    parts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The record as a JSON-ready dict."""
        return asdict(self)


@dataclass
class ContextMetrics:
    llm_calls: int = 0
    received: int = 0
    relevant: int = 0
    excess: int = 0
    ratio: float | None = None
    basis: str | None = None
    worst: CallContext | None = None
    calls: list[CallContext] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The metrics as a JSON-ready dict."""
        d = asdict(self)
        d["worst"] = self.worst.to_dict() if self.worst else None
        return d


def compute_context(traj: Trajectory) -> ContextMetrics:
    """Received versus relevant tokens for every LLM call of one run."""
    seq = {ev.event_id: i for i, ev in enumerate(traj.events)}
    events = sorted(traj.events, key=lambda e: (round(e.timestamp, 9), seq[e.event_id]))
    by_span: dict[str, list[Event]] = defaultdict(list)
    reads: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        if ev.span_id:
            by_span[ev.span_id].append(ev)
        if ev.event_type == EventType.MESSAGE_RECEIVE.value and ev.agent_id:
            reads[ev.agent_id].append(ev)
    sent = {ev.message_id: ev for ev in traj.messages() if ev.message_id}
    received_ids = {ev.message_id for ev in traj.events_of(EventType.MESSAGE_RECEIVE)}
    to_agent: dict[str, list[Event]] = defaultdict(list)
    for ev in traj.messages():
        if ev.receiver and not (ev.message_id and ev.message_id in received_ids):
            to_agent[ev.receiver].append(ev)
    task_tokens = estimate_tokens(traj.task.input)
    arts = traj.artifacts

    out = ContextMetrics()
    for call in events:
        if call.event_type != EventType.LLM_CALL.value:
            continue
        t0 = call.timestamp
        art_ids: set[str] = set(call.input_artifacts)
        msg_ids: set[str] = set()
        prior_out, inferred = 0, False
        for ev in by_span.get(call.span_id or "", []):
            if ev.event_id == call.event_id or ev.timestamp > t0 + EPS:
                continue
            et = ev.event_type
            if et in (EventType.ARTIFACT_USE.value, EventType.LLM_CALL.value, EventType.TOOL_CALL.value):
                art_ids.update(ev.input_artifacts)
            if et in (EventType.LLM_CALL.value, EventType.TOOL_CALL.value) and ev.end_timestamp <= t0 + EPS:
                prior_out += ev.output_tokens if et == EventType.LLM_CALL.value else ev.content_tokens
        msg_ids.update(ev.message_id or "" for ev in reads.get(call.agent_id or "", [])
                       if ev.timestamp <= t0 + EPS)
        msg_body = sum(body_tokens(sent[i], traj) for i in msg_ids if i in sent)
        if call.agent_id:
            msgs = [m for m in to_agent.get(call.agent_id, []) if m.timestamp <= t0 + EPS]
            if msgs:
                msg_body += sum(body_tokens(m, traj) for m in msgs)
                inferred = True
        art_tokens = sum(arts[a].tokens for a in art_ids if a in arts)
        system = call.metadata.get("system_tokens")
        system = int(system) if isinstance(system, (int, float)) else 0
        parts = {"task": task_tokens, "artifacts": art_tokens, "messages": msg_body, "prior_outputs": prior_out,
                 "system": system}
        relevant = sum(parts.values())
        received = call.prompt_tokens
        referenced = art_tokens + msg_body + prior_out + system > 0
        basis = "inferred" if inferred else ("referenced" if referenced else "task_only")
        cc = CallContext(call.event_id, call.agent_id, call.span_id, received, relevant,
                         max(0, received - relevant), received / max(relevant, 1), basis, parts)
        out.calls.append(cc)
    if out.calls:
        out.llm_calls = len(out.calls)
        out.received = sum(c.received for c in out.calls)
        out.relevant = sum(c.relevant for c in out.calls)
        out.excess = sum(c.excess for c in out.calls)
        out.ratio = out.received / max(out.relevant, 1)
        bases = {c.basis for c in out.calls}
        out.basis = "inferred" if "inferred" in bases else ("referenced" if "referenced" in bases else "task_only")
        out.worst = max(enumerate(out.calls), key=lambda ic: (ic[1].ratio, ic[1].excess, -ic[0]))[1]
    return out


@dataclass
class ContextSummary:
    n_runs: int
    ratio: Summary | None
    excess: Summary
    worst_ratio: Summary | None
    worst_agents: dict[str, int] = field(default_factory=dict)
    bases: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dict."""
        return {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}


def summarize_context(per_run: Sequence[ContextMetrics]) -> ContextSummary:
    """Context bloat over runs."""
    ratios = [r.ratio for r in per_run if r.ratio is not None]
    worst = [r.worst for r in per_run if r.worst is not None]
    agents: dict[str, int] = defaultdict(int)
    for w in worst:
        agents[w.agent_id or "-"] += 1
    bases: dict[str, int] = defaultdict(int)
    for r in per_run:
        if r.basis:
            bases[r.basis] += 1
    return ContextSummary(n_runs=len(per_run), ratio=summarize(ratios) if ratios else None,
                          excess=summarize([r.excess for r in per_run]),
                          worst_ratio=summarize([w.ratio for w in worst]) if worst else None,
                          worst_agents=dict(agents), bases=dict(bases))
