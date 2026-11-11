"""Layer 3 -- swarm smells: recurring coordination patterns worth testing (Phase 2).

A smell is a *hypothesis* generated from one trajectory, never a verdict.
Each ``Diagnostic`` keeps three things apart:

* ``observation`` -- what was measured, with the numbers;
* ``hypothesis`` -- what the pattern may indicate ("may", "is consistent
  with"); the record alone cannot show cause and effect;
* ``intervention`` -- a concrete experiment that would test it (a Phase 3
  ablation such as "re-run without researcher_c and compare success and
  cost").

``inferred=True`` marks diagnostics whose evidence rests on edges inferred
from timing rather than recorded (Phase 1 data); their severity is capped.

Each detector's thresholds live in its own config dataclass (``ContextBloatConfig`` and so on), registered
with it.  ``SmellConfig`` maps each registration name to that detector's section.  The defaults are first
guesses, to be validated against Phase 3 ablations; override them with nested JSON,
``{"context_bloat": {"ratio": 2.0}}`` (``load_smell_config``, ``SmellConfig.from_dict``), where unknown sections
or fields are an error.

Ping-pong counts changes of direction in the longest back-and-forth of a
pair, whatever the message kinds.  The first message of an exchange makes
its sender the opener; the other side's replies always continue it.  The
opener turning back continues it only with evidence that it is answering:
a ``reply_to`` naming a message of the exchange, or a revision of an
artifact carried in it (same name with a new id, or ``derived_from`` it).
Turning back to a checker (a reviewer or critic) always continues it.
Otherwise it starts a new exchange, so a run of delegated tasks, each
answered once, is a series of round trips.  A plain-text loop with a
responder that is not a checker, no ``reply_to`` and no artifacts
therefore does not fire.

Premature synthesis is not flagged for an answer to the receiver's own output: after producing, the receiver
sent that agent its work or a request, or the late message replies to one of the receiver's later messages.

Custom smells: ``register_smell(name, detector, smells=None, config_cls=None)`` adds ``detector(ctx, section)`` to
the detectors run by ``detect_smells``, after those already registered, and appends the smells it emits (default
``(name,)``) to ``SMELLS``; registering a name again replaces its detector.  ``section`` is the detector's
``config_cls`` instance from section ``name`` of the SmellConfig, or None without a ``config_cls``.  The detector
receives a ``SmellContext`` with ``traj`` (the Trajectory), ``cm`` (its CoordinationMetrics), ``graph`` (the execution
graph or None), ``agents`` (agent ids in first-start order), ``roles`` (agent id to role), ``receive_cov`` and
``recorded`` (whether message receipts were fully recorded), ``order(agents)`` and
``add(smell, severity, agents, observation, hypothesis, intervention, evidence, inferred=False)``, which records a
Diagnostic for the run.

Classes:
- Diagnostic: One smell observed in one run (see the module docstring).
- SmellConfig: Detection thresholds, one section per registered detector (see the module docstring).
- RedundantDelegationConfig, AgentPingPongConfig, ...: One detector's thresholds (initial guesses).
- SmellContext: What a registered detector receives (see the module docstring).
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

from ..graph.builder import EPS
from ..metrics.dependency import REQUEST_KINDS
from ..metrics.redundancy import VERIFICATION_ROLES, is_verification, tool_key
from ..schema import MessageKind, Trajectory, to_text
from ..similarity import LexicalSimilarity

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.coordination import CoordinationMetrics

SEVERITIES = ("low", "medium", "high")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
SMELLS: list[str] = []
_CONFIG_CLASSES: dict[str, type] = {}
_CONTENT_KINDS = frozenset({MessageKind.RESULT.value, MessageKind.FEEDBACK.value, MessageKind.INFORMATION.value})


@dataclass
class Diagnostic:
    smell: str
    severity: str
    run_id: str
    agents: list[str]
    observation: str
    hypothesis: str
    intervention: str
    evidence: dict[str, Any] = field(default_factory=dict)
    inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The diagnostic as a JSON-ready dict."""
        return asdict(self)


@dataclass
class RedundantDelegationConfig:
    similarity: float = 0.8


@dataclass
class AgentPingPongConfig:
    min_alternations: int = 5
    high: int = 8


@dataclass
class OverAgentizationConfig:
    min_agents: int = 4
    low_contrib_frac: float = 0.34
    simple_task_max_agents: int = 2


@dataclass
class UnderSpecializationConfig:
    overlap: float = 0.5


@dataclass
class CoordinatorBottleneckConfig:
    share: float = 0.6
    min_peers: int = 3
    relay_frac: float = 0.5
    cp_share: float = 0.25
    high_share: float = 0.8


@dataclass
class ContextBloatConfig:
    ratio: float = 3.0
    min_excess: int = 2000
    high_ratio: float = 10.0


@dataclass
class DeadAgentConfig:
    min_coverage: float = 1.0
    high_token_share: float = 0.2


@dataclass
class SerializedExecutionConfig:
    ratio: float = 1.5
    high: float = 2.5
    min_gap_s: float = 1.0
    min_gap_frac: float = 0.2
    min_coverage: float = 0.5


@dataclass
class CommunicationOverheadConfig:
    ratio: float = 0.25
    high: float = 0.5
    min_messages: int = 3


def _reject_unknown_sections(names: Iterable[str]) -> None:
    """Raise ValueError naming any section that no registered detector configures."""
    unknown = sorted(k for k in names if k not in _CONFIG_CLASSES)
    if unknown:
        raise ValueError(f"unknown smell config section(s) {unknown}; known: {sorted(_CONFIG_CLASSES)}")


class SmellConfig(dict[str, Any]):
    def __init__(self, **sections: Any) -> None:
        _reject_unknown_sections(sections)
        for name, section in sections.items():
            if not isinstance(section, _CONFIG_CLASSES[name]):
                raise TypeError(f"smell config section {name!r}: expected {_CONFIG_CLASSES[name].__name__}, "
                                f"got {type(section).__name__}")
        super().__init__({name: cls() for name, cls in _CONFIG_CLASSES.items()})
        self.update(sections)

    def to_dict(self) -> dict[str, Any]:
        """The thresholds as a nested dict, one section per detector."""
        return {name: asdict(section) for name, section in self.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SmellConfig:
        """Defaults overridden by nested ``{section: {field: value}}``, rejecting unknown sections and fields."""
        _reject_unknown_sections(d)
        cfg = cls()
        for name, overrides in d.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"smell config section {name!r}: expected a JSON object, got {overrides!r}")
            known = {f.name for f in fields(_CONFIG_CLASSES[name])}
            bad = sorted(k for k in overrides if k not in known)
            if bad:
                raise ValueError(f"smell config section {name!r}: unknown field(s) {bad}; known: {sorted(known)}")
            cfg[name] = replace(cfg[name], **overrides)
        return cfg


def load_smell_config(path: str) -> SmellConfig:
    """Defaults overridden by the keys of a JSON object file; unknown keys
    are an error (a typo would otherwise silently keep the default)."""
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object of threshold overrides")
    return SmellConfig.from_dict(d)


def resolve_smell_config(spec: Any) -> SmellConfig:
    """``None`` → defaults; a ``SmellConfig``; a dict of overrides; a JSON path."""
    if spec is None:
        return SmellConfig()
    if isinstance(spec, SmellConfig):
        return spec
    if isinstance(spec, dict):
        return SmellConfig.from_dict(spec)
    if isinstance(spec, str):
        return load_smell_config(spec)
    raise TypeError(f"expected SmellConfig, dict or path, got {type(spec).__name__}")


def _pct(x: float) -> str:
    """A fraction as a whole percentage."""
    return f"{x * 100:.0f}%"


def _names(agents: Iterable[str]) -> str:
    """Agent names joined as 'a, b and c'."""
    a = list(agents)
    return a[0] if len(a) == 1 else ", ".join(a[:-1]) + " and " + a[-1]


def _cap(severity: str, cap: str) -> str:
    """``severity``, lowered to ``cap`` if it is more severe."""
    return severity if SEVERITY_RANK[severity] <= SEVERITY_RANK[cap] else cap


def _roles(traj: Trajectory) -> dict[str, str | None]:
    """Each agent's first recorded role."""
    roles: dict[str, str | None] = {}
    for s in sorted(traj.spans.values(), key=lambda s: s.start):
        roles[s.agent_id] = roles.get(s.agent_id) or s.role
    return roles


def _is_checker(traj: Trajectory, agent: str, roles: dict[str, str | None]) -> bool:
    """True if the agent verifies others' work, by role or by what it does."""
    return (roles.get(agent) or "").lower() in VERIFICATION_ROLES or is_verification(traj, agent, None)


class SmellContext:
    def __init__(self, traj: Trajectory, cm: CoordinationMetrics) -> None:
        self.traj, self.cm = traj, cm
        self.graph = cm.graph
        self.agents = list(self.graph.agents) if self.graph is not None else traj.agent_ids()
        self.rank = {a: i for i, a in enumerate(self.agents)}
        self.roles = _roles(traj)
        self.out: list[Diagnostic] = []
        cov = cm.coverage.message_receive_coverage
        self.receive_cov = cov
        self.recorded = cov is None or cov >= 1.0

    def order(self, agents: Iterable[str]) -> list[str]:
        """Agents in first-start order, then by id, without duplicates."""
        return sorted(set(agents), key=lambda a: (self.rank.get(a, len(self.rank)), a))

    def add(self, smell: str, severity: str, agents: Iterable[str], observation: str, hypothesis: str,
            intervention: str, evidence: dict[str, Any], inferred: bool = False) -> None:
        """Append a diagnostic for this run."""
        self.out.append(Diagnostic(smell, severity, self.traj.run_id, self.order(agents), observation, hypothesis,
                                   intervention, evidence, inferred))


def _redundant_delegation(c: SmellContext, section: RedundantDelegationConfig) -> None:
    """Flag similar requests reaching one agent from several senders, or delegated work repeated."""
    traj = c.traj
    seq = {ev.event_id: i for i, ev in enumerate(traj.events)}
    asks = sorted((m for m in traj.messages() if m.kind in REQUEST_KINDS and m.sender and m.receiver
                   and m.sender != m.receiver), key=lambda m: (m.timestamp, seq[m.event_id]))
    sim = LexicalSimilarity(threshold=section.similarity)

    def score(a: Any, b: Any) -> float:
        if a.content_hash and a.content_hash == b.content_hash:
            return 1.0
        return sim.scores([to_text(a.content), to_text(b.content)], [(0, 1)])[0]

    by_receiver: dict[str, list[Any]] = defaultdict(list)
    for m in asks:
        by_receiver[m.receiver].append(m)
    for receiver, ms in by_receiver.items():
        senders: set[str] = set()
        best = 0.0
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if a.sender != b.sender:
                    s = score(a, b)
                    if s >= section.similarity:
                        senders.update((a.sender, b.sender))
                        best = max(best, s)
        if len(senders) >= 2:
            names = c.order(senders)
            c.add("redundant_delegation", "medium", names + [receiver],
                  f"{receiver} received {_pct(best)}-similar requests from {len(names)} agents "
                  f"({_names(names)}).",
                  f"The architecture may be routing the same work to {receiver} more than once.",
                  f"Re-run with a single agent issuing this request to {receiver} and compare success and cost.",
                  {"receiver": receiver, "senders": names, "similarity": best})

    calls_after: dict[str, list[Any]] = defaultdict(list)
    for ev in traj.tool_calls():
        if ev.agent_id:
            calls_after[ev.agent_id].append(ev)
    by_sender: dict[str, dict[str, Any]] = defaultdict(dict)
    for m in asks:
        if not _is_checker(traj, m.receiver, c.roles):
            by_sender[m.sender].setdefault(m.receiver, m)
    for sender, first in by_sender.items():
        recv = c.order(first)
        keys = {r: {tool_key(ev) for ev in calls_after.get(r, []) if ev.timestamp >= first[r].timestamp - EPS
                    and not is_verification(traj, r, ev.span_id, ev.metadata)} for r in recv}
        involved: set[str] = set()
        shared_keys: set[tuple[str, str]] = set()
        best_sim = 0.0
        for i, a in enumerate(recv):
            for b in recv[i + 1:]:
                shared = keys[a] & keys[b]
                if shared:
                    involved.update((a, b))
                    shared_keys |= shared
                    best_sim = max(best_sim, score(first[a], first[b]))
        if not involved:
            continue
        workers = c.order(involved)
        tools = sorted({k[0] for k in shared_keys})
        sev = "high" if len(shared_keys) >= 3 and best_sim >= section.similarity else "medium"
        c.add("redundant_delegation", sev, [sender] + workers,
              f"{sender} delegated to {_names(workers)}, who then made {len(shared_keys)} identical tool "
              f"call(s) ({', '.join(tools)}); the most similar pair of their assignments was {_pct(best_sim)} "
              f"similar.",
              f"The delegation may be giving {_names(workers)} overlapping scopes, so the same lookups are "
              f"repeated.",
              f"Re-run with disjoint scopes for {_names(workers)} (or with one of them removed) and compare "
              f"success and cost.",
              {"sender": sender, "receivers": workers, "shared_tool_calls": len(shared_keys), "tools": tools,
               "assignment_similarity": best_sim})


def _ping_pong(c: SmellContext, section: AgentPingPongConfig) -> None:
    """Flag pairs whose longest back-and-forth exchange changes direction often enough;
    how an exchange starts, continues and restarts is described in the module overview."""
    arts = c.traj.artifacts
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    longest: dict[tuple[str, str], int] = defaultdict(int)
    ex: dict[tuple[str, str], dict[str, Any]] = {}

    def revises(aid: str, carried: set[str]) -> bool:
        a = arts.get(aid)
        if a is None or aid in carried:
            return False
        return (any(arts[x].name == a.name for x in carried if x in arts)
                or any(x in carried for x in a.derived_from or []))

    for m in sorted(c.traj.messages(), key=lambda e: e.timestamp):
        if not (m.sender and m.receiver and m.sender != m.receiver):
            continue
        a, b = c.order((m.sender, m.receiver))
        key = (a, b)
        senders, cur = pairs[key], ex.get(key)
        if cur is not None and senders[-1] != m.sender:
            if m.sender != cur["opener"] or _is_checker(c.traj, m.receiver, c.roles) or (
                    m.reply_to and m.reply_to in cur["ids"]) or any(revises(x, cur["arts"]) for x in m.input_artifacts):
                cur["run"] += 1
                longest[key] = max(longest[key], cur["run"])
            else:
                cur = None
        if cur is None:
            cur = ex[key] = {"opener": m.sender, "run": 0, "ids": set(), "arts": set()}
        cur["ids"].add(m.message_id)
        cur["arts"].update(m.input_artifacts)
        senders.append(m.sender)
    for (a, b), senders in pairs.items():
        alternations = longest[(a, b)]
        if alternations < section.min_alternations:
            continue
        sev = "high" if alternations >= section.high else "medium"
        c.add("agent_ping_pong", sev, [a, b],
              f"{a} and {b} exchanged {len(senders)} messages; their longest back-and-forth had {alternations} "
              f"changes of direction.",
              "Repeated back-and-forth may indicate repeated corrections or an unclear hand-off; a deliberate "
              "review loop looks the same in the record.",
              f"Re-run with the {a}–{b} exchange capped at one round (or with explicit acceptance criteria) "
              f"and compare success and cost.",
              {"pair": [a, b], "messages": len(senders), "alternations": alternations})


def _over_agentization(c: SmellContext, section: OverAgentizationConfig) -> None:
    """Flag many low-contribution agents, or too many agents for a simple task."""
    cm = c.cm
    agents = [a for a in c.agents if a in c.roles]
    n = len(agents)
    if n == 0:
        return
    low = [a for a in agents if a in cm.contribution and (cm.contribution[a].dead is True
                                                          or cm.contribution[a].status == "unused")]
    inferred = any(cm.contribution[a].dead is None for a in low)
    frac = len(low) / n
    many = n >= section.min_agents and low and frac >= section.low_contrib_frac
    simple = c.traj.task.difficulty == "simple" and n > section.simple_task_max_agents
    if not (many or simple):
        return
    facts = []
    if many:
        facts.append(f"{len(low)} of {n} agents ({_pct(frac)}) had no output observed in use "
                     f"({_names(low)})")
    if simple:
        facts.append(f"the task is marked simple and {n} agents ran (threshold {section.simple_task_max_agents})")
    sev = "medium" if many else "low"
    if many and simple:
        sev = "high"
    text = "; ".join(facts)
    c.add("over_agentization", _cap(sev, "low") if inferred else sev, low or agents,
          text[:1].upper() + text[1:] + ".",
          "The swarm may be using more agents than the task needs; extra agents are not bad in themselves "
          "if they add enough value for their cost.",
          f"Re-run with {_names(low) if low else 'fewer agents'} removed and compare success and cost.",
          {"agents": n, "low_value_agents": low, "low_value_fraction": frac,
           "difficulty": c.traj.task.difficulty}, inferred)


def _under_specialization(c: SmellContext, section: UnderSpecializationConfig) -> None:
    """Flag agent pairs with overlapping tools and similar outputs."""
    red = c.cm.redundancy
    if red is None:
        return
    overlap = {tuple(p["agents"]): p for p in red.pairwise_overlap}
    similar: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in red.redundant_artifact_pairs:
        if p.get("class") == "accidental" and p.get("producer") and p.get("matches_producer"):
            similar[tuple(sorted((p["producer"], p["matches_producer"])))].append(p)   # type: ignore[index]
    agents = [a for a in c.agents if c.roles.get(a)]
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            role = c.roles[a]
            if role != c.roles[b] or _is_checker(c.traj, a, c.roles) or _is_checker(c.traj, b, c.roles):
                continue
            key = tuple(sorted((a, b)))
            ov = overlap.get(key)          # type: ignore[arg-type]
            jac = ov["jaccard"] if ov else 0.0
            arts = similar.get(key, [])    # type: ignore[arg-type]
            tools_hit = jac >= section.overlap
            if not (tools_hit or arts):
                continue
            facts = []
            if tools_hit:
                facts.append(f"{_pct(jac)} of their distinct tool calls overlap (Jaccard, {ov['shared']} shared)")
            if arts:
                best = max(p["score"] for p in arts)
                facts.append(f"{len(arts)} of their artifacts match each other ({_pct(best)} similar)")
            c.add("under_specialization", "high" if tools_hit and arts else "medium", [a, b],
                  f"{a} and {b} share the role {role!r}; " + " and ".join(facts) + ".",
                  f"The two {role} agents may not have distinct responsibilities, which is consistent with "
                  f"overlapping scopes in their instructions.",
                  f"Re-run with disjoint scopes for {a} and {b} (or with one of them removed) and compare "
                  f"success and cost.",
                  {"role": role, "tool_jaccard": jac, "similar_artifacts": len(arts)})


def _bottleneck(c: SmellContext, section: CoordinatorBottleneckConfig) -> None:
    """Flag a hub agent that most messages pass through, worse when it relays or sits on the critical path."""
    comm, dep = c.cm.communication, c.cm.dependency
    if comm is None or comm.hub_agent is None or comm.hub_share is None or comm.hub_share < section.share:
        return
    hub = comm.hub_agent
    peers: set[str] = set()
    for m in c.traj.messages():
        if hub in (m.sender, m.receiver):
            peers.update(x for x in (m.sender, m.receiver) if x and x != hub)
    if len(peers) < section.min_peers:
        return
    relay = comm.per_agent[hub].relay_fraction if hub in comm.per_agent else None
    cp_share = None
    if dep is not None and dep.critical_path_s > EPS:
        cp_share = dep.critical_path.per_agent_s.get(hub, 0.0) / dep.critical_path_s
    relay_hit = relay is not None and relay >= section.relay_frac
    cp_hit = cp_share is not None and cp_share >= section.cp_share
    if not (relay_hit or cp_hit):
        return
    sev = "high" if comm.hub_share >= section.high_share and relay_hit and cp_hit else "medium"
    inferred = cp_hit and not relay_hit and dep is not None and dep.inferred_edges_on_path > 0
    facts = [f"{hub} took part in {_pct(comm.hub_share)} of {comm.messages} messages, with {len(peers)} peers"]
    if relay is not None:
        facts.append(f"{_pct(relay)} of the messages it sent passed on other agents' artifacts")
    if cp_share is not None:
        facts.append(f"its own calls are {_pct(cp_share)} of the critical path "
                     f"({dep.critical_path.per_agent_s.get(hub, 0.0):.1f}s of {dep.critical_path_s:.1f}s)")
    c.add("coordinator_bottleneck", _cap(sev, "low") if inferred else sev, [hub],
          "; ".join(facts) + ".",
          f"Routing work through {hub} may be lengthening the run and concentrating failure risk in one agent.",
          f"Re-run with direct hand-offs between {hub}'s peers (bypassing {hub} for relays) and compare "
          f"wall clock, success and cost.",
          {"hub": hub, "hub_share": comm.hub_share, "peers": c.order(peers), "relay_fraction": relay,
           "critical_path_share": cp_share}, inferred)


def _late_and_premature(c: SmellContext, section: None) -> None:
    """Flag agents that act before their inputs exist or receive inputs after their last call."""
    g = c.graph
    if g is None:
        return
    by_agent: dict[str, list[Any]] = defaultdict(list)
    for u in g.dag_units:
        if u.agent_id:
            by_agent[u.agent_id].append(u)
    produced: dict[str, list[float]] = defaultdict(list)
    for a in c.traj.artifacts.values():
        if a.producer:
            produced[a.producer].append(a.produced_at)
    finals = {c.traj.artifacts[a].producer for a in c.traj.output_artifacts if a in c.traj.artifacts}
    phase1 = g.schema_version is None
    shared: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_id = {ev.message_id: ev for ev in c.traj.messages() if ev.message_id}
    for ev in c.traj.messages():
        if ev.kind in REQUEST_KINDS or any(
                x in c.traj.artifacts and c.traj.artifacts[x].producer == ev.sender for x in ev.input_artifacts):
            shared[(ev.sender, ev.receiver)].append(ev.timestamp)
    for m in g.messages:
        r = m.receiver
        if not r or r == m.sender or not (m.kind in _CONTENT_KINDS or m.artifacts):
            continue
        sent_at = g.unit(m.send_node).start
        if m.receive_node is None:
            if not any(u.start < sent_at - EPS for u in by_agent.get(r, [])):
                continue
            if any(u.start >= sent_at - EPS for u in by_agent.get(r, [])):
                continue
            c.add("late_information", "low" if phase1 else "medium", [m.sender or "-", r],
                  f"A {m.kind} message from {m.sender} ({m.tokens} tokens) was sent to {r} at "
                  f"t={sent_at - (c.traj.started_at or 0.0):.1f}s, after {r}'s last recorded work.",
                  f"Information from {m.sender} may be arriving too late to be used by {r}.",
                  f"Re-run with {r} waiting for {m.sender}'s message (or with {m.sender} removed) and compare "
                  f"success and cost.",
                  {"message_id": m.message_id, "sender": m.sender, "receiver": r, "kind": m.kind}, phase1)
            continue
        got_at = g.unit(m.receive_node).start
        before = [t for t in produced.get(r, []) if t < got_at - EPS]
        after = [t for t in produced.get(r, []) if t >= got_at - EPS]
        if not before or after:
            continue
        q = by_id.get(getattr(by_id.get(m.message_id), "reply_to", None))
        if (q is not None and q.sender == r and q.timestamp >= min(before) - EPS) or any(
                min(before) - EPS <= t <= sent_at for t in shared.get((r, m.sender), [])):
            continue
        sev = "high" if r in finals else "medium"
        inferred = not m.received
        c.add("premature_synthesis", _cap(sev, "low") if inferred else sev, [r, m.sender or "-"],
              f"{r} produced {len(before)} artifact(s) before receiving a {m.kind} message from {m.sender} "
              f"and produced nothing after it" + (" (it produced the final output)." if r in finals else "."),
              f"{r} may be synthesizing before all of its inputs are available.",
              f"Re-run with {r} waiting for {m.sender} before producing, and compare success and cost.",
              {"message_id": m.message_id, "sender": m.sender, "receiver": r, "kind": m.kind,
               "final_producer": r in finals}, inferred)


def _context_bloat(c: SmellContext, section: ContextBloatConfig) -> None:
    """Flag agents whose LLM calls receive far more tokens than are relevant."""
    ctx = c.cm.context
    if ctx is None:
        return
    flagged: dict[str, list[Any]] = defaultdict(list)
    for call in ctx.calls:
        if call.ratio >= section.ratio and call.excess >= section.min_excess:
            flagged[call.agent_id or "-"].append(call)
    for agent in sorted(c.order(flagged), key=lambda a: -max(x.ratio for x in flagged[a])):
        calls = flagged[agent]
        worst = max(calls, key=lambda x: (x.ratio, x.excess))
        sev = "high" if worst.ratio >= section.high_ratio else "medium"
        bases = {x.basis for x in calls}
        inferred = bool(bases & {"task_only", "inferred"})
        if "task_only" in bases:
            sev = _cap(sev, "low")
        elif inferred:
            sev = _cap(sev, "medium")
        c.add("context_bloat", sev, [agent],
              f"{agent}'s largest prompt was {worst.received:,} tokens, {worst.ratio:.1f}× the "
              f"{worst.relevant:,} tokens traceable to its recorded inputs ({len(calls)} such call(s), "
              f"{sum(x.excess for x in calls):,} untraced tokens in total).",
              f"{agent} may be receiving more context than its task needs, or reading inputs that were not "
              f"recorded.",
              f"Re-run with {agent}'s prompt limited to its recorded inputs and compare success and cost.",
              {"calls": len(calls), "worst_event_id": worst.event_id, "received": worst.received,
               "relevant": worst.relevant, "ratio": worst.ratio, "excess": sum(x.excess for x in calls),
               "basis": sorted(bases)}, inferred)


def _dead_agents(c: SmellContext, section: DeadAgentConfig) -> None:
    """Flag agents whose outputs were never used, when provenance is complete enough to tell."""
    for a in c.agents:
        oc = c.cm.contribution.get(a)
        if oc is None or oc.dead is not True:
            continue
        if oc.provenance_coverage is None or oc.provenance_coverage < section.min_coverage:
            continue
        u = c.cm.utilization.get(a)
        share = u.token_share if u is not None else None
        sev = "high" if share is not None and share >= section.high_token_share else "medium"
        n_art = u.artifacts_produced if u is not None else 0
        n_msg = u.messages_sent if u is not None else 0
        share_txt = f"used {_pct(share)} of the run's tokens; " if share is not None else ""
        c.add("dead_agent", sev, [a],
              f"{a} {share_txt}none of its {n_art} artifact(s) or {n_msg} message(s) was observed in use by "
              f"another agent (provenance coverage {_pct(oc.provenance_coverage)}).",
              f"{a}'s work may not be reaching the result; removing it is consistent with the record, but only "
              f"an ablation can show its effect on success.",
              f"Re-run without {a} and compare success and cost.",
              {"token_share": share, "artifacts": n_art, "messages": n_msg, "status": oc.status,
               "provenance_coverage": oc.provenance_coverage})


def _serialized(c: SmellContext, section: SerializedExecutionConfig) -> None:
    """Flag runs whose busy wall time far exceeds the critical path."""
    dep = c.cm.dependency
    if dep is None or dep.critical_path_s <= EPS or dep.wall_s <= EPS:
        return
    busy_wall = dep.wall_s - (dep.unrecorded_s or 0.0)
    ratio = busy_wall / dep.critical_path_s
    gap = busy_wall - dep.critical_path_s
    if ratio < section.ratio or gap < max(section.min_gap_s, section.min_gap_frac * dep.wall_s):
        return
    sev = "high" if ratio >= section.high else "medium"
    weak = c.receive_cov is not None and c.receive_cov < section.min_coverage
    inferred = not c.recorded or dep.inferred_edges_on_path > 0
    if weak:
        sev = "low"
    cov_txt = "" if c.receive_cov is None else f"; {_pct(c.receive_cov)} of messages have a recorded receive"
    c.add("serialized_execution", sev, [],
          f"Recorded work spanned {busy_wall:.1f}s of wall clock but the longest dependency chain is "
          f"{dep.critical_path_s:.1f}s ({ratio:.1f}×, {gap:.1f}s gap; potential parallelism "
          f"{(dep.potential_parallelism or 0):.1f}× vs actual {(dep.actual_parallelism or 0):.1f}×){cov_txt}.",
          "Work that the recorded dependencies allow to overlap may be running one after another"
          + ("; with few recorded receives, dependencies may be missing from the graph." if weak else "."),
          "Re-run with the independent agents started concurrently and compare wall clock, success and cost.",
          {"busy_wall_s": busy_wall, "critical_path_s": dep.critical_path_s, "ratio": ratio, "gap_s": gap,
           "unrecorded_s": dep.unrecorded_s, "message_receive_coverage": c.receive_cov}, inferred or weak)


def _comm_overhead(c: SmellContext, section: CommunicationOverheadConfig) -> None:
    """Flag runs whose message bodies are large relative to LLM tokens."""
    comm = c.cm.communication
    if comm is None or comm.comm_work_ratio is None or comm.messages < section.min_messages:
        return
    r = comm.comm_work_ratio
    if r < section.ratio:
        return
    cats = ", ".join(f"{k} {v['body_tokens']:,}" for k, v in comm.by_category.items() if v.get("messages"))
    top = max(comm.per_agent.values(), key=lambda a: a.body_tokens_sent, default=None)
    c.add("communication_overhead", "high" if r >= section.high else "medium",
          [top.agent_id] if top is not None and top.body_tokens_sent else [],
          f"Message bodies came to {comm.body_tokens:,} tokens, {r:.2f}× the {comm.llm_tokens:,} LLM tokens "
          f"(Communication / Work) over {comm.messages} messages ({cats}).",
          "Communication is not automatically waste, but this much talking relative to working may indicate "
          "restated content or chatty coordination.",
          "Re-run with shorter hand-offs (pass artifact references instead of restating content) and compare "
          "success and cost.",
          {"comm_work_ratio": r, "messages": comm.messages, "body_tokens": comm.body_tokens,
           "by_category": comm.by_category})


_REGISTRY: dict[str, Callable[[SmellContext, Any], None]] = {}


def register_smell(name: str, detector: Callable[[SmellContext, Any], None], smells: tuple[str, ...] | None = None,
                   config_cls: type | None = None) -> None:
    """Run ``detector(ctx, section)`` in ``detect_smells`` under ``name`` (replacing one registered there), with its
    ``config_cls`` thresholds as section ``name`` of ``SmellConfig``; append its smells to ``SMELLS``."""
    _REGISTRY[name] = detector
    _CONFIG_CLASSES.pop(name, None)
    if config_cls is not None:
        _CONFIG_CLASSES[name] = config_cls
    SMELLS.extend(s for s in dict.fromkeys(smells or (name,)) if s not in SMELLS)


register_smell("redundant_delegation", _redundant_delegation, config_cls=RedundantDelegationConfig)
register_smell("agent_ping_pong", _ping_pong, config_cls=AgentPingPongConfig)
register_smell("over_agentization", _over_agentization, config_cls=OverAgentizationConfig)
register_smell("under_specialization", _under_specialization, config_cls=UnderSpecializationConfig)
register_smell("coordinator_bottleneck", _bottleneck, config_cls=CoordinatorBottleneckConfig)
register_smell("late_and_premature", _late_and_premature, smells=("late_information", "premature_synthesis"))
register_smell("context_bloat", _context_bloat, config_cls=ContextBloatConfig)
register_smell("dead_agent", _dead_agents, config_cls=DeadAgentConfig)
register_smell("serialized_execution", _serialized, config_cls=SerializedExecutionConfig)
register_smell("communication_overhead", _comm_overhead, config_cls=CommunicationOverheadConfig)


def detect_smells(traj: Trajectory, metrics: CoordinationMetrics,
                  config: SmellConfig | None = None) -> list[Diagnostic]:
    """All smells found in one run, in ``SMELLS`` order then by severity."""
    cfg = resolve_smell_config(config)
    c = SmellContext(traj, metrics)
    for name, detect in _REGISTRY.items():
        cls = _CONFIG_CLASSES.get(name)
        detect(c, cfg[name] if name in cfg else cls() if cls is not None else None)
    order = {s: i for i, s in enumerate(SMELLS)}
    return sorted(c.out, key=lambda d: (order.get(d.smell, len(order)), -SEVERITY_RANK[d.severity]))
