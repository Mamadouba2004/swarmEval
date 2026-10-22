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

Thresholds live in ``SmellConfig``.  The defaults are first guesses, to be
validated against Phase 3 ablations; override them with a JSON file
(``load_smell_config``) or ``SmellConfig(...)``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from ..graph.builder import EPS
from ..metrics.dependency import REQUEST_KINDS
from ..metrics.redundancy import VERIFICATION_ROLES, is_verification, tool_key
from ..schema import MessageKind, Trajectory, to_text
from ..similarity import LexicalSimilarity

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.coordination import CoordinationMetrics

SEVERITIES = ("low", "medium", "high")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
SMELLS = ("redundant_delegation", "agent_ping_pong", "over_agentization", "under_specialization",
          "coordinator_bottleneck", "late_information", "premature_synthesis", "context_bloat", "dead_agent",
          "serialized_execution", "communication_overhead")
# messages that carry information the receiver might need (not just control)
_CONTENT_KINDS = frozenset({MessageKind.RESULT.value, MessageKind.FEEDBACK.value, MessageKind.INFORMATION.value})


@dataclass
class Diagnostic:
    """One smell observed in one run (see the module docstring)."""
    smell: str
    severity: str                   # low | medium | high
    run_id: str
    agents: List[str]
    observation: str
    hypothesis: str
    intervention: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    inferred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SmellConfig:
    """Detection thresholds (initial guesses; see the module docstring)."""
    delegation_similarity: float = 0.8
    pingpong_min_alternations: int = 5
    pingpong_high: int = 8
    overagent_min_agents: int = 4
    overagent_low_contrib_frac: float = 0.34
    simple_task_max_agents: int = 2
    underspec_overlap: float = 0.5
    hub_share: float = 0.6
    hub_min_peers: int = 3
    hub_relay_frac: float = 0.5
    hub_cp_share: float = 0.25
    hub_high_share: float = 0.8
    bloat_ratio: float = 3.0
    bloat_min_excess: int = 2000
    bloat_high_ratio: float = 10.0
    dead_min_coverage: float = 1.0
    dead_high_token_share: float = 0.2
    serial_ratio: float = 1.5
    serial_high: float = 2.5
    serial_min_gap_s: float = 1.0
    serial_min_gap_frac: float = 0.2
    serial_min_coverage: float = 0.5
    comm_ratio: float = 0.25
    comm_high: float = 0.5
    comm_min_messages: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmellConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in d if k not in known)
        if unknown:
            raise ValueError(f"unknown smell config key(s) {unknown}; known: {sorted(known)}")
        return cls(**d)


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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _names(agents: Iterable[str]) -> str:
    a = list(agents)
    return a[0] if len(a) == 1 else ", ".join(a[:-1]) + " and " + a[-1]


def _cap(severity: str, cap: str) -> str:
    return severity if SEVERITY_RANK[severity] <= SEVERITY_RANK[cap] else cap


def _roles(traj: Trajectory) -> Dict[str, Optional[str]]:
    roles: Dict[str, Optional[str]] = {}
    for s in sorted(traj.spans.values(), key=lambda s: s.start):
        roles[s.agent_id] = roles.get(s.agent_id) or s.role
    return roles


def _is_checker(traj: Trajectory, agent: str, roles: Dict[str, Optional[str]]) -> bool:
    return (roles.get(agent) or "").lower() in VERIFICATION_ROLES or is_verification(traj, agent, None)


class _Ctx:
    def __init__(self, traj: Trajectory, cm: "CoordinationMetrics", cfg: SmellConfig) -> None:
        self.traj, self.cm, self.cfg = traj, cm, cfg
        self.graph = cm.graph
        self.agents = list(self.graph.agents) if self.graph is not None else traj.agent_ids()
        self.rank = {a: i for i, a in enumerate(self.agents)}
        self.roles = _roles(traj)
        self.out: List[Diagnostic] = []
        cov = cm.coverage.message_receive_coverage
        self.receive_cov = cov            # None = no messages at all
        self.recorded = cov is None or cov >= 1.0

    def order(self, agents: Iterable[str]) -> List[str]:
        return sorted(set(agents), key=lambda a: (self.rank.get(a, len(self.rank)), a))

    def add(self, smell: str, severity: str, agents: Iterable[str], observation: str, hypothesis: str,
            intervention: str, evidence: Dict[str, Any], inferred: bool = False) -> None:
        self.out.append(Diagnostic(smell, severity, self.traj.run_id, self.order(agents), observation, hypothesis,
                                   intervention, evidence, inferred))


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
def _redundant_delegation(c: _Ctx) -> None:
    traj, cfg = c.traj, c.cfg
    seq = {ev.event_id: i for i, ev in enumerate(traj.events)}
    asks = sorted((m for m in traj.messages() if m.kind in REQUEST_KINDS and m.sender and m.receiver
                   and m.sender != m.receiver), key=lambda m: (m.timestamp, seq[m.event_id]))
    sim = LexicalSimilarity(threshold=cfg.delegation_similarity)

    def score(a: Any, b: Any) -> float:
        if a.content_hash and a.content_hash == b.content_hash:
            return 1.0
        return sim.scores([to_text(a.content), to_text(b.content)], [(0, 1)])[0]

    # (a) similar requests reaching one receiver from two or more senders
    by_receiver: Dict[str, List[Any]] = defaultdict(list)
    for m in asks:
        by_receiver[m.receiver].append(m)
    for receiver, ms in by_receiver.items():
        senders: Set[str] = set()
        best = 0.0
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if a.sender != b.sender:
                    s = score(a, b)
                    if s >= cfg.delegation_similarity:
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

    # (b) one sender delegating to several workers who then repeat each other's tool calls
    calls_after: Dict[str, List[Any]] = defaultdict(list)
    for ev in traj.tool_calls():
        if ev.agent_id:
            calls_after[ev.agent_id].append(ev)
    by_sender: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for m in asks:
        if not _is_checker(traj, m.receiver, c.roles):
            by_sender[m.sender].setdefault(m.receiver, m)   # first delegation to each receiver
    for sender, first in by_sender.items():
        recv = c.order(first)
        keys = {r: {tool_key(ev) for ev in calls_after.get(r, []) if ev.timestamp >= first[r].timestamp - EPS
                    and not is_verification(traj, r, ev.span_id, ev.metadata)} for r in recv}
        involved: Set[str] = set()
        shared_keys: Set[Tuple[str, str]] = set()
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
        sev = "high" if len(shared_keys) >= 3 and best_sim >= cfg.delegation_similarity else "medium"
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


def _ping_pong(c: _Ctx) -> None:
    cfg = c.cfg
    seq = {ev.event_id: i for i, ev in enumerate(c.traj.events)}
    pairs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for m in sorted(c.traj.messages(), key=lambda m: (m.timestamp, seq[m.event_id])):
        if m.sender and m.receiver and m.sender != m.receiver:
            a, b = c.order((m.sender, m.receiver))
            pairs[(a, b)].append(m.sender)
    for (a, b), senders in pairs.items():
        alternations = sum(1 for x, y in zip(senders, senders[1:]) if x != y)
        if alternations < cfg.pingpong_min_alternations:
            continue
        sev = "high" if alternations >= cfg.pingpong_high else "medium"
        c.add("agent_ping_pong", sev, [a, b],
              f"{a} and {b} exchanged {len(senders)} messages with {alternations} changes of direction.",
              "Repeated back-and-forth may indicate repeated corrections or an unclear hand-off; a deliberate "
              "review loop looks the same in the record.",
              f"Re-run with the {a}–{b} exchange capped at one round (or with explicit acceptance criteria) "
              f"and compare success and cost.",
              {"pair": [a, b], "messages": len(senders), "alternations": alternations})


def _over_agentization(c: _Ctx) -> None:
    cfg, cm = c.cfg, c.cm
    agents = [a for a in c.agents if a in c.roles]     # agents that actually ran
    n = len(agents)
    if n == 0:
        return
    low = [a for a in agents if a in cm.contribution and (cm.contribution[a].dead is True
                                                          or cm.contribution[a].status == "unused")]
    inferred = any(cm.contribution[a].dead is None for a in low)
    frac = len(low) / n
    many = n >= cfg.overagent_min_agents and low and frac >= cfg.overagent_low_contrib_frac
    simple = c.traj.task.difficulty == "simple" and n > cfg.simple_task_max_agents
    if not (many or simple):
        return
    facts = []
    if many:
        facts.append(f"{len(low)} of {n} agents ({_pct(frac)}) had no output observed in use "
                     f"({_names(low)})")
    if simple:
        facts.append(f"the task is marked simple and {n} agents ran (threshold {cfg.simple_task_max_agents})")
    sev = "medium" if many else "low"
    if many and simple:
        sev = "high"
    c.add("over_agentization", _cap(sev, "low") if inferred else sev, low or agents,
          "; ".join(facts).capitalize() + ".",
          "The swarm may be using more agents than the task needs; extra agents are not bad in themselves "
          "if they add enough value for their cost.",
          f"Re-run with {_names(low) if low else 'fewer agents'} removed and compare success and cost.",
          {"agents": n, "low_value_agents": low, "low_value_fraction": frac,
           "difficulty": c.traj.task.difficulty}, inferred)


def _under_specialization(c: _Ctx) -> None:
    red = c.cm.redundancy
    if red is None:
        return
    overlap = {tuple(p["agents"]): p for p in red.pairwise_overlap}
    similar: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
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
            tools_hit = jac >= c.cfg.underspec_overlap
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


def _bottleneck(c: _Ctx) -> None:
    cfg, comm, dep = c.cfg, c.cm.communication, c.cm.dependency
    if comm is None or comm.hub_agent is None or comm.hub_share is None or comm.hub_share < cfg.hub_share:
        return
    hub = comm.hub_agent
    peers: Set[str] = set()
    for m in c.traj.messages():
        if hub in (m.sender, m.receiver):
            peers.update(x for x in (m.sender, m.receiver) if x and x != hub)
    if len(peers) < cfg.hub_min_peers:
        return
    relay = comm.per_agent[hub].relay_fraction if hub in comm.per_agent else None
    cp_share = None
    if dep is not None and dep.critical_path_s > EPS:
        cp_share = dep.critical_path.per_agent_s.get(hub, 0.0) / dep.critical_path_s
    relay_hit = relay is not None and relay >= cfg.hub_relay_frac
    cp_hit = cp_share is not None and cp_share >= cfg.hub_cp_share
    if not (relay_hit or cp_hit):
        return
    sev = "high" if comm.hub_share >= cfg.hub_high_share and relay_hit and cp_hit else "medium"
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


def _late_and_premature(c: _Ctx) -> None:
    g = c.graph
    if g is None:
        return
    by_agent: Dict[str, List[Any]] = defaultdict(list)
    for u in g.dag_units:
        if u.agent_id:
            by_agent[u.agent_id].append(u)
    produced: Dict[str, List[float]] = defaultdict(list)
    for a in c.traj.artifacts.values():
        if a.producer:
            produced[a.producer].append(a.produced_at)
    finals = {c.traj.artifacts[a].producer for a in c.traj.output_artifacts if a in c.traj.artifacts}
    phase1 = g.schema_version is None
    for m in g.messages:
        r = m.receiver
        if not r or r == m.sender or not (m.kind in _CONTENT_KINDS or m.artifacts):
            continue
        sent_at = g.unit(m.send_node).start
        if m.receive_node is None:
            if not any(u.start < sent_at - EPS for u in by_agent.get(r, [])):
                continue   # the receiver never ran before either: not a timing problem
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
        sev = "high" if r in finals else "medium"
        inferred = not m.received
        c.add("premature_synthesis", _cap(sev, "low") if inferred else sev, [r, m.sender or "-"],
              f"{r} produced {len(before)} artifact(s) before receiving a {m.kind} message from {m.sender} "
              f"and produced nothing after it" + (" (it produced the final output)." if r in finals else "."),
              f"{r} may be synthesizing before all of its inputs are available.",
              f"Re-run with {r} waiting for {m.sender} before producing, and compare success and cost.",
              {"message_id": m.message_id, "sender": m.sender, "receiver": r, "kind": m.kind,
               "final_producer": r in finals}, inferred)


def _context_bloat(c: _Ctx) -> None:
    ctx, cfg = c.cm.context, c.cfg
    if ctx is None:
        return
    flagged: Dict[str, List[Any]] = defaultdict(list)
    for call in ctx.calls:
        if call.ratio >= cfg.bloat_ratio and call.excess >= cfg.bloat_min_excess:
            flagged[call.agent_id or "-"].append(call)
    # worst first, so a cross-run summary quotes the most bloated agent
    for agent in sorted(c.order(flagged), key=lambda a: -max(x.ratio for x in flagged[a])):
        calls = flagged[agent]
        worst = max(calls, key=lambda x: (x.ratio, x.excess))
        sev = "high" if worst.ratio >= cfg.bloat_high_ratio else "medium"
        bases = {x.basis for x in calls}
        inferred = bool(bases & {"task_only", "inferred"})
        if "task_only" in bases:
            sev = _cap(sev, "low")      # nothing recorded beyond the task: the inputs may just be untracked
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


def _dead_agents(c: _Ctx) -> None:
    for a in c.agents:
        oc = c.cm.contribution.get(a)
        if oc is None or oc.dead is not True:
            continue
        if oc.provenance_coverage is None or oc.provenance_coverage < c.cfg.dead_min_coverage:
            continue
        u = c.cm.utilization.get(a)
        share = u.token_share if u is not None else None
        sev = "high" if share is not None and share >= c.cfg.dead_high_token_share else "medium"
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


def _serialized(c: _Ctx) -> None:
    dep, cfg = c.cm.dependency, c.cfg
    if dep is None or dep.critical_path_s <= EPS or dep.wall_s <= EPS:
        return
    # time in which nothing was recorded is idle, not serialized work
    busy_wall = dep.wall_s - (dep.unrecorded_s or 0.0)
    ratio = busy_wall / dep.critical_path_s
    gap = busy_wall - dep.critical_path_s
    if ratio < cfg.serial_ratio or gap < max(cfg.serial_min_gap_s, cfg.serial_min_gap_frac * dep.wall_s):
        return
    sev = "high" if ratio >= cfg.serial_high else "medium"
    weak = c.receive_cov is not None and c.receive_cov < cfg.serial_min_coverage
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


def _comm_overhead(c: _Ctx) -> None:
    comm, cfg = c.cm.communication, c.cfg
    if comm is None or comm.comm_work_ratio is None or comm.messages < cfg.comm_min_messages:
        return
    r = comm.comm_work_ratio
    if r < cfg.comm_ratio:
        return
    cats = ", ".join(f"{k} {v['body_tokens']:,}" for k, v in comm.by_category.items() if v.get("messages"))
    top = max(comm.per_agent.values(), key=lambda a: a.body_tokens_sent, default=None)
    c.add("communication_overhead", "high" if r >= cfg.comm_high else "medium",
          [top.agent_id] if top is not None and top.body_tokens_sent else [],
          f"Message bodies came to {comm.body_tokens:,} tokens, {r:.2f}× the {comm.llm_tokens:,} LLM tokens "
          f"(Communication / Work) over {comm.messages} messages ({cats}).",
          "Communication is not automatically waste, but this much talking relative to working may indicate "
          "restated content or chatty coordination.",
          "Re-run with shorter hand-offs (pass artifact references instead of restating content) and compare "
          "success and cost.",
          {"comm_work_ratio": r, "messages": comm.messages, "body_tokens": comm.body_tokens,
           "by_category": comm.by_category})


_DETECTORS = (_redundant_delegation, _ping_pong, _over_agentization, _under_specialization, _bottleneck,
              _late_and_premature, _context_bloat, _dead_agents, _serialized, _comm_overhead)


def detect_smells(traj: Trajectory, metrics: "CoordinationMetrics",
                  config: Optional[SmellConfig] = None) -> List[Diagnostic]:
    """All smells found in one run, in ``SMELLS`` order then by severity."""
    c = _Ctx(traj, metrics, resolve_smell_config(config))
    for detect in _DETECTORS:
        detect(c)
    order = {s: i for i, s in enumerate(SMELLS)}
    return sorted(c.out, key=lambda d: (order.get(d.smell, len(order)), -SEVERITY_RANK[d.severity]))
