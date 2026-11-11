"""Layer 3 -- redundant work in one run, and its summary over runs (Phase 2).

Two separate numbers, never merged:

* **tool redundancy** = accidental duplicate tool calls ÷ tool calls.  Two
  calls duplicate each other when they have the same tool and the same
  arguments (strings compared exactly; dict keys and sets in any order).
  The first call (by start time) is the original; each later one is
  classified as

  - ``retry``: the same agent's previous call with that key failed;
  - ``verification``: made by a checking role (critic, verifier, reviewer,
    validator, judge) or marked ``intent="verification"`` on the call or span;
  - ``accidental``: anything else.  A non-retry repeat by the *same* agent is
    accidental too, and is also counted in ``self_repeats``.

  Retries and verification are intentional and kept out of the ratio.
  Arguments are compared by a hash of their full canonical form taken at
  record time; ``tool_degraded`` is True when a call lacks that hash and its
  stored arguments were truncated, so distinct calls may look the same.
* **artifact redundancy** = accidentally redundant artifacts ÷ artifacts.  An
  artifact is redundant when an earlier artifact from a *different* producer
  has the same content hash or is similar (``similarity`` backend, default
  lexical).  A copy of an artifact its producer derived from, consumed or
  received by message is not redundant work, and a verification producer's artifact counts as
  intentional.  ``method`` names how text was compared; ``degraded`` is True
  when artifact text was not captured, so only exact hashes were compared.

Intentional verification and accidental duplication look alike in the
record; the role / intent conventions are what tells them apart.

Classes:
- RedundancyMetrics: Redundant work of one run (definitions in the module docstring). Ratios are ``None`` when there
  is nothing to measure. ``pairwise_overlap`` lists agent pairs whose sets of tool keys overlap: ``{"agents": [a,
  b], "jaccard": ..., "shared": n}``. ``duplicate_groups`` is evidence (first ``MAX_EVIDENCE`` keys called more than
  once; ``duplicate_groups_total`` counts them all); ``redundant_artifact_pairs`` likewise for artifacts.
- RedundancySummary: Redundancy over runs. Ratio summaries cover the runs where the ratio is defined (``n``).
  ``methods`` counts runs per artifact comparison method; ``degraded_runs`` how many compared hashes only;
  ``similarity_truncated_runs`` how many hit the similarity backend's pair cap, so their artifact redundancy may be
  under-counted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph.builder import ExecutionGraph
from ..schema import Event, EventType, Status, Trajectory, canonical_args, content_hash, to_text
from ..similarity import resolve_similarity, similar_pairs
from ..stats import Summary, known_summary, ranged, summarize, summary_dict

VERIFICATION_ROLES = frozenset({"critic", "verifier", "reviewer", "validator", "judge"})
DUPLICATE_CLASSES = ("accidental", "verification", "retry")
MAX_EVIDENCE = 20
REDUNDANCY_METRICS = (
    ("tool_redundancy", "Redundant Tool Calls", False, "pct", "tool_redundancy"),
    ("artifact_redundancy", "Redundant Artifacts", False, "pct", "artifact_redundancy"),
)


def tool_key(ev: Event) -> tuple[str, str]:
    """``(tool, hash of the full canonical args)``, from the recorded ``args_hash`` when present
    (older data hashes the stored, possibly truncated, args)."""
    h = ev.metadata.get("args_hash")
    return (ev.tool_name or "", h if h is not None else content_hash(canonical_args(ev.tool_args)))


def is_verification(traj: Trajectory, agent_id: str | None, span_id: str | None,
                    metadata: dict[str, Any] | None = None) -> bool:
    """True for work done to check others' work: a checking role on the span
    (or, without a span, any span of the agent), or ``intent="verification"``."""
    if (metadata or {}).get("intent") == "verification":
        return True
    span = traj.spans.get(span_id or "")
    spans = [span] if span is not None else [s for s in traj.spans.values() if s.agent_id == agent_id]
    return any((s.role or "").lower() in VERIFICATION_ROLES or s.metadata.get("intent") == "verification"
               for s in spans)


@dataclass
class RedundancyMetrics:
    tool_calls: int = 0
    duplicate_tool_calls: int = 0
    accidental_duplicates: int = 0
    verification_duplicates: int = 0
    retry_duplicates: int = 0
    self_repeats: int = 0
    tool_redundancy: float | None = None
    artifacts: int = 0
    redundant_artifacts: int = 0
    intentional_artifacts: int = 0
    artifact_redundancy: float | None = None
    method: str = "hash"
    degraded: bool = False
    tool_degraded: bool = False
    similarity_threshold: float | None = None
    similarity_truncated: bool = False
    pairwise_overlap: list[dict[str, Any]] = field(default_factory=list)
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    duplicate_groups_total: int = 0
    redundant_artifact_pairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The metrics as a JSON-ready dict."""
        return asdict(self)


def _tool_redundancy(traj: Trajectory, out: RedundancyMetrics) -> None:
    """Count repeated tool calls, splitting accidental duplicates from verification."""
    seq = {ev.event_id: i for i, ev in enumerate(traj.events)}
    calls = sorted(traj.tool_calls(), key=lambda e: (round(e.timestamp, 9), e.agent_id or "", seq[e.event_id]))
    out.tool_calls = len(calls)
    groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for ev in calls:
        groups[tool_key(ev)].append(ev)
        if ev.metadata.get("args_hash") is None and (
                traj.metadata.get("text_truncated") or str(ev.tool_args).endswith(("…", "…[truncated]"))):
            out.tool_degraded = True
    keys_of: dict[str, set[tuple[str, str]]] = defaultdict(set)
    counts: Counter = Counter()
    evidence: list[dict[str, Any]] = []
    for key, evs in groups.items():
        for ev in evs:
            if ev.agent_id is not None:
                keys_of[ev.agent_id].add(key)
        if len(evs) < 2:
            continue
        out.duplicate_groups_total += 1
        last_by_agent: dict[str | None, Event] = {evs[0].agent_id: evs[0]}
        dups = []
        for ev in evs[1:]:
            prev = last_by_agent.get(ev.agent_id)
            if prev is not None and prev.status == Status.ERROR.value:
                cls = "retry"
            elif is_verification(traj, ev.agent_id, ev.span_id, ev.metadata):
                cls = "verification"
            else:
                cls = "accidental"
                if prev is not None:
                    out.self_repeats += 1
            counts[cls] += 1
            last_by_agent[ev.agent_id] = ev
            dups.append({"agent": ev.agent_id, "class": cls, "event_id": ev.event_id})
        if len(evidence) < MAX_EVIDENCE:
            evidence.append({"tool": key[0], "args": evs[0].tool_args, "original_agent": evs[0].agent_id,
                             "original_event_id": evs[0].event_id, "duplicates": dups})
    out.duplicate_tool_calls = sum(counts.values())
    out.accidental_duplicates = counts["accidental"]
    out.verification_duplicates = counts["verification"]
    out.retry_duplicates = counts["retry"]
    out.tool_redundancy = out.accidental_duplicates / len(calls) if calls else None
    out.duplicate_groups = evidence

    agents = sorted(keys_of)
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            shared = keys_of[a] & keys_of[b]
            if shared:
                out.pairwise_overlap.append({"agents": [a, b], "shared": len(shared),
                                             "jaccard": len(shared) / len(keys_of[a] | keys_of[b])})


def _ancestors(traj: Trajectory, art_id: str) -> set[str]:
    """Artifacts ``art_id`` was (transitively) derived from."""
    seen: set[str] = set()
    stack = list((traj.artifacts[art_id].derived_from or []) if art_id in traj.artifacts else [])
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        if a in traj.artifacts:
            stack.extend(traj.artifacts[a].derived_from or [])
    return seen


def _artifact_redundancy(traj: Trajectory, out: RedundancyMetrics, similarity: Any) -> None:
    """Find artifacts that repeat earlier ones, excluding copies of their own inputs."""
    seq = {ev.event_id: i for i, ev in enumerate(traj.events)}
    arts = sorted(traj.artifacts.values(),
                  key=lambda a: (round(a.produced_at, 9), seq.get(a.event_id or "", len(seq))))
    out.artifacts = len(arts)
    if not arts:
        return
    captured = not traj.metadata.get("text_truncated") and all(a.content is not None for a in arts)
    out.degraded = not captured
    sim = None if out.degraded else resolve_similarity(similarity)
    out.method = sim.name if sim is not None else "hash"
    out.similarity_threshold = sim.threshold if sim is not None else None

    matches: dict[int, dict[int, float]] = defaultdict(dict)
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(arts):
        for j in by_hash[a.content_hash]:
            matches[i][j] = 1.0
        by_hash[a.content_hash].append(i)
    if sim is not None:
        for i, j, s in similar_pairs(sim, [to_text(a.content) for a in arts]):
            lo, hi = min(i, j), max(i, j)
            matches[hi].setdefault(lo, s)
        out.similarity_truncated = bool(getattr(sim, "truncated", False))

    consumed: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    inputs = (EventType.ARTIFACT_USE.value, EventType.LLM_CALL.value, EventType.TOOL_CALL.value,
              EventType.MESSAGE_RECEIVE.value)
    for ev in traj.events:
        if ev.event_type in inputs:
            for a in ev.input_artifacts:
                consumed[(a, ev.agent_id)].append(ev.timestamp)

    pairs: list[dict[str, Any]] = []
    for i, later in enumerate(arts):
        cands = matches.get(i)
        if not cands:
            continue
        ancestors = _ancestors(traj, later.artifact_id)

        def is_input(earlier: Any, later: Any = later, ancestors: set[str] = ancestors) -> bool:
            return earlier.artifact_id in ancestors or any(
                t <= later.produced_at for t in consumed.get((earlier.artifact_id, later.producer), []))

        if any(is_input(arts[j]) for j in cands):
            continue
        hit: tuple[int, float] | None = None
        for j, score in sorted(cands.items(), key=lambda kv: (-kv[1], kv[0])):
            if arts[j].producer != later.producer:
                hit = (j, score)
                break
        if hit is None:
            continue
        intentional = is_verification(traj, later.producer, later.span_id, later.metadata)
        if intentional:
            out.intentional_artifacts += 1
        else:
            out.redundant_artifacts += 1
        if len(pairs) < MAX_EVIDENCE:
            earlier = arts[hit[0]]
            pairs.append({"artifact": later.artifact_id, "name": later.name, "producer": later.producer,
                          "matches": earlier.artifact_id, "matches_name": earlier.name,
                          "matches_producer": earlier.producer, "score": hit[1],
                          "exact": later.content_hash == earlier.content_hash,
                          "class": "verification" if intentional else "accidental"})
    out.redundant_artifact_pairs = pairs
    out.artifact_redundancy = out.redundant_artifacts / len(arts)


def compute_redundancy(traj: Trajectory, graph: ExecutionGraph | None = None,
                       similarity: Any = None) -> RedundancyMetrics:
    """Tool and artifact redundancy of one run; ``similarity`` is a backend or spec for ``resolve_similarity``.
    ``graph`` is accepted for symmetry with the other analyses and not used."""
    out = RedundancyMetrics()
    _tool_redundancy(traj, out)
    _artifact_redundancy(traj, out, similarity)
    return out



@dataclass
class RedundancySummary:
    n_runs: int
    tool_redundancy: Summary | None
    artifact_redundancy: Summary | None
    accidental_duplicates: Summary
    verification_duplicates: Summary
    retry_duplicates: Summary
    self_repeats: Summary
    redundant_artifacts: Summary
    intentional_artifacts: Summary
    methods: dict[str, int] = field(default_factory=dict)
    degraded_runs: int = 0
    similarity_truncated_runs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dict."""
        return summary_dict(self)


def summarize_redundancy(per_run: Sequence[RedundancyMetrics]) -> RedundancySummary:
    """Redundancy over runs."""
    S = summarize
    return RedundancySummary(
        n_runs=len(per_run),
        tool_redundancy=known_summary([r.tool_redundancy for r in per_run]),
        artifact_redundancy=known_summary([r.artifact_redundancy for r in per_run]),
        accidental_duplicates=S([r.accidental_duplicates for r in per_run]),
        verification_duplicates=S([r.verification_duplicates for r in per_run]),
        retry_duplicates=S([r.retry_duplicates for r in per_run]),
        self_repeats=S([r.self_repeats for r in per_run]),
        redundant_artifacts=S([r.redundant_artifacts for r in per_run]),
        intentional_artifacts=S([r.intentional_artifacts for r in per_run]),
        methods=dict(Counter(r.method for r in per_run if r.artifacts)),
        degraded_runs=sum(1 for r in per_run if r.degraded),
        similarity_truncated_runs=sum(1 for r in per_run if r.similarity_truncated))


def redundancy_rows(red: RedundancySummary) -> list[tuple[str, str] | None]:
    """The redundancy rows of the COORDINATION box."""
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    method = ", ".join(sorted(red.methods)) or "none"
    if red.degraded_runs:
        method += ", hash only in " + str(red.degraded_runs)
    return [("Redundant Tool Calls (accidental ÷ calls)", ranged(red.tool_redundancy, pct)),
            (f"Redundant Artifacts ({method})", ranged(red.artifact_redundancy, pct))]
