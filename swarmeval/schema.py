"""Core data model for SwarmEval (Phase 1: trajectory schema and task format).

Everything downstream (metrics, reports, exports and the later coordination
and causal layers) is computed from the structures defined here.  The guiding
rule from the design:

    If the execution cannot be represented as structured events,
    it cannot be reliably analyzed.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


# Version of the recorded data, stored as ``trajectory.metadata["schema_version"]``.
# Absent means Phase 1 data: no MESSAGE_RECEIVE events, message ids or
# ``derived_from``, so analyses must treat those as unknown rather than absent.
SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    MESSAGE_RECEIVE = "message_receive"  # a message was taken from the receiver's mailbox
    ARTIFACT = "artifact"          # an artifact was produced
    ARTIFACT_USE = "artifact_use"  # an artifact was consumed
    WAIT = "wait"
    ERROR = "error"
    EVAL = "eval"
    ANNOTATION = "annotation"


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class MessageKind(str, Enum):
    """Coarse classification of messages.  Recorded now so that the
    communication analysis planned for Phase 2 has data to work with."""
    TASK_ASSIGNMENT = "task_assignment"
    COORDINATION = "coordination"
    INFORMATION = "information"
    RESULT = "result"
    REQUEST = "request"
    FEEDBACK = "feedback"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(obj)


def estimate_tokens(obj: Any) -> int:
    """Cheap token estimate (~4 characters per token) used when a real count
    is unavailable.  Always prefer real usage numbers when you have them."""
    text = to_text(obj)
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def content_hash(obj: Any) -> str:
    return hashlib.sha1(to_text(obj).encode("utf-8")).hexdigest()


def preview(obj: Any, limit: int = 240) -> str:
    text = to_text(obj)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class TextPolicy:
    """What text is kept in recorded and saved data.  With ``capture`` off
    only a short preview is kept; otherwise anything longer than ``max_chars``
    (measured on its text form) is truncated."""
    capture: bool = True
    max_chars: int = 20000

    def apply(self, obj: Any) -> Any:
        if obj is None:
            return None
        if not self.capture:
            return preview(obj, 120)
        if isinstance(obj, (bool, int, float)):
            return obj
        text = to_text(obj)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "…[truncated]"
        return obj

    def to_metadata(self) -> Dict[str, Any]:
        return {"capture_text": self.capture, "max_text_chars": self.max_chars}

    @classmethod
    def from_metadata(cls, md: Dict[str, Any]) -> "TextPolicy":
        return cls(bool(md.get("capture_text", True)), int(md.get("max_text_chars", 20000)))


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A single structured execution event.

    The schema is intentionally flat so it serialises to one JSON object per
    line.  Not every field applies to every event type; unused fields keep
    their defaults.
    """
    event_id: str
    run_id: str
    task_id: str
    event_type: str
    timestamp: float                       # seconds (wall clock or simulated)
    agent_id: Optional[str] = None
    span_id: Optional[str] = None          # the agent invocation this belongs to
    parent_event_id: Optional[str] = None
    status: str = Status.SUCCESS.value
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0             # prompt-cache reads (billed below the input rate)
    cache_write_tokens: int = 0            # prompt-cache writes (billed above the input rate)
    cost_usd: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Any = None
    sender: Optional[str] = None
    receiver: Optional[str] = None
    kind: Optional[str] = None             # message kind / artifact name / wait reason / role
    content: Any = None                    # message content, output preview, ...
    content_tokens: int = 0
    content_hash: Optional[str] = None
    input_artifacts: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Message identity (schema v2): MESSAGE and MESSAGE_RECEIVE events of one
    # message share ``message_id``; ``reply_to`` names the message answered.
    message_id: Optional[str] = None
    reply_to: Optional[str] = None

    @property
    def end_timestamp(self) -> float:
        return self.timestamp + (self.latency_ms or 0.0) / 1000.0

    @property
    def duration_s(self) -> float:
        return (self.latency_ms or 0.0) / 1000.0

    @property
    def prompt_tokens(self) -> int:
        """All input processed by the model, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Artifact:
    """An intermediate or final product of an agent (notes, code, a decision)."""
    artifact_id: str
    name: str
    producer: Optional[str]
    span_id: Optional[str]
    produced_at: float
    content: Any
    content_hash: str
    tokens: int
    event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Ids of the artifacts this one was made from.  ``None`` = not recorded
    # (Phase 1 data, or the producer did not say); ``[]`` = recorded, no inputs.
    derived_from: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Span:
    """One invocation of an agent (start..end)."""
    span_id: str
    agent_id: str
    role: Optional[str]
    parent_span_id: Optional[str]
    start: float
    end: Optional[float] = None
    status: str = Status.SUCCESS.value
    model: Optional[str] = None
    start_event_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Tasks & benchmarks (the standardized task format)
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """One benchmark task.

    ``evaluator`` is either an ``Evaluator`` instance or a JSON spec such as
    ``{"type": "contains_all", "keywords": [...]}`` so benchmarks can live in
    plain files.
    """
    task_id: str
    input: Any
    category: str = "general"
    expected: Any = None
    success_criteria: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    reference: Any = None
    evaluator: Any = None
    difficulty: Optional[str] = None      # "simple" | "moderate" | "hard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a non-empty string")
        if self.input is None:
            raise ValueError(f"task {self.task_id!r} has no input")
        if self.difficulty not in (None, "simple", "moderate", "hard"):
            raise ValueError(f"task {self.task_id!r}: difficulty must be simple|moderate|hard")

    def to_dict(self) -> Dict[str, Any]:
        # The evaluator may hold locks or clients that cannot be deep-copied,
        # so it is left out of asdict() and replaced by its spec.
        d = asdict(replace(self, evaluator=None))
        d["evaluator"] = evaluator_spec(self.evaluator)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(k for k in d if k not in known)
        if unknown:
            raise ValueError(f"task {d.get('task_id')!r}: unknown field(s) {unknown}; known: {sorted(known)}")
        return cls(**d)


def evaluator_spec(ev: Any) -> Any:
    """JSON spec for an evaluator.  Evaluators that cannot be rebuilt from JSON
    (plain callables, unregistered classes) get an ``unserializable`` marker so
    that reloading them fails with a clear message instead of a wrong judgment."""
    if ev is None or isinstance(ev, dict):
        return ev
    spec = getattr(ev, "to_spec", None)
    if callable(spec):
        return spec()
    return {"type": getattr(ev, "__name__", type(ev).__name__), "unserializable": True}


@dataclass
class Benchmark:
    name: str
    tasks: List[Task]
    default_evaluator: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)

    def validate(self) -> None:
        """Check ids and fields, and build every task's evaluator so a bad
        spec fails here rather than after the swarm has already run."""
        from .evaluators import resolve  # local import: evaluators imports schema
        seen = set()
        for t in self.tasks:
            t.validate()
            if t.task_id in seen:
                raise ValueError(f"duplicate task_id {t.task_id!r} in benchmark {self.name!r}")
            seen.add(t.task_id)
            try:
                ev = resolve(t, self.default_evaluator)
                if ev is not None:
                    ev.check(t)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"task {t.task_id!r}: invalid evaluator: {exc}") from None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "default_evaluator": evaluator_spec(self.default_evaluator),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Benchmark":
        return cls(
            name=d.get("name", "benchmark"),
            tasks=[Task.from_dict(t) for t in d.get("tasks", [])],
            default_evaluator=d.get("default_evaluator"),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        """Write the benchmark as JSON.  Raises if an evaluator could not be
        rebuilt from the file (e.g. a lambda) instead of writing a benchmark
        that only fails on reload."""
        from .evaluators import from_spec  # local import: evaluators imports schema
        d = self.to_dict()
        for spec in [d["default_evaluator"]] + [t["evaluator"] for t in d["tasks"]]:
            if spec is not None:
                from_spec(spec)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        with open(path) as f:
            if path.endswith(".jsonl"):
                tasks = [Task.from_dict(json.loads(line)) for line in f if line.strip()]
                return cls(name=path, tasks=tasks)
            return cls.from_dict(json.load(f))

    def subset(self, task_ids: Iterable[str]) -> "Benchmark":
        ids = set(task_ids)
        return Benchmark(self.name, [t for t in self.tasks if t.task_id in ids],
                         self.default_evaluator, dict(self.metadata))


# --------------------------------------------------------------------------- #
# Evaluation results
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    success: bool
    score: float
    evaluator: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    constraints_passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evaluation":
        return cls(
            success=bool(d.get("success", False)),
            score=float(d.get("score", 0.0)),
            evaluator=d.get("evaluator", "unknown"),
            details=d.get("details", {}),
            constraints_passed=d.get("constraints_passed"),
        )


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """The complete structured record of one run of a swarm on one task."""
    run_id: str
    task_id: str
    task: Task
    seed: int = 0
    trial: int = 0
    events: List[Event] = field(default_factory=list)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    spans: Dict[str, Span] = field(default_factory=dict)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    final_output: Any = None
    output_artifacts: List[str] = field(default_factory=list)
    status: str = Status.SUCCESS.value
    error: Optional[str] = None
    evaluation: Optional[Evaluation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------
    @property
    def duration(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    def events_of(self, *types: str) -> List[Event]:
        wanted = {t.value if isinstance(t, EventType) else t for t in types}
        return [e for e in self.events if e.event_type in wanted]

    def llm_calls(self) -> List[Event]:
        return self.events_of(EventType.LLM_CALL)

    def tool_calls(self) -> List[Event]:
        return self.events_of(EventType.TOOL_CALL)

    def messages(self) -> List[Event]:
        return self.events_of(EventType.MESSAGE)

    def agent_ids(self) -> List[str]:
        seen: List[str] = []
        # agent id breaks ties so concurrent agents starting together keep a stable order
        for s in sorted(self.spans.values(), key=lambda s: (s.start, s.agent_id)):
            if s.agent_id not in seen:
                seen.append(s.agent_id)
        return seen

    def spans_for(self, agent_id: str) -> List[Span]:
        return sorted((s for s in self.spans.values() if s.agent_id == agent_id),
                      key=lambda s: s.start)

    @property
    def input_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.llm_calls())

    @property
    def output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.llm_calls())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        """Sum of known LLM costs: ``0.0`` with no LLM calls, ``None`` when no
        call could be priced.  Check ``cost_is_complete`` for partial sums."""
        calls = self.llm_calls()
        if not calls:
            return 0.0
        known = [e.cost_usd for e in calls if e.cost_usd is not None]
        if not known:
            return None
        return float(sum(known))

    @property
    def cost_is_complete(self) -> bool:
        return all(e.cost_usd is not None for e in self.llm_calls())

    @property
    def succeeded(self) -> bool:
        return bool(self.evaluation and self.evaluation.success)

    @property
    def score(self) -> float:
        return float(self.evaluation.score) if self.evaluation else 0.0

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        # Artifacts and the final output stay complete in memory (swarms read
        # them back), so the text policy is applied here, on the way out.
        policy = TextPolicy.from_metadata(self.metadata)
        artifacts = {}
        for k, a in self.artifacts.items():
            ad = asdict(replace(a, content=None))
            ad["content"] = policy.apply(a.content)
            artifacts[k] = ad
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task": self.task.to_dict(),
            "seed": self.seed,
            "trial": self.trial,
            "events": [e.to_dict() for e in self.events],
            "artifacts": artifacts,
            "spans": {k: s.to_dict() for k, s in self.spans.items()},
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "final_output": policy.apply(self.final_output),
            "output_artifacts": self.output_artifacts,
            "status": self.status,
            "error": self.error,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(
            run_id=d["run_id"],
            task_id=d["task_id"],
            task=Task.from_dict(d["task"]),
            seed=d.get("seed", 0),
            trial=d.get("trial", 0),
            events=[Event.from_dict(e) for e in d.get("events", [])],
            artifacts={k: Artifact.from_dict(a) for k, a in d.get("artifacts", {}).items()},
            spans={k: Span.from_dict(s) for k, s in d.get("spans", {}).items()},
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            final_output=d.get("final_output"),
            output_artifacts=d.get("output_artifacts", []),
            status=d.get("status", "success"),
            error=d.get("error"),
            evaluation=Evaluation.from_dict(d["evaluation"]) if d.get("evaluation") else None,
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with open(path) as f:
            return cls.from_dict(json.load(f))
