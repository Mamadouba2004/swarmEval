"""Core data model for SwarmEval (Phase 1: trajectory schema and task format).

Everything downstream (metrics, reports, exports and the later coordination
and causal layers) is computed from the structures defined here.  The guiding
rule from the design:

    If the execution cannot be represented as structured events,
    it cannot be reliably analyzed.

SCHEMA_VERSION is stored as ``trajectory.metadata["schema_version"]``. Absent means Phase 1 data: no
MESSAGE_RECEIVE events, message ids or ``derived_from``, so analyses must treat those as unknown, not absent.
``Artifact.derived_from`` is ``None`` when not recorded and ``[]`` when recorded with no inputs. Event
timestamps are seconds (wall clock or simulated). Artifacts and the final output stay complete in memory; the
text policy is applied when a trajectory is serialized, so text in loaded data may be truncated for good.

Classes:
- MessageKind: Coarse classification of messages. Recorded now so that the communication analysis planned for Phase
  2 has data to work with.
- TextPolicy: What text is kept in recorded and saved data. With ``capture`` off only a short preview is kept;
  otherwise anything longer than ``max_chars`` (measured on its text form) is truncated.
- Event: A single structured execution event. The schema is intentionally flat so it serialises to one JSON object
  per line. Not every field applies to every event type; unused fields keep their defaults.
- Artifact: An intermediate or final product of an agent (notes, code, a decision).
- Span: One invocation of an agent (start..end).
- Task: One benchmark task. ``evaluator`` is either an ``Evaluator`` instance or a JSON spec such as ``{"type":
  "contains_all", "keywords": [...]}`` so benchmarks can live in plain files.
- Trajectory: The complete structured record of one run of a swarm on one task.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

SCHEMA_VERSION = 2


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    MESSAGE_RECEIVE = "message_receive"
    ARTIFACT = "artifact"
    ARTIFACT_USE = "artifact_use"
    WAIT = "wait"
    ERROR = "error"
    EVAL = "eval"
    ANNOTATION = "annotation"


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class MessageKind(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    COORDINATION = "coordination"
    INFORMATION = "information"
    RESULT = "result"
    REQUEST = "request"
    FEEDBACK = "feedback"


def new_id(prefix: str) -> str:
    """A short random id with ``prefix``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def to_text(obj: Any) -> str:
    """Text form of a value: strings as is, others as sorted JSON."""
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
    """SHA-1 of a value's text form."""
    return hashlib.sha1(to_text(obj).encode("utf-8")).hexdigest()


def _normalize(obj: Any) -> Any:
    """Arguments with dict keys made type-tagged strings and sets sorted, for ``canonical_args``."""
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k if isinstance(k, str) else f"{type(k).__name__}:{k!r}": _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_normalize(v) for v in obj), key=repr)
    return obj


def canonical_args(args: Any) -> str:
    """Stable text form of tool arguments: key and set order ignored, strings compared as given, other
    values by ``repr`` (an object whose repr holds a memory address won't match across runs)."""
    return json.dumps(_normalize(args), sort_keys=True, default=repr)


def preview(obj: Any, limit: int = 240) -> str:
    """A value's text form cut to ``limit`` characters."""
    text = to_text(obj)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class TextPolicy:
    capture: bool = True
    max_chars: int = 20000

    def apply(self, obj: Any) -> Any:
        """The value as it may be kept: a preview, truncated text, or unchanged."""
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

    def to_metadata(self) -> dict[str, Any]:
        """The policy as trajectory metadata."""
        return {"capture_text": self.capture, "max_text_chars": self.max_chars}

    @classmethod
    def from_metadata(cls, md: dict[str, Any]) -> TextPolicy:
        """The policy stored in trajectory metadata."""
        return cls(bool(md.get("capture_text", True)), int(md.get("max_text_chars", 20000)))


@dataclass
class Event:
    event_id: str
    run_id: str
    task_id: str
    event_type: str
    timestamp: float
    agent_id: str | None = None
    span_id: str | None = None
    parent_event_id: str | None = None
    status: str = Status.SUCCESS.value
    latency_ms: float | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    tool_name: str | None = None
    tool_args: Any = None
    sender: str | None = None
    receiver: str | None = None
    kind: str | None = None
    content: Any = None
    content_tokens: int = 0
    content_hash: str | None = None
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str | None = None
    reply_to: str | None = None

    @property
    def end_timestamp(self) -> float:
        """When the event ended, in seconds."""
        return self.timestamp + (self.latency_ms or 0.0) / 1000.0

    @property
    def duration_s(self) -> float:
        """Event duration in seconds."""
        return (self.latency_ms or 0.0) / 1000.0

    @property
    def prompt_tokens(self) -> int:
        """All input processed by the model, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        """Prompt plus output tokens."""
        return self.prompt_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """The event as a JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Rebuild an event, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Artifact:
    artifact_id: str
    name: str
    producer: str | None
    span_id: str | None
    produced_at: float
    content: Any
    content_hash: str
    tokens: int
    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_from: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The artifact as a JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        """Rebuild an artifact, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Span:
    span_id: str
    agent_id: str
    role: str | None
    parent_span_id: str | None
    start: float
    end: float | None = None
    status: str = Status.SUCCESS.value
    model: str | None = None
    start_event_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Span duration in seconds, 0 while still open."""
        if self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        """The span as a JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        """Rebuild a span, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Task:
    task_id: str
    input: Any
    category: str = "general"
    expected: Any = None
    success_criteria: str | None = None
    constraints: list[str] = field(default_factory=list)
    reference: Any = None
    evaluator: Any = None
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError for a missing id or input, or an unknown difficulty."""
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a non-empty string")
        if self.input is None:
            raise ValueError(f"task {self.task_id!r} has no input")
        if self.difficulty not in (None, "simple", "moderate", "hard"):
            raise ValueError(f"task {self.task_id!r}: difficulty must be simple|moderate|hard")

    def to_dict(self) -> dict[str, Any]:
        """The task as a JSON-ready dict with its evaluator as a spec."""
        d = asdict(replace(self, evaluator=None))
        d["evaluator"] = evaluator_spec(self.evaluator)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        """Rebuild a task, rejecting unknown fields."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(k for k in d if k not in known)
        if unknown:
            raise ValueError(f"task {d.get('task_id')!r}: unknown field(s) {unknown}; known: {sorted(known)}")
        return cls(**d)


def evaluator_spec(ev: Any) -> Any:
    """JSON spec for an evaluator; one that cannot be rebuilt from JSON gets an ``unserializable`` marker,
    so reloading fails clearly instead of judging wrongly."""
    if ev is None or isinstance(ev, dict):
        return ev
    spec = getattr(ev, "to_spec", None)
    if callable(spec):
        return spec()
    return {"type": getattr(ev, "__name__", type(ev).__name__), "unserializable": True}


@dataclass
class Benchmark:
    name: str
    tasks: list[Task]
    default_evaluator: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def get(self, task_id: str) -> Task:
        """The task with ``task_id``; KeyError if absent."""
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)

    def validate(self) -> None:
        """Check ids and fields, and build every task's evaluator so a bad
        spec fails here rather than after the swarm has already run."""
        from .evaluators import resolve
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

    def to_dict(self) -> dict[str, Any]:
        """The benchmark as a JSON-ready dict."""
        return {
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "default_evaluator": evaluator_spec(self.default_evaluator),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Benchmark:
        """Rebuild a benchmark from a dict."""
        return cls(
            name=d.get("name", "benchmark"),
            tasks=[Task.from_dict(t) for t in d.get("tasks", [])],
            default_evaluator=d.get("default_evaluator"),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        """Write the benchmark as JSON, raising if an evaluator could not be rebuilt from the file (e.g. a
        lambda) instead of writing a benchmark that only fails on reload."""
        from .evaluators import from_spec
        d = self.to_dict()
        for spec in [d["default_evaluator"]] + [t["evaluator"] for t in d["tasks"]]:
            if spec is not None:
                from_spec(spec)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> Benchmark:
        """Load a benchmark from JSON, or from JSONL with one task per line."""
        with open(path) as f:
            if path.endswith(".jsonl"):
                tasks = [Task.from_dict(json.loads(line)) for line in f if line.strip()]
                return cls(name=path, tasks=tasks)
            return cls.from_dict(json.load(f))


@dataclass
class Evaluation:
    success: bool
    score: float
    evaluator: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    constraints_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """The evaluation as a JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evaluation:
        """Rebuild an evaluation from a dict."""
        return cls(
            success=bool(d.get("success", False)),
            score=float(d.get("score", 0.0)),
            evaluator=d.get("evaluator", "unknown"),
            details=d.get("details", {}),
            constraints_passed=d.get("constraints_passed"),
        )


@dataclass
class Trajectory:
    run_id: str
    task_id: str
    task: Task
    seed: int = 0
    trial: int = 0
    events: list[Event] = field(default_factory=list)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    spans: dict[str, Span] = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None
    final_output: Any = None
    output_artifacts: list[str] = field(default_factory=list)
    status: str = Status.SUCCESS.value
    error: str | None = None
    evaluation: Evaluation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Run duration in seconds, 0 if not finished."""
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    def events_of(self, *types: str) -> list[Event]:
        """Events of the given types, in recording order."""
        wanted = {t.value if isinstance(t, EventType) else t for t in types}
        return [e for e in self.events if e.event_type in wanted]

    def llm_calls(self) -> list[Event]:
        """All LLM_CALL events."""
        return self.events_of(EventType.LLM_CALL)

    def tool_calls(self) -> list[Event]:
        """All TOOL_CALL events."""
        return self.events_of(EventType.TOOL_CALL)

    def messages(self) -> list[Event]:
        """All MESSAGE events."""
        return self.events_of(EventType.MESSAGE)

    def agent_ids(self) -> list[str]:
        """Agents in order of first start, ties broken by id."""
        seen: list[str] = []
        for s in sorted(self.spans.values(), key=lambda s: (s.start, s.agent_id)):
            if s.agent_id not in seen:
                seen.append(s.agent_id)
        return seen

    def spans_for(self, agent_id: str) -> list[Span]:
        """Spans of one agent in start order."""
        return sorted((s for s in self.spans.values() if s.agent_id == agent_id),
                      key=lambda s: s.start)

    @property
    def input_tokens(self) -> int:
        """Prompt tokens over all LLM calls, cached included."""
        return sum(e.prompt_tokens for e in self.llm_calls())

    @property
    def output_tokens(self) -> int:
        """Output tokens over all LLM calls."""
        return sum(e.output_tokens for e in self.llm_calls())

    @property
    def total_tokens(self) -> int:
        """Prompt plus output tokens over all LLM calls."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float | None:
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
        """True if every LLM call was priced."""
        return all(e.cost_usd is not None for e in self.llm_calls())

    @property
    def succeeded(self) -> bool:
        """True if the run's evaluation passed."""
        return bool(self.evaluation and self.evaluation.success)

    @property
    def score(self) -> float:
        """The evaluation score, 0.0 if not evaluated."""
        return float(self.evaluation.score) if self.evaluation else 0.0

    def to_dict(self) -> dict[str, Any]:
        """The trajectory as a JSON-ready dict, with the text policy applied."""
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
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        """Rebuild a trajectory, flagging text that was truncated when it was saved."""
        metadata = dict(d.get("metadata", {}))
        texts = [a.get("content") for a in d.get("artifacts", {}).values()] + [d.get("final_output")]
        if (not TextPolicy.from_metadata(metadata).capture
                or any(isinstance(t, str) and t.endswith("…[truncated]") for t in texts)):
            metadata["text_truncated"] = True
        trial = d.get("trial", 0)
        if not isinstance(trial, int) or isinstance(trial, bool):
            raise ValueError(f"run {d['run_id']!r}: trial must be an integer, got {trial!r}")
        return cls(
            run_id=d["run_id"],
            task_id=d["task_id"],
            task=Task.from_dict(d["task"]),
            seed=d.get("seed", 0),
            trial=trial,
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
            metadata=metadata,
        )

    def save(self, path: str) -> None:
        """Write the trajectory as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> Trajectory:
        """Load a trajectory from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
