"""Trajectory recorder and run context (Phase 1: observability).

The recorder captures every agent invocation, LLM call, tool call, message,
artifact and wait as a structured event.  It is also the agent interface:
a swarm interacts with SwarmEval only through the ``RunContext`` it is
handed and the ``AgentSpan`` objects it opens.

Typical use inside a swarm::

    def run(self, task, ctx):
        with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
            plan = coord.llm(call_model, prompt)
            coord.send("researcher", plan, kind="task_assignment")
        with ctx.agent("researcher", role="research") as r:
            for msg in r.receive():
                notes = r.call_tool("web_search", search, query=msg.content)
                r.produce("notes", notes)
        ...
        return answer

Async swarms use ``async with ctx.agent(...)``, ``await span.allm(...)``,
``await span.acall_tool(...)`` and ``await ctx.gather(...)``.

``_current_llm_call`` holds the LLM call being recorded, so client instrumentation fills it in rather than
recording the same API call twice. ``_current_call`` holds (span_id, event_id) of the innermost LLM/tool call;
a call started inside another call of the same span records the outer one as ``metadata["within_call"]``, which
lets the graph tell nesting (a tool run by an LLM call) from ``gather``ed sibling calls with similar intervals.
A span opened inside a call (e.g. an agent run as a tool) records it the same way in ``span.metadata``.

``ctx.send`` and ``ctx.receive`` take an explicit agent id; when another agent's span is active, the event is
not attached to that span (no span id, no parent event).  On a ``SimClock`` each parallel branch has its own
time, so a receive moves the reader's clock up to the send if needed: a read never precedes its send.

Classes:
- SimClock: A virtual clock. Simulated swarms advance it explicitly so that demos and tests run instantly while
  still producing realistic timings. Concurrent branches started with ``ctx.parallel`` / ``ctx.gather`` each get
  their own timeline (held in a context variable), and the parent resumes at the latest branch end -- so simulated
  parallel work overlaps instead of adding up.
- LLMResult: Return this from a function passed to ``AgentSpan.llm`` to report real token usage.
- LLMCall: Mutable handle yielded by ``AgentSpan.llm_call``.
- ToolCall: Mutable handle yielded by ``AgentSpan.tool_call``.
- _ExceptionTracker: Which event first recorded each exception, so a re-recorded (or wrapping) one is marked
  ``propagated``. Has its own lock.
- _Mailbox: Per-agent message queues (storage only), with their own lock.
- RunContext: Everything a swarm needs during one run: recording, messaging, seeded randomness and the clock.
- AgentSpan: Context manager (sync or async) for one agent invocation.
"""
from __future__ import annotations

import contextvars
import inspect
import random
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .pricing import DEFAULT_PRICING, PricingTable
from .schema import (
    SCHEMA_VERSION,
    Artifact,
    Event,
    EventType,
    Span,
    Status,
    Task,
    TextPolicy,
    Trajectory,
    canonical_args,
    content_hash,
    estimate_tokens,
    new_id,
    preview,
    to_text,
)


class Clock:
    def now(self) -> float:  # pragma: no cover - interface
        """Current time in seconds."""
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:  # pragma: no cover - interface
        """Wait ``seconds`` on this clock."""
        raise NotImplementedError

    def fork(self) -> None:
        """Called at the start of a concurrent branch (inside its context)."""

    def join(self, end: float) -> None:
        """Called in the parent after its branches finish; ``end`` is the
        latest branch end time."""


class WallClock(Clock):
    def now(self) -> float:
        """Current wall-clock time in seconds."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Sleep for real."""
        if seconds > 0:
            time.sleep(seconds)


class SimClock(Clock):

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()
        self._branch: contextvars.ContextVar = contextvars.ContextVar(f"simclock_{id(self)}", default=None)

    def now(self) -> float:
        """Current simulated time, on this branch's timeline when inside one."""
        box = self._branch.get()
        return box[0] if box is not None else self._t

    def sleep(self, seconds: float) -> None:
        """Advance simulated time instead of sleeping."""
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move simulated time forward by ``seconds`` on the current timeline."""
        box = self._branch.get()
        if box is not None:
            box[0] += max(0.0, float(seconds))
            return
        with self._lock:
            self._t += max(0.0, float(seconds))

    def set(self, t: float) -> None:
        """Set simulated time on the current timeline."""
        box = self._branch.get()
        if box is not None:
            box[0] = float(t)
            return
        with self._lock:
            self._t = float(t)

    def fork(self) -> None:
        """Start a separate timeline for a concurrent branch at the current time."""
        self._branch.set([self.now()])

    def join(self, end: float) -> None:
        """Resume the parent timeline at the latest branch end."""
        self.set(max(self.now(), end))


@dataclass
class Message:
    message_id: str
    sender: str
    receiver: str
    content: Any
    kind: str
    artifacts: list[str]
    timestamp: float
    tokens: int
    event_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None


@dataclass
class LLMResult:
    text: Any
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    raw: Any = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMCall:

    def __init__(self, model: str | None, input_tokens: int, output_tokens: int,
                 input_artifacts: list[str], prompt: Any) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.input_artifacts = input_artifacts
        self.prompt = prompt
        self.output: Any = None
        self.metadata: dict[str, Any] = {}
        self.event_id = new_id("ev")


class ToolCall:

    def __init__(self, name: str, args: Any, input_artifacts: list[str]) -> None:
        self.name = name
        self.args = args
        self.input_artifacts = input_artifacts
        self.result: Any = None
        self.metadata: dict[str, Any] = {}
        self.event_id = new_id("ev")


_current_span: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_span", default=None)
_current_llm_call: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_llm_call", default=None)
_current_call: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_call", default=None)


def _enter_call(span: AgentSpan, call: LLMCall | ToolCall) -> contextvars.Token:
    """Mark ``call`` as the innermost call, noting an enclosing call of the same span; return the reset token."""
    outer = _current_call.get()
    if outer is not None and outer[0] == span.span_id:
        call.metadata["within_call"] = outer[1]
    return _current_call.set((span.span_id, call.event_id))


def _ids(artifacts: Iterable[Artifact | str]) -> list[str]:
    """Artifact ids from a mix of Artifact objects and id strings."""
    return [a.artifact_id if isinstance(a, Artifact) else str(a) for a in artifacts]


WaitTarget = Message | Artifact | str


def _wait_refs(on: WaitTarget | Iterable[WaitTarget] | None) -> tuple[list[str], dict[str, Any]]:
    """What a wait was for: artifact ids (as ``input_artifacts``) and metadata naming awaited messages and
    agents (a plain string is an agent id); a wait without ``on`` records nothing extra."""
    if on is None:
        return [], {}
    items = [on] if isinstance(on, (Message, Artifact, str)) else list(on)
    arts: list[str] = []
    msgs: list[str] = []
    agents: list[str] = []
    for x in items:
        if isinstance(x, Message):
            msgs.append(x.message_id)
        elif isinstance(x, Artifact):
            arts.append(x.artifact_id)
        else:
            agents.append(str(x))
    meta: dict[str, Any] = {}
    if msgs:
        meta["on_messages"] = msgs
    if agents:
        meta["on_agents"] = agents
    return arts, meta


def _reject_awaitable(result: Any, sync_name: str, async_name: str) -> None:
    """Raise TypeError if a sync wrapper got an awaitable, pointing to the async variant."""
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(f"{sync_name}() got an awaitable from an async function; "
                        f"use `await span.{async_name}(...)` instead")


def _call_record(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """How a function's arguments are recorded (prompt or tool args)."""
    if len(args) == 1 and not kwargs:
        return args[0]
    if args:
        return {"args": list(args), **kwargs}
    return dict(kwargs) if kwargs else None


def _takes_positional(fn: Callable[..., Any]) -> bool:
    """True if ``fn`` accepts a positional argument."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in params)


class _ExceptionTracker:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._origin: dict[int, tuple[BaseException, str]] = {}

    def mark(self, exc: BaseException, event_id: str) -> None:
        """Remember the first event that recorded ``exc``."""
        with self._lock:
            self._origin.setdefault(id(exc), (exc, event_id))

    def origin_of(self, exc: BaseException) -> tuple[BaseException, str] | None:
        """The first recorded event for ``exc`` or for an exception it wraps
        (``raise X from e`` / an exception raised while handling ``e``)."""
        seen: set = set()
        e: BaseException | None = exc
        with self._lock:
            while e is not None and id(e) not in seen:
                seen.add(id(e))
                origin = self._origin.get(id(e))
                if origin is not None and origin[0] is e:
                    return origin
                e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        return None

    def propagation_meta(self, exc: BaseException) -> dict[str, Any]:
        """Metadata for an event recording ``exc`` again: empty the first time, else ``propagated`` and a
        pointer to the original event, so one failure is counted once."""
        origin = self.origin_of(exc)
        if origin is None:
            return {}
        return {"propagated": True, "origin_event_id": origin[1]}


class _Mailbox:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: dict[str, list[Message]] = {}

    def put(self, receiver: str, msg: Message) -> None:
        """Append ``msg`` to ``receiver``'s mailbox."""
        with self._lock:
            self._boxes.setdefault(receiver, []).append(msg)

    def drain(self, agent_id: str) -> list[Message]:
        """Remove and return ``agent_id``'s messages in arrival order."""
        with self._lock:
            return self._boxes.pop(agent_id, [])

    def peek(self, agent_id: str) -> list[Message]:
        """A copy of ``agent_id``'s waiting messages, leaving them in place."""
        with self._lock:
            return list(self._boxes.get(agent_id, []))


class RunContext:

    def __init__(self, task: Task, run_id: str | None = None, clock: Clock | None = None,
                 seed: int = 0, trial: int = 0, pricing: PricingTable | None = None,
                 capture_text: bool = True, max_text_chars: int = 20000) -> None:
        self.task = task
        self.clock = clock or WallClock()
        self.pricing = pricing or DEFAULT_PRICING
        self.seed = seed
        self.trial = trial
        self._rng = random.Random(seed)
        self._branch_rng: contextvars.ContextVar = contextvars.ContextVar(f"swarmeval_rng_{id(self)}", default=None)
        self.text_policy = TextPolicy(capture_text, max_text_chars)
        self.trajectory = Trajectory(run_id=run_id or new_id("run"), task_id=task.task_id, task=task,
                                     seed=seed, trial=trial)
        self.trajectory.metadata.update(self.text_policy.to_metadata())
        self.trajectory.metadata["schema_version"] = SCHEMA_VERSION
        self._lock = threading.RLock()
        self._mailbox = _Mailbox()
        self._errors = _ExceptionTracker()
        self._started = False
        self._finished = False

    @property
    def capture_text(self) -> bool:
        """Whether full text is kept in recorded data."""
        return self.text_policy.capture

    @property
    def max_text_chars(self) -> int:
        """Length above which recorded text is truncated."""
        return self.text_policy.max_chars

    @property
    def rng(self) -> random.Random:
        """Seeded randomness; each ``parallel``/``gather`` branch gets its own generator derived from its parent,
        so draws do not depend on thread scheduling."""
        return self._branch_rng.get() or self._rng

    def start(self) -> None:
        """Record the run start once."""
        if self._started:
            return
        self._started = True
        self.trajectory.started_at = self.now()
        self.record(EventType.RUN_START, content=preview(self.task.input))

    def finish(self, status: str = Status.SUCCESS.value, error: str | None = None) -> Trajectory:
        """Record the run end once with its status and return the trajectory."""
        if self._finished:
            return self.trajectory
        self._finished = True
        self.trajectory.ended_at = self.now()
        self.trajectory.status = status
        self.trajectory.error = error
        self.record(EventType.RUN_END, status=status, error=error,
                    latency_ms=self.trajectory.duration * 1000.0)
        return self.trajectory

    def set_output(self, output: Any, artifacts: Iterable[Artifact | str] = ()) -> None:
        """Set the run's final output and the artifacts it consists of."""
        self.trajectory.final_output = output
        self.trajectory.output_artifacts = _ids(artifacts)

    def now(self) -> float:
        """Current time on the run's clock."""
        return self.clock.now()

    def record(self, event_type: EventType | str, _span: AgentSpan | None = None,
               **fields: Any) -> Event:
        """Append an event.  Agent and span come from ``_span`` when given (the
        span whose method is recording), else from the active span."""
        span = _span if _span is not None else _current_span.get()
        et = event_type.value if isinstance(event_type, EventType) else event_type
        fields.setdefault("timestamp", self.now())
        if span is not None:
            fields.setdefault("agent_id", span.agent_id)
            fields.setdefault("span_id", span.span_id)
            fields.setdefault("parent_event_id", span.start_event_id)
        event_id = fields.pop("event_id", None) or new_id("ev")
        ev = Event(event_id=event_id, run_id=self.trajectory.run_id, task_id=self.task.task_id,
                   event_type=et, **fields)
        with self._lock:
            self.trajectory.events.append(ev)
        return ev

    def _text(self, obj: Any) -> Any:
        """Apply the run's text policy to a value before recording it."""
        return self.text_policy.apply(obj)

    def record_exception(self, exc: BaseException, _span: AgentSpan | None = None) -> Event:
        """Record an ERROR for ``exc``; later records of the same (or a wrapping) exception are marked
        ``propagated`` and point to the first, so one failure counts once at any nesting depth."""
        err = f"{type(exc).__name__}: {exc}"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]
        meta = self._errors.propagation_meta(exc)
        ev = self.record(EventType.ERROR, _span=_span, error=err,
                         content=self._text(tb) if self.capture_text else None, metadata=meta)
        if not meta:
            self._errors.mark(exc, ev.event_id)
        return ev

    def agent(self, agent_id: str, role: str | None = None, model: str | None = None,
              **metadata: Any) -> AgentSpan:
        """Open a span for one invocation of ``agent_id``; use it as a context manager."""
        return AgentSpan(self, agent_id, role, model, metadata)

    @property
    def current_span(self) -> AgentSpan | None:
        """The span active in this context, if any."""
        return _current_span.get()

    def send(self, sender: str, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Artifact | str] = (), _span: AgentSpan | None = None,
             reply_to: Message | str | None = None, **metadata: Any) -> Message:
        """Record a message and deliver it to the receiver's mailbox.
        ``reply_to`` (a Message or its id) marks it as an answer."""
        art_ids = _ids(artifacts)
        tokens = estimate_tokens(content) + sum(
            self.trajectory.artifacts[a].tokens for a in art_ids if a in self.trajectory.artifacts)
        reply_id = reply_to.message_id if isinstance(reply_to, Message) else reply_to
        msg_id = new_id("msg")
        span = _span if _span is not None else _current_span.get()
        if span is not None and span.agent_id != sender:
            span = None
        ev = self.record(EventType.MESSAGE, _span=span, agent_id=sender, span_id=span.span_id if span else None,
                         parent_event_id=span.start_event_id if span else None,
                         sender=sender, receiver=receiver, kind=kind, content=self._text(content),
                         content_tokens=tokens, content_hash=content_hash(content),
                         input_artifacts=art_ids, metadata=dict(metadata), message_id=msg_id, reply_to=reply_id)
        msg = Message(msg_id, sender, receiver, content, kind, art_ids, ev.timestamp, tokens, ev.event_id,
                      dict(metadata), reply_id)
        self._mailbox.put(receiver, msg)
        return msg

    def receive(self, agent_id: str, _span: AgentSpan | None = None) -> list[Message]:
        """Drain ``agent_id``'s mailbox, recording one MESSAGE_RECEIVE per message; its artifacts are listed
        as ``input_artifacts`` because they arrived, not because they were used."""
        msgs = self._mailbox.drain(agent_id)
        span = _span if _span is not None else _current_span.get()
        if span is not None and span.agent_id != agent_id:
            span = None
        for m in msgs:
            if isinstance(self.clock, SimClock) and self.clock.now() < m.timestamp:
                self.clock.set(m.timestamp)
            now = self.now()
            self.record(EventType.MESSAGE_RECEIVE, _span=span, agent_id=agent_id,
                        span_id=span.span_id if span else None,
                        parent_event_id=span.start_event_id if span else None, sender=m.sender, receiver=agent_id,
                        kind=m.kind, content_tokens=0, input_artifacts=list(m.artifacts),
                        message_id=m.message_id, reply_to=m.reply_to, timestamp=now,
                        metadata={"delay_s": max(0.0, now - m.timestamp)})
        return msgs

    def inbox(self, agent_id: str) -> list[Message]:
        """Peek at the mailbox without draining it; records nothing."""
        return self._mailbox.peek(agent_id)

    def _branch_seeds(self, n: int) -> list[int]:
        """Draw one seed per concurrent branch from the current generator."""
        rng = self.rng
        return [rng.getrandbits(64) for _ in range(n)]

    def _enter_branch(self, seed: int) -> None:
        """Give the current branch its own timeline and seeded generator."""
        self.clock.fork()
        self._branch_rng.set(random.Random(seed))

    def parallel(self, *fns: Callable[[], Any], max_workers: int | None = None) -> list[Any]:
        """Run callables concurrently in threads; each branch inherits the current span and gets its own
        SimClock timeline and seeded ``ctx.rng``, and the parent resumes when the last branch ends."""
        if not fns:
            return []
        seeds = self._branch_seeds(len(fns))
        ends = [self.now()] * len(fns)

        def branch(i: int, fn: Callable[[], Any]) -> Any:
            self._enter_branch(seeds[i])
            try:
                return fn()
            finally:
                ends[i] = self.now()

        contexts = [contextvars.copy_context() for _ in fns]
        try:
            with ThreadPoolExecutor(max_workers=max_workers or len(fns)) as ex:
                futures = [ex.submit(c.run, branch, i, fn) for i, (c, fn) in enumerate(zip(contexts, fns, strict=True))]
                return [f.result() for f in futures]
        finally:
            self.clock.join(max(ends))

    async def gather(self, *aws: Awaitable[Any]) -> list[Any]:
        """``asyncio.gather`` for agent coroutines, with the same per-branch
        timeline and randomness as ``parallel``."""
        import asyncio
        if not aws:
            return []
        seeds = self._branch_seeds(len(aws))
        ends = [self.now()] * len(aws)

        async def branch(i: int, aw: Awaitable[Any]) -> Any:
            self._enter_branch(seeds[i])
            try:
                return await aw
            finally:
                ends[i] = self.now()

        try:
            return list(await asyncio.gather(*(branch(i, aw) for i, aw in enumerate(aws))))
        finally:
            self.clock.join(max(ends))

    def cost_of(self, model: str | None, input_tokens: int, output_tokens: int,
                cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
        """Cost of one call under the run's pricing table, or None if the model is unpriced."""
        return self.pricing.cost(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)


class AgentSpan:

    def __init__(self, ctx: RunContext, agent_id: str, role: str | None, model: str | None,
                 metadata: dict[str, Any]) -> None:
        self.ctx = ctx
        self.agent_id = agent_id
        self.role = role
        self.model: str | None = model
        self.metadata = metadata
        self.span_id: str = ""
        self.start_event_id: str | None = None
        self.parent: AgentSpan | None = None
        self._token: Any = None
        self._closed = False
        self.span: Span | None = None

    def __enter__(self) -> AgentSpan:
        ctx = self.ctx
        self.parent = _current_span.get()
        self.span_id = new_id("span")
        start = ctx.now()
        ev = ctx.record(EventType.AGENT_START, agent_id=self.agent_id, span_id=self.span_id,
                        parent_event_id=self.parent.start_event_id if self.parent else None,
                        model=self.model, kind=self.role, metadata=dict(self.metadata), timestamp=start)
        self.start_event_id = ev.event_id
        self.span = Span(self.span_id, self.agent_id, self.role,
                         self.parent.span_id if self.parent else None, start, model=self.model,
                         start_event_id=ev.event_id, metadata=dict(self.metadata))
        outer = _current_call.get()
        if outer is not None:
            self.span.metadata["within_call"] = outer[1]
        with ctx._lock:
            ctx.trajectory.spans[self.span_id] = self.span
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._closed:
            return False
        if exc is not None:
            self.ctx.record_exception(exc, _span=self)
            self._close(Status.ERROR.value, f"{type(exc).__name__}: {exc}")
        else:
            self._close(Status.SUCCESS.value, None)
        return False

    async def __aenter__(self) -> AgentSpan:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def _close(self, status: str, error: str | None) -> None:
        """End the span once with ``status`` and record AGENT_END."""
        if self._closed:
            return
        self._closed = True
        end = self.ctx.now()
        if self.span is not None:
            self.span.end = end
            self.span.status = status
            self.span.error = error
        self.ctx.record(EventType.AGENT_END, agent_id=self.agent_id, span_id=self.span_id,
                        parent_event_id=self.start_event_id, status=status, error=error,
                        latency_ms=(end - self.span.start) * 1000.0 if self.span else None, timestamp=end)
        if self._token is not None:
            try:
                _current_span.reset(self._token)
            except ValueError:
                _current_span.set(self.parent)
            self._token = None

    @contextmanager
    def llm_call(self, model: str | None = None, input_tokens: int = 0, output_tokens: int = 0,
                 input_artifacts: Iterable[Artifact | str] = (), prompt: Any = None,
                 **metadata: Any) -> Iterator[LLMCall]:
        """Record one LLM call.  Set ``call.input_tokens`` / ``call.output_tokens``
        / ``call.output`` inside the block when they become known."""
        ctx = self.ctx
        call = LLMCall(model or self.model, input_tokens, output_tokens, _ids(input_artifacts), prompt)
        call.metadata.update(metadata)
        t0 = ctx.now()
        token = _current_llm_call.set(call)
        call_token = _enter_call(self, call)
        try:
            yield call
        except Exception as exc:
            call.metadata.update(ctx._errors.propagation_meta(exc))
            ev = self._record_llm(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            ctx._errors.mark(exc, ev.event_id)
            raise
        finally:
            _current_call.reset(call_token)
            _current_llm_call.reset(token)
        self._record_llm(call, t0, Status.SUCCESS.value, None)

    def _record_llm(self, call: LLMCall, t0: float, status: str, error: str | None) -> Event:
        """Record an LLM_CALL event for ``call`` started at ``t0``."""
        ctx = self.ctx
        end = ctx.now()
        meta = dict(call.metadata)
        if call.prompt is not None:
            meta["prompt"] = ctx._text(to_text(call.prompt))
            meta["prompt_hash"] = content_hash(call.prompt)
        tin, tout = int(call.input_tokens), int(call.output_tokens)
        cr, cw = int(call.cache_read_tokens), int(call.cache_write_tokens)
        return ctx.record(EventType.LLM_CALL, _span=self, model=call.model, input_tokens=tin, output_tokens=tout,
                          cache_read_tokens=cr, cache_write_tokens=cw,
                          cost_usd=None if meta.get("cost_unknown") else ctx.cost_of(call.model, tin, tout, cr, cw),
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.output) if call.output is not None else None,
                          content_tokens=estimate_tokens(call.output) if call.output is not None else 0,
                          content_hash=content_hash(call.output) if call.output is not None else None,
                          input_artifacts=list(call.input_artifacts), metadata=meta, timestamp=t0,
                          event_id=call.event_id)

    @staticmethod
    def _llm_args(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any],
                  prompt: Any) -> tuple[tuple[Any, ...], Any]:
        """Arguments for ``fn`` and the prompt to record: ``prompt=`` alone is passed when ``fn`` takes a
        positional argument, and without ``prompt=`` the arguments are recorded as the prompt."""
        if prompt is not None and not args and not kwargs and _takes_positional(fn):
            args = (prompt,)
        return args, prompt if prompt is not None else _call_record(args, kwargs)

    @staticmethod
    def _finish_llm(call: LLMCall, result: Any, input_tokens: int | None,
                    output_tokens: int | None) -> Any:
        """Fill ``call`` from ``fn``'s result, estimating missing tokens, and return the output text."""
        instrumented = bool(call.metadata.get("api_calls"))
        if isinstance(result, LLMResult):
            if not instrumented:
                call.input_tokens = result.input_tokens
                call.output_tokens = result.output_tokens
                call.cache_read_tokens = result.cache_read_tokens
                call.cache_write_tokens = result.cache_write_tokens
                if result.model:
                    call.model = result.model
            out = result.text
        else:
            out = result
        call.output = out
        if input_tokens is not None:
            call.input_tokens = input_tokens
        if output_tokens is not None:
            call.output_tokens = output_tokens
        if not call.input_tokens and not call.cache_read_tokens and call.prompt is not None:
            call.input_tokens = estimate_tokens(call.prompt)
            call.metadata["input_tokens_estimated"] = True
        if not call.output_tokens and out is not None:
            call.output_tokens = estimate_tokens(out)
            call.metadata["output_tokens_estimated"] = True
        return out

    def llm(self, fn: Callable[..., Any], *args: Any, model: str | None = None,
            input_artifacts: Iterable[Artifact | str] = (), prompt: Any = None,
            input_tokens: int | None = None, output_tokens: int | None = None,
            **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` as a recorded LLM call: an ``LLMResult`` supplies token usage and its
        ``text`` is returned, otherwise tokens are estimated from the prompt and output."""
        args, recorded = self._llm_args(fn, args, kwargs, prompt)
        with self.llm_call(model=model, input_artifacts=input_artifacts, prompt=recorded) as call:
            result = fn(*args, **kwargs)
            _reject_awaitable(result, "llm", "allm")
            return self._finish_llm(call, result, input_tokens, output_tokens)

    async def allm(self, fn: Callable[..., Any], *args: Any, model: str | None = None,
                   input_artifacts: Iterable[Artifact | str] = (), prompt: Any = None,
                   input_tokens: int | None = None, output_tokens: int | None = None,
                   **kwargs: Any) -> Any:
        """Async ``llm``: awaits ``fn(*args, **kwargs)`` so latency and usage are real."""
        args, recorded = self._llm_args(fn, args, kwargs, prompt)
        with self.llm_call(model=model, input_artifacts=input_artifacts, prompt=recorded) as call:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return self._finish_llm(call, result, input_tokens, output_tokens)

    @contextmanager
    def tool_call(self, name: str, args: Any = None,
                  input_artifacts: Iterable[Artifact | str] = ()) -> Iterator[ToolCall]:
        """Record one tool call; set ``call.result`` inside the block."""
        ctx = self.ctx
        call = ToolCall(name, args, _ids(input_artifacts))
        t0 = ctx.now()
        call_token = _enter_call(self, call)
        try:
            yield call
        except Exception as exc:
            call.metadata.update(ctx._errors.propagation_meta(exc))
            ev = self._record_tool(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            ctx._errors.mark(exc, ev.event_id)
            raise
        finally:
            _current_call.reset(call_token)
        self._record_tool(call, t0, Status.SUCCESS.value, None)

    def _record_tool(self, call: ToolCall, t0: float, status: str, error: str | None) -> Event:
        """Record a TOOL_CALL event for ``call`` started at ``t0``."""
        ctx = self.ctx
        end = ctx.now()
        return ctx.record(EventType.TOOL_CALL, _span=self, tool_name=call.name, tool_args=ctx._text(call.args),
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.result) if call.result is not None else None,
                          content_tokens=estimate_tokens(call.result) if call.result is not None else 0,
                          content_hash=content_hash(call.result) if call.result is not None else None,
                          input_artifacts=list(call.input_artifacts), timestamp=t0, event_id=call.event_id,
                          metadata={**call.metadata, "args_hash": content_hash(canonical_args(call.args))})

    def call_tool(self, name: str, fn: Callable[..., Any], *args: Any,
                  input_artifacts: Iterable[Artifact | str] = (), **kwargs: Any) -> Any:
        """Managed tool call: runs ``fn(*args, **kwargs)`` and records it."""
        recorded_args: Any = {"args": list(args), **kwargs} if args else dict(kwargs)
        with self.tool_call(name, recorded_args, input_artifacts=input_artifacts) as call:
            result = fn(*args, **kwargs)
            _reject_awaitable(result, "call_tool", "acall_tool")
            call.result = result
            return result

    async def acall_tool(self, name: str, fn: Callable[..., Any], *args: Any,
                         input_artifacts: Iterable[Artifact | str] = (), **kwargs: Any) -> Any:
        """Async ``call_tool``: awaits ``fn(*args, **kwargs)``."""
        recorded_args: Any = {"args": list(args), **kwargs} if args else dict(kwargs)
        with self.tool_call(name, recorded_args, input_artifacts=input_artifacts) as call:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            call.result = result
            return result

    def send(self, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Artifact | str] = (), reply_to: Message | str | None = None,
             **metadata: Any) -> Message:
        """Send a message from this agent; see ``RunContext.send``."""
        return self.ctx.send(self.agent_id, receiver, content, kind=kind, artifacts=artifacts, _span=self,
                             reply_to=reply_to, **metadata)

    def receive(self) -> list[Message]:
        """Drain this agent's mailbox, recording the receives."""
        return self.ctx.receive(self.agent_id, _span=self)

    def inbox(self) -> list[Message]:
        """Messages waiting for this agent, without receiving them."""
        return self.ctx.inbox(self.agent_id)

    def produce(self, name: str, content: Any, derived_from: Iterable[Artifact | str] | None = None,
                **metadata: Any) -> Artifact:
        """Record an artifact; ``derived_from`` lists its sources (``[]`` = none), and leaving it out records
        "unknown", which lineage then approximates from what the span consumed."""
        ctx = self.ctx
        sources = None if derived_from is None else _ids(derived_from)
        art = Artifact(new_id("art"), name, self.agent_id, self.span_id, ctx.now(), content,
                       content_hash(content), estimate_tokens(content), None, dict(metadata), sources)
        ev = ctx.record(EventType.ARTIFACT, _span=self, kind=name, content=ctx._text(content),
                        content_tokens=art.tokens, content_hash=art.content_hash,
                        input_artifacts=list(sources or []), output_artifacts=[art.artifact_id],
                        metadata=dict(metadata), timestamp=art.produced_at)
        art.event_id = ev.event_id
        with ctx._lock:
            ctx.trajectory.artifacts[art.artifact_id] = art
        return art

    def consume(self, artifact: Artifact | str, purpose: str | None = None) -> Artifact | None:
        """Record that this agent used ``artifact`` and return it."""
        art_id = artifact.artifact_id if isinstance(artifact, Artifact) else str(artifact)
        art = self.ctx.trajectory.artifacts.get(art_id)
        self.ctx.record(EventType.ARTIFACT_USE, _span=self, kind=art.name if art else None,
                        input_artifacts=[art_id], metadata={"purpose": purpose} if purpose else {})
        return art

    def wait(self, reason: str, seconds: float,
             on: WaitTarget | Iterable[WaitTarget] | None = None) -> None:
        """Record a wait.  ``on`` names what was awaited: Messages, Artifacts
        and/or agent ids, alone or in a list."""
        arts, meta = _wait_refs(on)
        t0 = self.ctx.now()
        self.ctx.clock.sleep(seconds)
        self.ctx.record(EventType.WAIT, _span=self, kind=reason, input_artifacts=arts, metadata=meta,
                        latency_ms=(self.ctx.now() - t0) * 1000.0, timestamp=t0)

    @contextmanager
    def waiting(self, reason: str, on: WaitTarget | Iterable[WaitTarget] | None = None) -> Iterator[None]:
        """Record the time spent inside the block as a wait for ``on``."""
        arts, meta = _wait_refs(on)
        t0 = self.ctx.now()
        try:
            yield
        finally:
            self.ctx.record(EventType.WAIT, _span=self, kind=reason, input_artifacts=arts, metadata=meta,
                            latency_ms=(self.ctx.now() - t0) * 1000.0, timestamp=t0)

    def note(self, text: str, **metadata: Any) -> Event:
        """Record a free-text annotation."""
        return self.ctx.record(EventType.ANNOTATION, _span=self, content=self.ctx._text(text), metadata=metadata)

    def error(self, message: str, **metadata: Any) -> Event:
        """Record an error message without raising."""
        return self.ctx.record(EventType.ERROR, _span=self, error=message, metadata=metadata)

    def sub_agent(self, agent_id: str, role: str | None = None, model: str | None = None,
                  **metadata: Any) -> AgentSpan:
        """Alias for ``ctx.agent`` that reads naturally when nesting."""
        return self.ctx.agent(agent_id, role=role, model=model, **metadata)
