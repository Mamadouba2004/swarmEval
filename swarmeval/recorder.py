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
"""
from __future__ import annotations

import contextvars
import inspect
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .pricing import DEFAULT_PRICING, PricingTable
from .schema import (SCHEMA_VERSION, Artifact, Event, EventType, Span, Status, Task, TextPolicy, Trajectory,
                     canonical_args, content_hash, estimate_tokens, new_id, preview, to_text)


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #
class Clock:
    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def fork(self) -> None:
        """Called at the start of a concurrent branch (inside its context)."""

    def join(self, end: float) -> None:
        """Called in the parent after its branches finish; ``end`` is the
        latest branch end time."""


class WallClock(Clock):
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock(Clock):
    """A virtual clock.  Simulated swarms advance it explicitly so that
    demos and tests run instantly while still producing realistic timings.

    Concurrent branches started with ``ctx.parallel`` / ``ctx.gather`` each
    get their own timeline (held in a context variable), and the parent
    resumes at the latest branch end -- so simulated parallel work overlaps
    instead of adding up."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()
        self._branch: contextvars.ContextVar = contextvars.ContextVar(f"simclock_{id(self)}", default=None)

    def now(self) -> float:
        box = self._branch.get()
        return box[0] if box is not None else self._t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        box = self._branch.get()
        if box is not None:
            box[0] += max(0.0, float(seconds))
            return
        with self._lock:
            self._t += max(0.0, float(seconds))

    def set(self, t: float) -> None:
        box = self._branch.get()
        if box is not None:
            box[0] = float(t)
            return
        with self._lock:
            self._t = float(t)

    def fork(self) -> None:
        self._branch.set([self.now()])

    def join(self, end: float) -> None:
        self.set(max(self.now(), end))


# --------------------------------------------------------------------------- #
# Value objects handed to swarm code
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    message_id: str
    sender: str
    receiver: str
    content: Any
    kind: str
    artifacts: List[str]
    timestamp: float
    tokens: int
    event_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None     # message_id of the message this one answers


@dataclass
class LLMResult:
    """Return this from a function passed to ``AgentSpan.llm`` to report real
    token usage."""
    text: Any
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None
    raw: Any = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMCall:
    """Mutable handle yielded by ``AgentSpan.llm_call``."""

    def __init__(self, model: Optional[str], input_tokens: int, output_tokens: int,
                 input_artifacts: List[str], prompt: Any) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.input_artifacts = input_artifacts
        self.prompt = prompt
        self.output: Any = None
        self.metadata: Dict[str, Any] = {}
        self.event_id = new_id("ev")   # fixed up front so calls nested inside can point to it


class ToolCall:
    """Mutable handle yielded by ``AgentSpan.tool_call``."""

    def __init__(self, name: str, args: Any, input_artifacts: List[str]) -> None:
        self.name = name
        self.args = args
        self.input_artifacts = input_artifacts
        self.result: Any = None
        self.metadata: Dict[str, Any] = {}
        self.event_id = new_id("ev")


_current_span: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_span", default=None)
# The LLM call being recorded right now, so client instrumentation can fill
# it in instead of recording the same API call a second time.
_current_llm_call: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_llm_call", default=None)
# (span_id, event_id) of the innermost LLM/tool call running in this context.
# A call started inside another call of the same span records the outer one as
# ``metadata["within_call"]``: the graph needs to tell nesting (a tool run by
# an LLM call) from ``gather``ed sibling calls, whose intervals can look alike.
_current_call: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_call", default=None)


def _enter_call(span: "AgentSpan", call: Union["LLMCall", "ToolCall"]) -> contextvars.Token:
    outer = _current_call.get()
    if outer is not None and outer[0] == span.span_id:
        call.metadata["within_call"] = outer[1]
    return _current_call.set((span.span_id, call.event_id))


def _ids(artifacts: Iterable[Union[Artifact, str]]) -> List[str]:
    return [a.artifact_id if isinstance(a, Artifact) else str(a) for a in artifacts]


WaitTarget = Union[Message, Artifact, str]


def _wait_refs(on: Union[WaitTarget, Iterable[WaitTarget], None]) -> Tuple[List[str], Dict[str, Any]]:
    """What a wait was for: artifact ids (recorded as ``input_artifacts``) and
    metadata naming the awaited messages and agents.  A plain string is an
    agent id.  Only non-empty keys are written, so a wait without ``on`` is
    recorded exactly as in Phase 1."""
    if on is None:
        return [], {}
    items = [on] if isinstance(on, (Message, Artifact, str)) else list(on)
    arts: List[str] = []
    msgs: List[str] = []
    agents: List[str] = []
    for x in items:
        if isinstance(x, Message):
            msgs.append(x.message_id)
        elif isinstance(x, Artifact):
            arts.append(x.artifact_id)
        else:
            agents.append(str(x))
    meta: Dict[str, Any] = {}
    if msgs:
        meta["on_messages"] = msgs
    if agents:
        meta["on_agents"] = agents
    return arts, meta


def _reject_awaitable(result: Any, sync_name: str, async_name: str) -> None:
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()  # avoid a "never awaited" warning; the error says what to do
        raise TypeError(f"{sync_name}() got an awaitable from an async function; "
                        f"use `await span.{async_name}(...)` instead")


def _call_record(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """How a function's arguments are recorded (prompt or tool args)."""
    if len(args) == 1 and not kwargs:
        return args[0]
    if args:
        return {"args": list(args), **kwargs}
    return dict(kwargs) if kwargs else None


def _takes_positional(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in params)


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #
class RunContext:
    """Everything a swarm needs during one run: recording, messaging,
    seeded randomness and the clock."""

    def __init__(self, task: Task, run_id: Optional[str] = None, clock: Optional[Clock] = None,
                 seed: int = 0, trial: int = 0, pricing: Optional[PricingTable] = None,
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
        # Marks data recorded with message ids, receive events and derived_from;
        # absent on Phase 1 trajectories.
        self.trajectory.metadata["schema_version"] = SCHEMA_VERSION
        self._lock = threading.RLock()
        self._mailbox: Dict[str, List[Message]] = {}
        self._error_origin: Dict[int, Tuple[BaseException, str]] = {}
        self._started = False
        self._finished = False

    @property
    def capture_text(self) -> bool:
        return self.text_policy.capture

    @property
    def max_text_chars(self) -> int:
        return self.text_policy.max_chars

    @property
    def rng(self) -> random.Random:
        """Seeded randomness.  Each ``parallel``/``gather`` branch gets its own
        generator derived from its parent, so draws do not depend on thread
        scheduling."""
        return self._branch_rng.get() or self._rng

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.trajectory.started_at = self.now()
        self.record(EventType.RUN_START, content=preview(self.task.input))

    def finish(self, status: str = Status.SUCCESS.value, error: Optional[str] = None) -> Trajectory:
        if self._finished:
            return self.trajectory
        self._finished = True
        self.trajectory.ended_at = self.now()
        self.trajectory.status = status
        self.trajectory.error = error
        self.record(EventType.RUN_END, status=status, error=error,
                    latency_ms=self.trajectory.duration * 1000.0)
        return self.trajectory

    def set_output(self, output: Any, artifacts: Iterable[Union[Artifact, str]] = ()) -> None:
        self.trajectory.final_output = output
        self.trajectory.output_artifacts = _ids(artifacts)

    def now(self) -> float:
        return self.clock.now()

    # -- recording ------------------------------------------------------------
    def record(self, event_type: Union[EventType, str], _span: Optional["AgentSpan"] = None,
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

    def annotate(self, note: str, **metadata: Any) -> Event:
        return self.record(EventType.ANNOTATION, content=self._text(note), metadata=metadata)

    def _text(self, obj: Any) -> Any:
        return self.text_policy.apply(obj)

    # -- exceptions -----------------------------------------------------------
    def _mark_exception(self, exc: BaseException, event_id: str) -> None:
        with self._lock:
            self._error_origin.setdefault(id(exc), (exc, event_id))

    def _origin_of(self, exc: BaseException) -> Optional[Tuple[BaseException, str]]:
        """The first recorded event for ``exc`` or for an exception it wraps
        (``raise X from e`` / an exception raised while handling ``e``)."""
        seen: set = set()
        e: Optional[BaseException] = exc
        with self._lock:
            while e is not None and id(e) not in seen:
                seen.add(id(e))
                origin = self._error_origin.get(id(e))
                if origin is not None and origin[0] is e:
                    return origin
                e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        return None

    def _propagation_meta(self, exc: BaseException) -> Dict[str, Any]:
        """Metadata for an event that records ``exc`` again: empty for the first
        record, else ``propagated`` plus a pointer to the original event.  Used
        for ERROR events and for failed LLM/tool calls that an exception passes
        through (a tool calling a tool), so the metrics count it once."""
        origin = self._origin_of(exc)
        if origin is None:
            return {}
        return {"propagated": True, "origin_event_id": origin[1]}

    def record_exception(self, exc: BaseException, _span: Optional["AgentSpan"] = None) -> Event:
        """Record an ERROR for ``exc``.  The first record of an exception counts
        as the error; later ones (the same exception, or one wrapping it,
        leaving enclosing spans or the run) are marked ``propagated`` and point
        to the original, so one failure is counted once regardless of nesting
        depth or re-raising."""
        err = f"{type(exc).__name__}: {exc}"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]
        meta = self._propagation_meta(exc)
        ev = self.record(EventType.ERROR, _span=_span, error=err,
                         content=self._text(tb) if self.capture_text else None, metadata=meta)
        if not meta:
            self._mark_exception(exc, ev.event_id)
        return ev

    # -- spans ----------------------------------------------------------------
    def agent(self, agent_id: str, role: Optional[str] = None, model: Optional[str] = None,
              **metadata: Any) -> "AgentSpan":
        return AgentSpan(self, agent_id, role, model, metadata)

    @property
    def current_span(self) -> Optional["AgentSpan"]:
        return _current_span.get()

    # -- messaging ------------------------------------------------------------
    def send(self, sender: str, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Union[Artifact, str]] = (), _span: Optional["AgentSpan"] = None,
             reply_to: Union[Message, str, None] = None, **metadata: Any) -> Message:
        """Record a message and deliver it to the receiver's mailbox.
        ``reply_to`` (a Message or its id) marks it as an answer."""
        art_ids = _ids(artifacts)
        tokens = estimate_tokens(content) + sum(
            self.trajectory.artifacts[a].tokens for a in art_ids if a in self.trajectory.artifacts)
        reply_id = reply_to.message_id if isinstance(reply_to, Message) else reply_to
        # The id is made before recording so the MESSAGE event carries it and
        # the receive event can be matched to it.
        msg_id = new_id("msg")
        span = _span if _span is not None else _current_span.get()
        ev = self.record(EventType.MESSAGE, _span=span, agent_id=sender, span_id=span.span_id if span else None,
                         sender=sender, receiver=receiver, kind=kind, content=self._text(content),
                         content_tokens=tokens, content_hash=content_hash(content),
                         input_artifacts=art_ids, metadata=dict(metadata), message_id=msg_id, reply_to=reply_id)
        msg = Message(msg_id, sender, receiver, content, kind, art_ids, ev.timestamp, tokens, ev.event_id,
                      dict(metadata), reply_id)
        with self._lock:
            self._mailbox.setdefault(receiver, []).append(msg)
        return msg

    def receive(self, agent_id: str, _span: Optional["AgentSpan"] = None) -> List[Message]:
        """Drain the mailbox of ``agent_id``, recording one MESSAGE_RECEIVE per
        message.  The receive event lists the message's artifacts as
        ``input_artifacts`` because they arrived, not because they were used:
        use is recorded by ``consume`` or by an LLM/tool call's inputs."""
        with self._lock:
            msgs = self._mailbox.get(agent_id, [])
            self._mailbox[agent_id] = []
        span = _span if _span is not None else _current_span.get()
        if span is not None and span.agent_id != agent_id:
            span = None  # another agent's span is active: the read is not part of it
        for m in msgs:
            now = self.now()
            # delay is clamped: with threads and a wall clock the read can be
            # stamped marginally before the send
            self.record(EventType.MESSAGE_RECEIVE, _span=span, agent_id=agent_id,
                        span_id=span.span_id if span else None, sender=m.sender, receiver=agent_id,
                        kind=m.kind, content_tokens=0, input_artifacts=list(m.artifacts),
                        message_id=m.message_id, reply_to=m.reply_to, timestamp=now,
                        metadata={"delay_s": max(0.0, now - m.timestamp)})
        return msgs

    def inbox(self, agent_id: str) -> List[Message]:
        """Peek at the mailbox without draining it; records nothing."""
        with self._lock:
            return list(self._mailbox.get(agent_id, []))

    # -- concurrency ----------------------------------------------------------
    def _branch_seeds(self, n: int) -> List[int]:
        rng = self.rng
        return [rng.getrandbits(64) for _ in range(n)]

    def _enter_branch(self, seed: int) -> None:
        self.clock.fork()
        self._branch_rng.set(random.Random(seed))

    def parallel(self, *fns: Callable[[], Any], max_workers: Optional[int] = None) -> List[Any]:
        """Run callables concurrently in threads.  Each branch inherits the
        current span (so nested ``ctx.agent`` calls are parented correctly),
        gets its own timeline on a ``SimClock`` and its own seeded ``ctx.rng``;
        the parent resumes when the last branch ends."""
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
                futures = [ex.submit(c.run, branch, i, fn) for i, (c, fn) in enumerate(zip(contexts, fns))]
                return [f.result() for f in futures]
        finally:
            self.clock.join(max(ends))

    async def gather(self, *aws: Awaitable[Any]) -> List[Any]:
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

    # -- cost -----------------------------------------------------------------
    def cost_of(self, model: Optional[str], input_tokens: int, output_tokens: int,
                cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> Optional[float]:
        return self.pricing.cost(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)


# --------------------------------------------------------------------------- #
# Agent span
# --------------------------------------------------------------------------- #
class AgentSpan:
    """Context manager (sync or async) for one agent invocation."""

    def __init__(self, ctx: RunContext, agent_id: str, role: Optional[str], model: Optional[str],
                 metadata: Dict[str, Any]) -> None:
        self.ctx = ctx
        self.agent_id = agent_id
        self.role = role
        self.model: Optional[str] = model
        self.metadata = metadata
        self.span_id: str = ""
        self.start_event_id: Optional[str] = None
        self.parent: Optional["AgentSpan"] = None
        self._token: Any = None
        self._closed = False
        self.span: Optional[Span] = None

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "AgentSpan":
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

    async def __aenter__(self) -> "AgentSpan":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def _close(self, status: str, error: Optional[str]) -> None:
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
            except ValueError:  # token created in a different context (threads)
                _current_span.set(self.parent)
            self._token = None

    # -- LLM calls ------------------------------------------------------------
    @contextmanager
    def llm_call(self, model: Optional[str] = None, input_tokens: int = 0, output_tokens: int = 0,
                 input_artifacts: Iterable[Union[Artifact, str]] = (), prompt: Any = None,
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
            call.metadata.update(ctx._propagation_meta(exc))
            ev = self._record_llm(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            ctx._mark_exception(exc, ev.event_id)
            raise
        finally:
            _current_call.reset(call_token)
            _current_llm_call.reset(token)
        self._record_llm(call, t0, Status.SUCCESS.value, None)

    def _record_llm(self, call: LLMCall, t0: float, status: str, error: Optional[str]) -> Event:
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
                          cost_usd=ctx.cost_of(call.model, tin, tout, cr, cw),
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.output) if call.output is not None else None,
                          content_tokens=estimate_tokens(call.output) if call.output is not None else 0,
                          content_hash=content_hash(call.output) if call.output is not None else None,
                          input_artifacts=list(call.input_artifacts), metadata=meta, timestamp=t0,
                          event_id=call.event_id)

    @staticmethod
    def _llm_args(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any],
                  prompt: Any) -> Tuple[Tuple[Any, ...], Any]:
        """Arguments for ``fn`` and the prompt to record.  ``prompt=`` alone is
        passed to ``fn`` when it takes a positional argument, and without
        ``prompt=`` the arguments themselves are recorded as the prompt."""
        if prompt is not None and not args and not kwargs and _takes_positional(fn):
            args = (prompt,)
        return args, prompt if prompt is not None else _call_record(args, kwargs)

    @staticmethod
    def _finish_llm(call: LLMCall, result: Any, input_tokens: Optional[int],
                    output_tokens: Optional[int]) -> Any:
        instrumented = bool(call.metadata.get("api_calls"))  # usage already filled by instrument_client
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

    def llm(self, fn: Callable[..., Any], *args: Any, model: Optional[str] = None,
            input_artifacts: Iterable[Union[Artifact, str]] = (), prompt: Any = None,
            input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
            **kwargs: Any) -> Any:
        """Managed LLM call: runs ``fn(*args, **kwargs)`` and records it.  If
        ``fn`` returns an ``LLMResult`` its token usage is recorded and
        ``result.text`` returned; otherwise token counts are estimated from the
        prompt (``prompt=``, or ``fn``'s arguments) and the output."""
        args, recorded = self._llm_args(fn, args, kwargs, prompt)
        with self.llm_call(model=model, input_artifacts=input_artifacts, prompt=recorded) as call:
            result = fn(*args, **kwargs)
            _reject_awaitable(result, "llm", "allm")
            return self._finish_llm(call, result, input_tokens, output_tokens)

    async def allm(self, fn: Callable[..., Any], *args: Any, model: Optional[str] = None,
                   input_artifacts: Iterable[Union[Artifact, str]] = (), prompt: Any = None,
                   input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                   **kwargs: Any) -> Any:
        """Async ``llm``: awaits ``fn(*args, **kwargs)`` so latency and usage are real."""
        args, recorded = self._llm_args(fn, args, kwargs, prompt)
        with self.llm_call(model=model, input_artifacts=input_artifacts, prompt=recorded) as call:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return self._finish_llm(call, result, input_tokens, output_tokens)

    # -- tool calls -----------------------------------------------------------
    @contextmanager
    def tool_call(self, name: str, args: Any = None,
                  input_artifacts: Iterable[Union[Artifact, str]] = ()) -> Iterator[ToolCall]:
        ctx = self.ctx
        call = ToolCall(name, args, _ids(input_artifacts))
        t0 = ctx.now()
        call_token = _enter_call(self, call)
        try:
            yield call
        except Exception as exc:
            call.metadata.update(ctx._propagation_meta(exc))
            ev = self._record_tool(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            ctx._mark_exception(exc, ev.event_id)
            raise
        finally:
            _current_call.reset(call_token)
        self._record_tool(call, t0, Status.SUCCESS.value, None)

    def _record_tool(self, call: ToolCall, t0: float, status: str, error: Optional[str]) -> Event:
        ctx = self.ctx
        end = ctx.now()
        return ctx.record(EventType.TOOL_CALL, _span=self, tool_name=call.name, tool_args=ctx._text(call.args),
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.result) if call.result is not None else None,
                          content_tokens=estimate_tokens(call.result) if call.result is not None else 0,
                          content_hash=content_hash(call.result) if call.result is not None else None,
                          input_artifacts=list(call.input_artifacts), timestamp=t0, event_id=call.event_id,
                          # hashed in full: tool_args may be a preview or truncated
                          metadata={**call.metadata, "args_hash": content_hash(canonical_args(call.args))})

    def call_tool(self, name: str, fn: Callable[..., Any], *args: Any,
                  input_artifacts: Iterable[Union[Artifact, str]] = (), **kwargs: Any) -> Any:
        """Managed tool call: runs ``fn(*args, **kwargs)`` and records it."""
        recorded_args: Any = {"args": list(args), **kwargs} if args else dict(kwargs)
        with self.tool_call(name, recorded_args, input_artifacts=input_artifacts) as call:
            result = fn(*args, **kwargs)
            _reject_awaitable(result, "call_tool", "acall_tool")
            call.result = result
            return result

    async def acall_tool(self, name: str, fn: Callable[..., Any], *args: Any,
                         input_artifacts: Iterable[Union[Artifact, str]] = (), **kwargs: Any) -> Any:
        """Async ``call_tool``: awaits ``fn(*args, **kwargs)``."""
        recorded_args: Any = {"args": list(args), **kwargs} if args else dict(kwargs)
        with self.tool_call(name, recorded_args, input_artifacts=input_artifacts) as call:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            call.result = result
            return result

    # -- messaging ------------------------------------------------------------
    def send(self, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Union[Artifact, str]] = (), reply_to: Union[Message, str, None] = None,
             **metadata: Any) -> Message:
        return self.ctx.send(self.agent_id, receiver, content, kind=kind, artifacts=artifacts, _span=self,
                             reply_to=reply_to, **metadata)

    def receive(self) -> List[Message]:
        return self.ctx.receive(self.agent_id, _span=self)

    def inbox(self) -> List[Message]:
        return self.ctx.inbox(self.agent_id)

    # -- artifacts ------------------------------------------------------------
    def produce(self, name: str, content: Any, derived_from: Optional[Iterable[Union[Artifact, str]]] = None,
                **metadata: Any) -> Artifact:
        """Record an artifact.  ``derived_from`` lists the artifacts it was made
        from (``[]`` = from none); leaving it out records "unknown", which the
        lineage analysis then approximates from what the span consumed."""
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

    def consume(self, artifact: Union[Artifact, str], purpose: Optional[str] = None) -> Optional[Artifact]:
        art_id = artifact.artifact_id if isinstance(artifact, Artifact) else str(artifact)
        art = self.ctx.trajectory.artifacts.get(art_id)
        self.ctx.record(EventType.ARTIFACT_USE, _span=self, kind=art.name if art else None,
                        input_artifacts=[art_id], metadata={"purpose": purpose} if purpose else {})
        return art

    # -- waiting --------------------------------------------------------------
    def wait(self, reason: str, seconds: float,
             on: Union[WaitTarget, Iterable[WaitTarget], None] = None) -> None:
        """Record a wait.  ``on`` names what was awaited: Messages, Artifacts
        and/or agent ids, alone or in a list."""
        arts, meta = _wait_refs(on)
        t0 = self.ctx.now()
        self.ctx.clock.sleep(seconds)
        self.ctx.record(EventType.WAIT, _span=self, kind=reason, input_artifacts=arts, metadata=meta,
                        latency_ms=(self.ctx.now() - t0) * 1000.0, timestamp=t0)

    @contextmanager
    def waiting(self, reason: str, on: Union[WaitTarget, Iterable[WaitTarget], None] = None) -> Iterator[None]:
        arts, meta = _wait_refs(on)
        t0 = self.ctx.now()
        try:
            yield
        finally:
            self.ctx.record(EventType.WAIT, _span=self, kind=reason, input_artifacts=arts, metadata=meta,
                            latency_ms=(self.ctx.now() - t0) * 1000.0, timestamp=t0)

    # -- misc -----------------------------------------------------------------
    def note(self, text: str, **metadata: Any) -> Event:
        return self.ctx.record(EventType.ANNOTATION, _span=self, content=self.ctx._text(text), metadata=metadata)

    def error(self, message: str, **metadata: Any) -> Event:
        return self.ctx.record(EventType.ERROR, _span=self, error=message, metadata=metadata)

    def sub_agent(self, agent_id: str, role: Optional[str] = None, model: Optional[str] = None,
                  **metadata: Any) -> "AgentSpan":
        """Alias for ``ctx.agent`` that reads naturally when nesting."""
        return self.ctx.agent(agent_id, role=role, model=model, **metadata)
