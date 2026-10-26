"""Optional Anthropic SDK integration.

* ``instrument_client(client)`` -- wraps ``client.messages.create`` and
  ``client.beta.messages.create`` so that any call made inside an active
  ``ctx.agent(...)`` span is recorded automatically with real token usage
  (including prompt-cache reads and writes).  Calls made inside
  ``span.llm`` / ``span.llm_call`` fill in that call instead of being
  recorded a second time.  Works with ``Anthropic`` and ``AsyncAnthropic``.
* ``anthropic_llm(client, model)`` -- returns ``fn(prompt) -> LLMResult``
  (``async`` for an async client) for use with ``AgentSpan.llm`` /
  ``AgentSpan.allm``.

The ``anthropic`` package is only imported when a client is not supplied.
Calls from threads that do not carry the span's context (a bare
``ThreadPoolExecutor``) cannot be attributed; use ``ctx.parallel``.
Streaming (``stream=True``) is not instrumented yet: such calls pass through unrecorded.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from ..recorder import LLMCall, LLMResult, _current_llm_call, _current_span

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _text_of(response: Any) -> str:
    if getattr(response, "stop_reason", None) == "refusal":
        return ""
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _usage(response: Any) -> tuple:
    """(input, output, cache_read, cache_write) tokens from ``response.usage``."""
    u = getattr(response, "usage", None)

    def n(name: str) -> int:
        return int(getattr(u, name, 0) or 0)

    return (n("input_tokens"), n("output_tokens"), n("cache_read_input_tokens"),
            n("cache_creation_input_tokens"))


def _client(client: Any) -> Any:
    if client is not None:
        return client
    import anthropic  # local import: optional dependency
    return anthropic.Anthropic()


def _is_async(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn)


def anthropic_llm(client: Any = None, model: str = DEFAULT_MODEL, max_tokens: int = 16000,
                  system: Optional[str] = None, fallbacks: bool = True, **extra: Any) -> Callable[..., Any]:
    """Build a prompt -> LLMResult callable backed by the Messages API.

    With ``fallbacks=True`` (default) requests use the beta server-side
    fallback so a safety refusal is retried on a fallback model.  For an
    async client the callable is ``async`` (use it with ``span.allm``)."""
    c = _client(client)

    def request(prompt: str) -> tuple:
        kwargs: dict = {"model": model, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}], **extra}
        if system:
            kwargs["system"] = system
        if fallbacks:
            return c.beta.messages.create, dict(kwargs, betas=[FALLBACK_BETA], fallbacks="default")
        return c.messages.create, kwargs

    def result(resp: Any) -> LLMResult:
        tin, tout, cr, cw = _usage(resp)
        return LLMResult(_text_of(resp), tin, tout, getattr(resp, "model", model), resp,
                         cache_read_tokens=cr, cache_write_tokens=cw)

    create, _ = request("")
    if _is_async(create):
        async def acall(prompt: str) -> LLMResult:
            fn, kwargs = request(prompt)
            return result(await fn(**kwargs))
        return acall

    def call(prompt: str) -> LLMResult:
        fn, kwargs = request(prompt)
        return result(fn(**kwargs))

    return call


def _fill(call: LLMCall, resp: Any) -> None:
    """Add one API response's usage to ``call`` (a call may span several)."""
    tin, tout, cr, cw = _usage(resp)
    call.input_tokens += tin
    call.output_tokens += tout
    call.cache_read_tokens += cr
    call.cache_write_tokens += cw
    call.model = getattr(resp, "model", None) or call.model
    if call.output is None:
        call.output = _text_of(resp)
    call.metadata["api_calls"] = call.metadata.get("api_calls", 0) + 1


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    if _is_async(original):
        async def acreate(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return await original(*args, **kwargs)
            active = _current_llm_call.get()
            if active is not None:
                resp = await original(*args, **kwargs)
                _fill(active, resp)
                return resp
            span = _current_span.get()
            if span is None:
                return await original(*args, **kwargs)
            with span.llm_call(model=kwargs.get("model"), prompt=kwargs.get("messages")) as call:
                resp = await original(*args, **kwargs)
                _fill(call, resp)
                return resp
        acreate.__swarmeval_wrapped__ = True  # type: ignore[attr-defined]
        return acreate

    def create(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return original(*args, **kwargs)
        active = _current_llm_call.get()
        if active is not None:
            resp = original(*args, **kwargs)
            _fill(active, resp)
            return resp
        span = _current_span.get()
        if span is None:
            return original(*args, **kwargs)
        with span.llm_call(model=kwargs.get("model"), prompt=kwargs.get("messages")) as call:
            resp = original(*args, **kwargs)
            _fill(call, resp)
            return resp
    create.__swarmeval_wrapped__ = True  # type: ignore[attr-defined]
    return create


def instrument_client(client: Any) -> Any:
    """Patch ``messages.create`` and ``beta.messages.create`` to record calls
    made inside an active agent span.  Idempotent; returns the same client."""
    targets = [getattr(client, "messages", None)]
    beta = getattr(client, "beta", None)
    targets.append(getattr(beta, "messages", None) if beta is not None else None)
    for messages in targets:
        original = getattr(messages, "create", None) if messages is not None else None
        if original is None or getattr(original, "__swarmeval_wrapped__", False):
            continue
        messages.create = _wrap(original)
    return client
