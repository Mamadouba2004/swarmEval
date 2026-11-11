"""Model pricing (USD per 1M tokens) used to estimate API cost.

Unknown models produce a cost of ``None`` rather than a silent zero so that
reports can say "cost unknown" instead of "cost $0.00".  Supply your own
table via ``PricingTable(custom={...})`` or ``PricingTable.load(path)``.

DEFAULT_PRICES follows https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-13). A price is
(input, output) or (input, output, cache_read) in USD per 1M tokens; without a cache-read rate, reads cost
CACHE_READ_MULTIPLIER x input, and 5-minute cache writes cost CACHE_WRITE_MULTIPLIER x input. Claude Haiku 5.5
is omitted because its price depends on prompt length; add it via ``PricingTable(custom=...)`` for your tier.
A provider or region prefix ("anthropic/", "us.anthropic.") and a date or version suffix ("-20260101",
"@20260101", "-v1:0") do not change a model's price.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

Price = tuple[float, ...]

DEFAULT_PRICES: dict[str, Price] = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-mythos-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

_PREFIX = re.compile(r"^(?:.*/|(?:[a-z]{2}\.)?anthropic\.)")
_SUFFIX = re.compile(r"(?:[-@]\d{8}|-v\d+(?::\d+)?)+$")


def canonical_model(model: str) -> str:
    """A model id with provider prefixes and date or version suffixes removed."""
    return _SUFFIX.sub("", _PREFIX.sub("", model.strip()))


class PricingTable:
    def __init__(self, custom: dict[str, Sequence[float]] | None = None,
                 include_defaults: bool = True, cache_write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                 cache_read_multiplier: float = CACHE_READ_MULTIPLIER) -> None:
        self.prices: dict[str, Price] = dict(DEFAULT_PRICES) if include_defaults else {}
        if custom:
            self.prices.update({k: tuple(float(x) for x in v) for k, v in custom.items()})
        self.cache_write_multiplier = cache_write_multiplier
        self.cache_read_multiplier = cache_read_multiplier

    def lookup(self, model: str | None) -> Price | None:
        """Price for ``model``, or ``None`` if unlisted; only dated or provider-prefixed forms of a listed id match,
        so a model that merely shares a prefix is unknown rather than mispriced."""
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        return self.prices.get(canonical_model(model))

    def cost(self, model: str | None, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
        """USD cost of one call, cache reads and writes included; None if the model is unpriced."""
        p = self.lookup(model)
        if p is None:
            return None
        inp, out = p[0], p[1]
        read = p[2] if len(p) > 2 else inp * self.cache_read_multiplier
        total = (input_tokens * inp + output_tokens * out + cache_read_tokens * read
                 + cache_write_tokens * inp * self.cache_write_multiplier)
        return total / 1_000_000.0

    @classmethod
    def load(cls, path: str, include_defaults: bool = True) -> PricingTable:
        """Read ``{"model": [input, output(, cache_read)], ...}`` (USD per 1M) from JSON."""
        import json
        with open(path) as f:
            return cls(json.load(f), include_defaults=include_defaults)


DEFAULT_PRICING = PricingTable()
