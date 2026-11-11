"""Small, dependency-free descriptive statistics.

Phase 1 reports means and spreads over repeated runs.  Confidence intervals
and paired comparisons are Phase 3 work (see ROADMAP.md).

Classes:
- Summary: Descriptive summary of one metric over a set of runs.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def mean(xs: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for no values."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation, 0.0 for fewer than two values."""
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


@dataclass
class Summary:
    n: int
    mean: float
    std: float
    minimum: float
    maximum: float
    total: float

    def to_dict(self) -> dict[str, float]:
        """The summary as a JSON-ready dict."""
        return {"n": self.n, "mean": self.mean, "std": self.std,
                "min": self.minimum, "max": self.maximum, "total": self.total}

    def fmt(self, unit: str = "", pct: bool = False, digits: int = 1) -> str:
        """The mean with its spread when runs differ, as a number or percentage."""
        if pct:
            s = f"{self.mean*100:.{digits}f}%"
        else:
            s = f"{self.mean:.{digits}f}{unit}"
        if self.n > 1 and self.std > 0:
            s += f" ±{self.std*100:.{digits}f}" if pct else f" ±{self.std:.{digits}f}"
        return s


def summarize(values: Sequence[float]) -> Summary:
    """Count, mean, spread, range and total of the values."""
    values = [float(v) for v in values]
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Summary(len(values), mean(values), stdev(values), min(values), max(values), sum(values))


def known_summary(values: Sequence[float | None]) -> Summary | None:
    """Summary of the defined values, None when there are none."""
    known = [v for v in values if v is not None]
    return summarize(known) if known else None
