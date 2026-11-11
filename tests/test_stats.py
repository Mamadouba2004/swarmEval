"""Tests for the descriptive statistics helpers."""
import pytest

from swarmeval import stats


def test_descriptives():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert stats.mean(xs) == 3.0
    assert stats.stdev(xs) == pytest.approx(1.5811, abs=1e-3)
    assert stats.mean([]) == 0.0 and stats.stdev([1.0]) == 0.0


def test_summary():
    s = stats.summarize([2.0, 4.0])
    assert s.n == 2 and s.mean == 3.0 and s.minimum == 2.0 and s.maximum == 4.0 and s.total == 6.0
    assert "±" in s.fmt("s")
    assert stats.summarize([]).n == 0
    assert stats.summarize([0.5]).fmt(pct=True) == "50.0%"


def test_summary_dict_serializes_nested_summaries():
    class Holder:
        def __init__(self):
            self.n_runs = 2
            self.tokens = stats.summarize([1.0, 3.0])
            self.ratio = None
            self.agents = ["a"]

    assert stats.summary_dict(Holder()) == {"n_runs": 2, "tokens": stats.summarize([1.0, 3.0]).to_dict(),
                                            "ratio": None, "agents": ["a"]}
