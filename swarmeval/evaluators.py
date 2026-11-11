"""Evaluators: deterministic judgments of task outcomes.

Every evaluator implements ``evaluate(task, output, trajectory) -> Evaluation``
and, where serialisable, ``to_spec()`` so benchmarks can be stored as JSON.
LLM-based judging is planned for a later phase; use ``CallableEvaluator`` to
plug in your own judge in the meantime.

Classes:
- ContainsAll: Success if every keyword appears in the output and no forbidden one does. Score is the fraction of
  required keywords found (minus forbidden hits).
- CallableEvaluator: Wrap any ``fn(task, output, trajectory) -> bool | float | Evaluation``.
- ToolUseEvaluator: Checks tool-use correctness from the trajectory: expected tools were called, forbidden tools
  were not, and no tool call errored.
- Constraints: All sub-evaluators must pass (hard constraints).
- Composite: Weighted combination of evaluators. Success requires the weighted score to reach ``threshold`` and
  every ``required`` evaluator to pass.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .schema import Evaluation, Task, Trajectory, to_text
from .stats import mean


class Evaluator:
    name: str = "evaluator"

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:  # pragma: no cover
        """Judge ``output`` for ``task``, optionally using its trajectory."""
        raise NotImplementedError

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name}

    def check(self, task: Task) -> None:
        """Raise ``ValueError`` if this evaluator cannot judge ``task`` (e.g. it
        has nothing to compare against).  Called by ``Benchmark.validate``."""


class ExactMatch(Evaluator):
    name = "exact_match"

    def __init__(self, expected: Any = None, normalize: bool = True) -> None:
        self.expected = expected
        self.normalize = normalize

    def _norm(self, x: Any) -> str:
        """Text form, lowercased with whitespace collapsed when normalizing."""
        s = to_text(x)
        return " ".join(s.lower().split()) if self.normalize else s

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if the output equals the expected value."""
        expected = self.expected if self.expected is not None else task.expected
        ok = self._norm(output) == self._norm(expected)
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"expected": expected})

    def check(self, task: Task) -> None:
        """Require an expected value in the spec or on the task."""
        if self.expected is None and task.expected is None:
            raise ValueError("exact_match needs 'expected' (in the spec or on the task)")

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "expected": self.expected, "normalize": self.normalize}


class _KeywordEvaluator(Evaluator):

    def __init__(self, keywords: Sequence[str] | None = None, forbidden: Sequence[str] = (),
                 case_insensitive: bool = True) -> None:
        self.keywords = list(keywords) if keywords is not None else None
        self.forbidden = list(forbidden)
        self.case_insensitive = case_insensitive

    def _keywords(self, task: Task) -> list[str]:
        """Keywords from the spec, else from the task's expected value."""
        keywords = self.keywords if self.keywords is not None else (
            task.expected if isinstance(task.expected, list) else [task.expected])
        return [str(k) for k in keywords if k is not None]

    def _match(self, task: Task, output: Any) -> tuple[list[str], list[str], list[str], list[str]]:
        """The keywords, those found, those missing and the forbidden ones present in ``output``."""
        keywords = self._keywords(task)
        text = to_text(output)
        if self.case_insensitive:
            text_cmp = text.lower()
            found = [k for k in keywords if k.lower() in text_cmp]
            bad = [f for f in self.forbidden if f.lower() in text_cmp]
        else:
            found = [k for k in keywords if k in text]
            bad = [f for f in self.forbidden if f in text]
        return keywords, found, [k for k in keywords if k not in found], bad

    def check(self, task: Task) -> None:
        """Require at least one keyword, so the evaluator cannot pass any output."""
        if not self._keywords(task):
            raise ValueError(f"{self.name} has no keywords (set 'keywords' in the spec or a list "
                             f"'expected' on the task); it would pass any output")

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "keywords": self.keywords, "forbidden": self.forbidden,
                "case_insensitive": self.case_insensitive}


class ContainsAll(_KeywordEvaluator):
    name = "contains_all"

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if every keyword appears and no forbidden one does; score is the share found."""
        keywords, found, missing, bad = self._match(task, output)
        frac = len(found) / len(keywords) if keywords else 1.0
        score = max(0.0, frac - 0.5 * len(bad) / max(1, len(self.forbidden) or 1)) if bad else frac
        ok = len(found) == len(keywords) and not bad
        return Evaluation(ok, score, self.name, {"found": found, "missing": missing, "forbidden_hits": bad},
                          constraints_passed=not bad)


class ContainsAny(_KeywordEvaluator):
    name = "contains_any"

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if any keyword appears and no forbidden one does."""
        _, found, missing, bad = self._match(task, output)
        ok = bool(found) and not bad
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"found": found, "missing": missing,
                                                             "forbidden_hits": bad}, constraints_passed=not bad)


class Regex(Evaluator):
    name = "regex"

    DEFAULT_FLAGS = int(re.IGNORECASE | re.DOTALL)

    def __init__(self, pattern: str, flags: int = DEFAULT_FLAGS) -> None:
        self.pattern = pattern
        self.flags = int(flags)
        try:
            re.compile(pattern, self.flags)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from None

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if the pattern matches the output."""
        ok = re.search(self.pattern, to_text(output), self.flags) is not None
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"pattern": self.pattern})

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec, flags included."""
        return {"type": self.name, "pattern": self.pattern, "flags": self.flags}


class NumericTolerance(Evaluator):
    name = "numeric"

    def __init__(self, expected: float | None = None, tolerance: float = 1e-6, relative: bool = False) -> None:
        self.expected = expected
        self.tolerance = tolerance
        self.relative = relative

    _NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

    def check(self, task: Task) -> None:
        """Require a numeric expected value."""
        expected = self.expected if self.expected is not None else task.expected
        try:
            float(expected)
        except (TypeError, ValueError):
            raise ValueError(f"numeric needs a numeric 'expected' (got {expected!r})") from None

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if a number in the output is within tolerance of the expected value."""
        expected = self.expected if self.expected is not None else task.expected
        nums = self._NUMBER.findall(to_text(output))
        ok = False
        value: float | None = None
        if nums and expected is not None:
            try:
                exp = float(expected)
                for n in reversed(nums):
                    value = float(n.replace(",", ""))
                    tol = self.tolerance * abs(exp) if self.relative else self.tolerance
                    if abs(value - exp) <= tol:
                        ok = True
                        break
            except ValueError:
                pass
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"expected": expected, "parsed": value})

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "expected": self.expected, "tolerance": self.tolerance, "relative": self.relative}


class CallableEvaluator(Evaluator):
    name = "callable"

    def __init__(self, fn: Callable[..., Any], label: str = "callable") -> None:
        self.fn = fn
        self.name = label

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Call the function and turn a bool or score into an Evaluation."""
        r = self.fn(task, output, trajectory)
        if isinstance(r, Evaluation):
            return r
        if isinstance(r, bool):
            return Evaluation(r, 1.0 if r else 0.0, self.name)
        score = float(r)
        return Evaluation(score >= 0.5, score, self.name)

    def to_spec(self) -> dict[str, Any]:
        """A spec marked unserializable, so reloading fails clearly."""
        return {"type": self.name, "unserializable": True}


class ToolUseEvaluator(Evaluator):
    name = "tool_use"

    def __init__(self, expected_tools: Sequence[str] = (), forbidden_tools: Sequence[str] = (),
                 allow_errors: bool = False) -> None:
        self.expected_tools = list(expected_tools)
        self.forbidden_tools = list(forbidden_tools)
        self.allow_errors = allow_errors

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Check that expected tools succeeded, forbidden ones were never attempted, and calls did not fail."""
        if trajectory is None:
            return Evaluation(False, 0.0, self.name, {"reason": "no trajectory"})
        calls = trajectory.tool_calls()
        used = {c.tool_name for c in calls if c.status == "success"}
        attempted = {c.tool_name for c in calls}
        missing = [t for t in self.expected_tools if t not in used]
        forbidden = [t for t in self.forbidden_tools if t in attempted]
        errors = [c for c in calls if c.status == "error" and not c.metadata.get("propagated")]
        checks = [not missing, not forbidden, self.allow_errors or not errors]
        score = sum(checks) / len(checks)
        return Evaluation(all(checks), score, self.name,
                          {"missing": missing, "forbidden": forbidden, "errors": len(errors)})

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "expected_tools": self.expected_tools,
                "forbidden_tools": self.forbidden_tools, "allow_errors": self.allow_errors}


class Constraints(Evaluator):
    name = "constraints"

    def __init__(self, checks: Sequence[Evaluator]) -> None:
        self.checks = list(checks)

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed only if every check passes; score is their mean."""
        results = [(c.name, c.evaluate(task, output, trajectory)) for c in self.checks]
        passed = all(r.success for _, r in results)
        score = mean([r.score for _, r in results]) if results else 1.0
        return Evaluation(passed, score, self.name, {n: r.to_dict() for n, r in results}, constraints_passed=passed)

    def check(self, task: Task) -> None:
        """Check every sub-evaluator."""
        for c in self.checks:
            c.check(task)

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "checks": [c.to_spec() for c in self.checks]}


class Composite(Evaluator):
    name = "composite"

    def __init__(self, evaluators: Sequence[Evaluator], weights: Sequence[float] | None = None,
                 threshold: float = 0.5, required: Sequence[str] = ()) -> None:
        self.evaluators = list(evaluators)
        if weights and len(weights) != len(self.evaluators):
            raise ValueError(f"composite got {len(weights)} weights for {len(self.evaluators)} evaluators")
        self.weights = list(weights) if weights else [1.0] * len(self.evaluators)
        self.threshold = threshold
        self.required = list(required)

    def evaluate(self, task: Task, output: Any, trajectory: Trajectory | None = None) -> Evaluation:
        """Succeed if the weighted score reaches the threshold and every required evaluator passes."""
        results = [(e.name, e.evaluate(task, output, trajectory)) for e in self.evaluators]
        total_w = sum(self.weights) or 1.0
        score = sum(w * r.score for w, (_, r) in zip(self.weights, results, strict=True)) / total_w
        req_ok = all(r.success for n, r in results if n in self.required)
        ok = score >= self.threshold and req_ok
        return Evaluation(ok, score, self.name, {n: r.to_dict() for n, r in results})

    def check(self, task: Task) -> None:
        """Check every sub-evaluator."""
        for e in self.evaluators:
            e.check(task)

    def to_spec(self) -> dict[str, Any]:
        """The evaluator as a JSON spec."""
        return {"type": self.name, "evaluators": [e.to_spec() for e in self.evaluators],
                "weights": self.weights, "threshold": self.threshold, "required": self.required}


_REGISTRY: dict[str, Callable[[dict[str, Any]], Evaluator]] = {}
_KEYS: dict[str, set[str] | None] = {}


def register(name: str, factory: Callable[[dict[str, Any]], Evaluator],
             keys: Iterable[str] | None = None) -> None:
    """Register a spec ``type``.  ``keys`` lists the spec keys the factory
    reads; other keys are rejected so typos fail loudly.  ``None`` accepts any."""
    _REGISTRY[name] = factory
    _KEYS[name] = set(keys) if keys is not None else None


register("exact_match", lambda s: ExactMatch(s.get("expected"), s.get("normalize", True)),
         ["expected", "normalize"])
register("contains_all",
         lambda s: ContainsAll(s.get("keywords"), s.get("forbidden", ()), s.get("case_insensitive", True)),
         ["keywords", "forbidden", "case_insensitive"])
register("contains_any",
         lambda s: ContainsAny(s.get("keywords"), s.get("forbidden", ()), s.get("case_insensitive", True)),
         ["keywords", "forbidden", "case_insensitive"])
register("regex", lambda s: Regex(s["pattern"], s.get("flags", Regex.DEFAULT_FLAGS)), ["pattern", "flags"])
register("numeric", lambda s: NumericTolerance(s.get("expected"), s.get("tolerance", 1e-6), s.get("relative", False)),
         ["expected", "tolerance", "relative"])
register("tool_use",
         lambda s: ToolUseEvaluator(s.get("expected_tools", ()), s.get("forbidden_tools", ()),
                                    s.get("allow_errors", False)),
         ["expected_tools", "forbidden_tools", "allow_errors"])
register("constraints", lambda s: Constraints([from_spec(c) for c in s.get("checks", [])]), ["checks"])
register("composite", lambda s: Composite([from_spec(e) for e in s.get("evaluators", [])], s.get("weights"),
                                          s.get("threshold", 0.5), s.get("required", ())),
         ["evaluators", "weights", "threshold", "required"])


def from_spec(spec: Any) -> Evaluator:
    """Build an evaluator from an instance, a spec dict, or a bare type name."""
    if isinstance(spec, Evaluator):
        return spec
    if callable(spec) and not isinstance(spec, dict):
        return CallableEvaluator(spec)
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"cannot build evaluator from {spec!r}")
    t = spec["type"]
    if spec.get("unserializable"):
        raise ValueError(f"evaluator {t!r} was a Python object that cannot be rebuilt from JSON; "
                         f"pass it again in code or register() a spec type for it")
    if t not in _REGISTRY:
        raise ValueError(f"unknown evaluator type {t!r}; known: {sorted(_REGISTRY)}")
    allowed = _KEYS.get(t)
    if allowed is not None:
        unknown = sorted(k for k in spec if k != "type" and k not in allowed)
        if unknown:
            raise ValueError(f"evaluator {t!r}: unknown key(s) {unknown}; known: {sorted(allowed)}")
    try:
        return _REGISTRY[t](spec)
    except KeyError as exc:
        raise ValueError(f"evaluator {t!r}: missing required key {exc}") from None


def resolve(task: Task, *fallbacks: Any, override: Any = None) -> Evaluator | None:
    """Pick the evaluator for ``task``: an explicit ``override`` (e.g. passed to
    ``evaluate()``) wins, then the task's own, then each fallback in order."""
    for candidate in (override, task.evaluator, *fallbacks):
        if candidate is not None:
            return from_spec(candidate)
    return None
