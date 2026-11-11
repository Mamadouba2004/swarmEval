"""Human-readable text reports (terminal friendly)."""
from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any

from ..metrics.coordination import ANALYSES
from ..schema import EventType, Trajectory, preview
from ..stats import Summary
from ..stats import ranged as _ranged


def bar(frac: float, width: int = 16) -> str:
    """A text bar of ``width`` cells filled to ``frac``."""
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _money(x: float | None, complete: bool = True) -> str:
    """A cost in dollars, 'unknown' for None, with '*' when incomplete."""
    if x is None:
        return "unknown"
    s = f"${x:.2f}" if x >= 0.1 else f"${x:.4f}"
    return s if complete else s + "*"


def _pct(x: float) -> str:
    """A fraction as a percentage with one decimal."""
    return f"{x*100:.1f}%"


def _spread(s: Summary, unit: str = "", digits: int = 1) -> str:
    """A summary as its mean with spread."""
    return s.fmt(unit=unit, digits=digits)


def box(title: str, rows: Sequence[Any], width: int = 58) -> str:
    """Rows framed in a box that widens to fit them; tuples are (label, value[, flag])."""
    longest = max([len(r) + 3 for r in rows if isinstance(r, str)]
                  + [max(28, len(r[0])) + len(r[1]) + len(r[2] if len(r) > 2 else "") + 6
                     for r in rows if isinstance(r, tuple)] + [width])
    inner = max(width, longest) - 2
    out = ["╭" + "─" * inner + "╮", "│" + title.center(inner) + "│", "├" + "─" * inner + "┤"]
    for row in rows:
        if row is None:
            out.append("├" + "─" * inner + "┤")
            continue
        if isinstance(row, tuple):
            label, value = row[0], row[1]
            flag = row[2] if len(row) > 2 else ""
            lw = max(28, len(label))
            line = f" {label:<{lw}}{value:>{inner - 4 - lw}} {flag:<2}"
        else:
            line = f" {row}"
        out.append("│" + line[:inner].ljust(inner) + "│")
    out.append("╰" + "─" * inner + "╯")
    return "\n".join(out)


def _run_shape(result: Any) -> str:
    """Report header shape; an interrupted set shows finished/planned runs."""
    n_runs = len(result.runs)
    if result.runs.metadata.get("interrupted"):
        return f"{n_runs}/{result.runs.metadata.get('planned_runs', n_runs)} runs (interrupted)"
    perf = result.performance
    return f"{perf.n_tasks} tasks × {perf.trials} trials = {n_runs} runs"


def _per_trial(perf: Any) -> str:
    """Per-trial success rates, noting when an incomplete trial was left out."""
    return (", ".join(_pct(x) for x in perf.per_trial)
            + ("  (incomplete trial excluded)" if len(perf.per_trial) < perf.trials else ""))


def render_text(result: Any, show_agents: bool = True, show_tasks: bool = True,
                show_coordination: bool = True) -> str:
    """The evaluation as a text report."""
    perf, eff = result.performance, result.efficiency
    n_runs = len(result.runs)
    planned = result.runs.metadata.get("planned_runs", n_runs)
    rows: list[Any] = [
        f"{result.label}  ·  {result.benchmark}  ·  {_run_shape(result)}",
        None,
        ("Task Success", f"{perf.n_successes}/{n_runs}  ({_pct(perf.success_rate)})"),
        ("Mean Score", f"{perf.mean_score:.3f}"),
    ]
    if perf.constraint_pass_rate is not None:
        rows.append(("Constraint Pass", _pct(perf.constraint_pass_rate)))
    if perf.run_errors:
        rows.append(("Runs Crashed", f"{perf.run_errors}/{n_runs}", "⚠"))
    if perf.evaluator_errors:
        rows.append(("Evaluator Errors", f"{perf.evaluator_errors}/{n_runs}", "⚠"))
    if result.runs.metadata.get("interrupted"):
        rows.append(("Runs", f"interrupted after {n_runs}/{planned} runs", "⚠"))
    rows += [
        None,
        ("Cost / Run", _money(result.cost["per_run"], eff.cost_complete)),
        ("Cost / Successful Task", _money(eff.cost_per_success, eff.cost_complete)),
        ("Total Cost", _money(eff.cost_total, eff.cost_complete)),
        ("Tokens / Run", f"{eff.tokens.mean:,.0f}  "
                         f"(in {eff.input_tokens.mean:,.0f} / out {eff.output_tokens.mean:,.0f})"),
        ("LLM Calls / Run", f"{eff.llm_calls.mean:.1f}"),
        ("Tool Calls / Run", f"{eff.tool_calls.mean:.1f}"),
        ("Messages / Run", f"{eff.messages.mean:.1f}"),
        None,
        ("Wall Clock / Run", _spread(eff.wall_clock_s, "s")),
        ("Agent Time / Run", f"{eff.agent_time_s.mean:.1f}s"),
        ("Waiting Time / Run", f"{eff.waiting_time_s.mean:.1f}s"),
    ]
    if eff.errors.total:
        rows.append(("Errors (all runs)", f"{eff.errors.total:.0f}", "⚠"))
    parts = [box("SWARM EVALUATION", rows)]
    parts += ["  * " + note for note in cost_notes(eff)]
    if not eff.duration_complete:
        parts.append("  * " + durations_note(eff))
    if perf.trials > 1:
        parts.append("  per-trial success: " + _per_trial(perf))

    if show_agents and eff.per_agent:
        parts.append("")
        parts.append("AGENTS (per run, mean)")
        parts.append(f"  {'agent':<16}{'role':<12}{'tokens':<24}{'llm':>5}{'tools':>6}{'msgs':>6}"
                     f"{'time':>8}{'cost':>10}")
        for aid, u in sorted(eff.per_agent.items(), key=lambda kv: -kv[1]["token_share"]):
            parts.append(f"  {aid:<16}{(u.get('role') or '-')[:11]:<12}{bar(u['token_share'])} "
                         f"{u['token_share']*100:>4.0f}%"
                         f"{u['llm_calls']:>5.1f}{u['tool_calls']:>6.1f}{u['messages_sent']:>6.1f}"
                         f"{u['self_time_s']:>7.1f}s{_money(u['cost_usd'], u.get('cost_complete', True)):>10}")

    if show_tasks and perf.per_task:
        parts.append("")
        parts.append("TASKS")
        parts.append(f"  {'task':<16}{'success':>9}{'score':>8}{'cost':>10}{'latency':>9}")
        for tid, t in perf.per_task.items():
            parts.append(f"  {tid:<16}{_pct(t['success_rate']):>9}{t['score']:>8.2f}"
                         f"{_money(t['cost_usd'], t.get('cost_complete', True)):>10}"
                         f"{t['latency_s']:>8.1f}s")
    if show_coordination and (getattr(result, "coordination", None) is not None
                              or getattr(result, "coordination_errors", None)):
        parts.append("")
        parts.append(render_coordination(result))
    parts.append("")
    parts.append(f"Evaluators: {', '.join(perf.evaluators) or 'none'}")
    return "\n".join(parts)


_LABEL_W, _VALUE_W = 46, 30
_SEV_MARK = {"high": "⚠ HIGH", "medium": "⚠ MED ", "low": "· LOW "}
CONTRIBUTION_NOTES = (
    "lineage = runs where its artifacts reached the final output; consumed = an artifact used by another agent;",
    "dead = nothing it sent was used (only over runs with complete provenance); coverage = provenance coverage",
)


def _row(label: str, value: str) -> str:
    """One fixed-width label and value row."""
    return f"{label:<{_LABEL_W}}{value:>{_VALUE_W}}"


def _wrap(prefix: str, text: str, indent: int, width: int = 104) -> list[str]:
    """Text wrapped after ``prefix`` with a hanging indent."""
    return textwrap.wrap(text, width=width, initial_indent=prefix, subsequent_indent=" " * indent) or [prefix]


def coordination_rows(cs: Any) -> list[tuple[str, str] | None]:
    """``(label, value)`` rows of the COORDINATION box, ``None`` between groups, each label carrying its
    definition; the text and HTML reports both render them."""
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    rows: list[tuple[str, str] | None] = []
    for a in ANALYSES:
        summary = cs.get(a.name)
        if summary is not None:
            rows += a.rows(summary)
    rows += [None,
             ("Dependency Coverage (receive events)", _ranged(cs.message_receive_coverage, pct)),
             ("Derivation Coverage (derived_from)", _ranged(cs.derivation_coverage, pct))]
    return rows


def cost_notes(eff: Any) -> list[str]:
    """Why the cost is incomplete, one note per cause (shared by the text and HTML reports)."""
    out = []
    if eff.cost_complete:
        return out
    if eff.unpriced_calls or not eff.streamed_cost_calls:
        out.append("cost incomplete: some models have no pricing entry (unpriced calls are excluded, not counted "
                   "as $0; re-price with `swarmeval report --pricing`)")
    if eff.streamed_cost_calls:
        out.append(f"cost incomplete: {eff.streamed_cost_calls} streamed call(s) have no usage, so their cost is "
                   "unknown")
    return out


def durations_note(eff: Any) -> str:
    """The note saying how many streamed calls have an unknown duration."""
    return (f"durations incomplete: {eff.unknown_duration_calls} streamed call(s) recorded with unknown duration "
            "(latency, work and critical path undercount)")


def coordination_caveats(cs: Any, eff: Any = None) -> list[tuple[bool, str]]:
    """``(is_warning, text)`` notes that qualify the COORDINATION numbers
    (shared by the text and HTML reports); ``eff`` is the efficiency summary."""
    out: list[tuple[bool, str]] = []
    if eff is not None and not eff.duration_complete:
        out.append((True, durations_note(eff)))
    cov = cs.message_receive_coverage
    if cov is not None and cov.mean < 1.0:
        out.append((True, f"receive events recorded for {cov.mean * 100:.0f}% of messages: the remaining message "
                          "edges are inferred from timing; dependencies may be over- or under-counted"))
    dep = cs.dependency
    if dep is not None and dep.unrecorded_s is not None and dep.unrecorded_s.mean > 0.05:
        out.append((False, "unrecorded time is idle or untracked work, not parallel headroom"))
    return out


def _failed_row(result: Any) -> tuple:
    """The COORDINATION box row naming how many runs' analysis failed and the first error."""
    errors = result.coordination_errors
    failed = f"{len(errors)}/{len(result.runs)} runs: {next(iter(errors.values()))}"
    return ("Analysis Failed", textwrap.shorten(failed, _LABEL_W + _VALUE_W - 30, placeholder="…"), "⚠")


def render_coordination(result: Any) -> str:
    """COORDINATION box, OBSERVED CONTRIBUTION table and DIAGNOSTICS list of an EvaluationResult; values are
    means over runs with the range when runs differ, and every ratio is named with its definition."""
    cs = result.coordination
    errors = result.coordination_errors
    if cs is None:
        return box("COORDINATION", [_failed_row(result)], width=_LABEL_W + _VALUE_W + 4) if errors else ""
    rows: list[Any] = [f"{cs.n_runs} runs  ·  values are means over runs (range)", None]
    if errors:
        rows += [_failed_row(result), None]
    rows += [None if r is None else _row(*r) for r in coordination_rows(cs)]
    parts = [box("COORDINATION", rows, width=_LABEL_W + _VALUE_W + 4)]
    parts += [("  ⚠ " if warn else "  ") + text for warn, text in coordination_caveats(cs, result.efficiency)]

    if cs.contribution:
        aw = max([16] + [len(a) + 1 for a in cs.contribution])
        parts += ["", "OBSERVED CONTRIBUTION (provenance, not marginal effect)",
                  f"  {'agent':<{aw}}{'lineage':>9}{'consumed':>10}{'dead':>9}{'coverage':>10}"
                  "  utilization (tokens)"]
        util = cs.utilization
        for aid, c in cs.contribution.items():
            cov_s = c.get("provenance_coverage")
            dead = c.get("dead_rate")
            share = util.get(aid, {}).get("token_share")
            dead_s = "unknown" if dead is None else f"{dead * 100:.0f}%"
            cov_txt = "unknown" if cov_s is None else f"{cov_s['mean'] * 100:.0f}%"
            parts.append(f"  {aid:<{aw}}{c['lineage_rate'] * 100:>8.0f}%{c['consumed_rate'] * 100:>9.0f}%"
                         f"{dead_s:>9}{cov_txt:>10}  "
                         + (f"{bar(share)} {share * 100:>3.0f}%" if share is not None else "unknown"))
        parts += ["  " + note for note in CONTRIBUTION_NOTES]

    diag = cs.diagnostics
    if diag is not None:
        parts += ["", "DIAGNOSTICS (hypotheses to test, not verdicts)"]
        if not diag.smells:
            parts.append("  no smells at the current thresholds")
        for sp in diag.smells:
            agents = ", ".join(f"{a} {n}" for a, n in list(sp.agents.items())[:4])
            head = (f"  {_SEV_MARK.get(sp.max_severity, sp.max_severity)}  {sp.smell}  ·  "
                    f"{sp.runs_affected}/{sp.n_runs} runs ({sp.prevalence * 100:.0f}%)")
            if agents:
                head += f"  ·  {agents}"
            if sp.inferred_runs:
                head += f"  ·  inferred in {sp.inferred_runs}"
            parts.append(head)
            parts += _wrap("      Observed:   ", sp.observation, 18)
            parts += _wrap("      Hypothesis: ", sp.hypothesis, 18)
            parts += _wrap("      Test:       ", sp.intervention, 18)
    return "\n".join(parts)


def _fmt(x: float | None, unit: str) -> str:
    """A metric value formatted for its unit, 'unknown' for None."""
    if x is None:
        return "unknown"
    if unit == "pct":
        return _pct(x)
    if unit == "usd":
        return _money(x)
    if unit == "s":
        return f"{x:.1f}s"
    if unit == "int":
        return f"{x:,.0f}"
    return f"{x:.2f}"


def _fmt_delta(m: Any) -> str:
    """A metric delta formatted for its unit, with the relative change."""
    d = m.delta
    if d is None:
        return "–"
    if m.unit == "pct":
        s = f"{d*100:+.1f} pts"
    elif m.unit == "usd":
        s = f"{'+' if d >= 0 else '-'}{_money(abs(d))}"
    elif m.unit == "s":
        s = f"{d:+.1f}s"
    elif m.unit == "int":
        s = f"{d:+,.0f}"
    else:
        s = f"{d:+.2f}"
    rel = m.relative
    return s + (f" ({rel*100:+.0f}%)" if rel is not None and m.unit != "pct" else "")


_MARK = {"improved": "▲", "regressed": "▼", "unchanged": "=", "unknown": "?", "changed": "~"}


def render_comparison(cmp: Any, max_rows: int = 20) -> str:
    """Side-by-side text report of a ``compare()`` result."""
    b, c = cmp.baseline, cmp.candidate
    lines = [f"COMPARISON  {b.label} → {c.label}  ·  {b.benchmark}  ·  {len(cmp.shared_tasks)} tasks, "
             f"{cmp.paired_runs} paired runs", ""]
    lines.append(f"  {'metric':<24}{b.label[:14]:>14}{c.label[:14]:>14}  {'change':<20}")
    for m in cmp.metrics:
        lines.append(f"  {m.label:<24}{_fmt(m.baseline, m.unit):>14}{_fmt(m.candidate, m.unit):>14}  "
                     f"{_fmt_delta(m):<20}{_MARK[m.verdict]}")
    lines.append("")
    if cmp.is_tradeoff:
        lines.append("  TRADEOFF  better: " + ", ".join(cmp.improved) + "  ·  worse: " + ", ".join(cmp.regressed))
    elif cmp.improved:
        lines.append("  better on: " + ", ".join(cmp.improved) + "  (no regressions)")
    elif cmp.regressed:
        lines.append("  worse on: " + ", ".join(cmp.regressed) + "  (no improvements)")
    else:
        lines.append("  no measurable difference")
    gained, lost = cmp.flips["gained"], cmp.flips["lost"]
    changed_neutral = [m.key for m in cmp.metrics if m.verdict == "changed"]
    if changed_neutral:
        lines.append("  changed (no better direction): " + ", ".join(changed_neutral))
    lines.append(f"  paired outcomes: {len(gained)} fail→pass, {len(lost)} pass→fail "
                 f"(of {cmp.paired_runs} runs with the same task and trial)")

    changed = [(tid, t) for tid, t in cmp.per_task.items() if t["success_rate"][0] != t["success_rate"][1]]
    if changed:
        lines += ["", "TASKS WITH CHANGED SUCCESS"]
        lines.append(f"  {'task':<16}{'success':>18}{'cost':>22}")
        for tid, t in changed[:max_rows]:
            (sb, sc), (cb, cc) = t["success_rate"], t["cost_usd"]
            lines.append(f"  {tid:<16}{_pct(sb):>8} → {_pct(sc):<7}{_money(cb):>10} → {_money(cc):<9}")
        if len(changed) > max_rows:
            lines.append(f"  … {len(changed) - max_rows} more")

    if cmp.per_agent:
        lines += ["", "AGENTS (per run, mean)"]
        lines.append(f"  {'agent':<16}{'status':<9}{'tokens':>24}{'cost':>24}")
        for aid, a in list(cmp.per_agent.items())[:max_rows]:
            tb, tc = a["tokens"]
            cb, cc = a["cost_usd"]
            tok = f"{'-' if tb is None else f'{tb:,.0f}'} → {'-' if tc is None else f'{tc:,.0f}'}"
            before = _money(cb) if a['status'] != 'added' else '-'
            after = _money(cc) if a['status'] != 'removed' else '-'
            cost = f"{before} → {after}"
            lines.append(f"  {aid:<16}{a['status']:<9}{tok:>24}{cost:>24}")

    if cmp.warnings:
        lines.append("")
        lines += [f"  ! {w}" for w in cmp.warnings]
    lines += ["", "  Differences are measured, not tested for significance (Phase 3)."]
    return "\n".join(lines)


def render_trajectory(traj: Trajectory, max_events: int = 200) -> str:
    """Plain-text listing of one trajectory: what happened, in order."""
    t0 = traj.started_at or (traj.events[0].timestamp if traj.events else 0.0)
    ev_status = "✓" if traj.succeeded else "✗"
    head = (f"run {traj.run_id}  task {traj.task_id}  trial {traj.trial}  status {traj.status}  "
            f"eval {ev_status} {traj.score:.2f}  {traj.duration:.1f}s  {traj.total_tokens:,} tokens  "
            f"{_money(traj.cost_usd, traj.cost_is_complete)}")
    lines = [head, f"  {'t(s)':>7}  {'event':<12}{'agent':<16}detail"]
    depth = {None: 0}
    for ev in traj.events[:max_events]:
        if ev.event_type == EventType.AGENT_START.value:
            parent = traj.spans[ev.span_id].parent_span_id if ev.span_id in traj.spans else None
            depth[ev.span_id] = depth.get(parent, 0) + 1
        boundary = ev.event_type in (EventType.AGENT_START.value, EventType.AGENT_END.value)
        indent = "  " * max(0, depth.get(ev.span_id, 0) - (1 if boundary else 0))
        et = ev.event_type
        if et == EventType.LLM_CALL.value:
            cached = f" ({ev.cache_read_tokens} cached)" if ev.cache_read_tokens else ""
            detail = (f"{ev.model or '?'}  {ev.prompt_tokens}{cached}→{ev.output_tokens} tok  "
                      f"{ev.duration_s:.2f}s  {_money(ev.cost_usd)}")
        elif et == EventType.TOOL_CALL.value:
            detail = f"{ev.tool_name}({preview(ev.tool_args, 60)})  {ev.duration_s:.2f}s"
        elif et == EventType.MESSAGE.value:
            detail = f"{ev.sender} → {ev.receiver}  [{ev.kind}]  {ev.content_tokens} tok"
        elif et == EventType.MESSAGE_RECEIVE.value:
            detail = f"reads {ev.sender} → {ev.receiver}  [{ev.kind}]  after {ev.metadata.get('delay_s', 0.0):.2f}s"
        elif et == EventType.ARTIFACT.value:
            detail = f"{ev.kind}  {ev.content_tokens} tok"
        elif et == EventType.ARTIFACT_USE.value:
            detail = f"uses {ev.kind}"
        elif et == EventType.AGENT_START.value:
            detail = f"role={ev.kind or '-'} model={ev.model or '-'}"
        elif et == EventType.AGENT_END.value:
            detail = f"{ev.status}  {ev.duration_s:.2f}s"
        elif et == EventType.WAIT.value:
            detail = f"{ev.kind}  {ev.duration_s:.2f}s"
        elif et == EventType.EVAL.value:
            detail = f"{ev.content}  score={ev.metadata.get('score')}"
        else:
            detail = ev.error or preview(ev.content, 80)
        flag = "" if ev.status == "success" else f" [{ev.status}]"
        lines.append(f"  {ev.timestamp - t0:>7.2f}  {et:<12}{indent}{(ev.agent_id or '-'):<16}{detail}{flag}")
    if len(traj.events) > max_events:
        lines.append(f"  … {len(traj.events) - max_events} more events")
    return "\n".join(lines)
