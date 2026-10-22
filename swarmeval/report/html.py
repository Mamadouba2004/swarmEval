"""Standalone HTML report (Phase 2: visualization).

One self-contained page per ``EvaluationResult``: the Phase 1 health
dimensions (performance, efficiency), the COORDINATION numbers as tiles, an
SVG agent graph and an SVG timeline of one run with its critical path, the
observed-contribution table and the diagnostics.  Dimensions are shown side
by side; there is deliberately no overall score.

The page has no scripts and no external resources (fonts, stylesheets,
images), so it can be mailed, archived or opened offline.  Agent ids, task
ids, labels and smell text come from user code and trajectories, so every
one of them goes through ``_e`` -- in HTML text, SVG text and attributes.
Numbers and labels are the ones of the text report (``coordination_rows``,
``coordination_caveats``), so the two reports never disagree.
"""
from __future__ import annotations

import html
import math
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from ..graph.builder import ExecutionGraph, build_graph
from ..graph.export import _edge_label, _inferred
from ..schema import Event, Trajectory
from .text import (CONTRIBUTION_NOTES, _money, _pct, _spread, coordination_caveats, coordination_rows)

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.coordination import CoordinationMetrics

CSS = """
:root{--bg:#f7f7f8;--fg:#1c1d21;--muted:#62656e;--card:#fff;--line:#e2e3e7;--node:#eef3ff;
--acc:#2f6fed;--ok:#1a8f55;--warn:#b86e00;--bad:#c9363f;--cp:#e0a400;--hatch:#9aa0aa}
@media (prefers-color-scheme: dark){:root{--bg:#141518;--fg:#e9eaee;--muted:#a2a6af;--card:#1d1f24;
--line:#34373e;--node:#22293a;--acc:#6f9bff;--ok:#43c486;--warn:#f0a83c;--bad:#ff6b73;--cp:#ffd24d;--hatch:#7c818b}}
*{box-sizing:border-box}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);
color:var(--fg);margin:0;padding:24px 16px;max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:13px;margin:14px 0 6px;color:var(--muted);font-weight:600}
.sub{color:var(--muted);margin-bottom:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
.tile .k{font-size:12px;color:var(--muted);font-weight:600}
.tile .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.tile .c{font-size:11px;color:var(--muted)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:13px;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.barwrap{display:inline-block;width:90px;height:9px;background:var(--line);border-radius:2px;vertical-align:middle;margin-right:6px}
.bar{display:block;height:9px;background:var(--acc);border-radius:2px}
.diag{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--muted);border-radius:8px;
padding:10px 14px;margin:10px 0}
.diag.medium{border-left-color:var(--warn)}.diag.high{border-left-color:var(--bad)}
.diag .t{font-weight:600}.diag .m{color:var(--muted);font-size:12px}.diag p{margin:4px 0}
.badge{display:inline-block;padding:0 7px;border-radius:10px;font-size:11px;font-weight:600;
background:var(--line);color:var(--fg);vertical-align:middle}
.badge.high{background:var(--bad);color:var(--card)}.badge.medium{background:var(--warn);color:var(--card)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.panel svg.fit{display:block;max-width:100%;height:auto}
.scroll{overflow-x:auto}
.note{font-size:12px;color:var(--muted);margin:6px 0}.warn{color:var(--warn)}
.legend{font-size:12px;color:var(--muted);margin:6px 0;display:flex;flex-wrap:wrap;gap:4px 14px}
.sw{display:inline-block;width:14px;height:9px;border-radius:2px;vertical-align:middle;margin-right:4px}
svg text{font:11px system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--fg)}
svg text.m{fill:var(--muted);font-size:10px}
svg .n{fill:var(--node);stroke:var(--acc);stroke-width:1.2}
svg .n.dead{stroke:var(--bad);stroke-dasharray:5 3;stroke-width:2}
svg .n.hub{stroke-width:2.6}
svg text.dead{fill:var(--bad)}
svg .e{fill:none;opacity:.85}
svg .e.message{stroke:var(--acc)}svg .e.artifact{stroke:var(--ok)}svg .e.invocation{stroke:var(--muted)}
svg .e.inf{stroke-dasharray:5 4}
svg .mk.message{fill:var(--acc)}svg .mk.artifact{fill:var(--ok)}svg .mk.invocation{fill:var(--muted)}
svg .grid{stroke:var(--line)}
svg .span{fill:var(--line);opacity:.55}svg .span.failed{fill:var(--bad);opacity:.25}
svg .u.llm{fill:var(--acc)}svg .u.tool{fill:var(--ok)}svg .u.wait{fill:url(#se-hatch);stroke:var(--hatch)}
svg .u.err{fill:var(--bad)}
svg .hatch-bg{fill:var(--card)}svg .hatch-ln{stroke:var(--hatch);stroke-width:2}
svg .tick.send{fill:var(--warn)}svg .tick.receive{fill:var(--muted)}
svg .tick.produce,svg .tick.use{stroke:var(--muted);stroke-width:1.5}
svg .cp{stroke:var(--cp);stroke-width:2.5}
svg .tick.cp{stroke-width:2}
.sw.llm{background:var(--acc)}.sw.tool{background:var(--ok)}.sw.err{background:var(--bad)}
.sw.wait{background:repeating-linear-gradient(45deg,var(--hatch) 0 2px,var(--card) 2px 5px);border:1px solid var(--hatch)}
.sw.send{background:var(--warn)}.sw.receive{background:var(--muted)}
.sw.cp{background:transparent;border:2.5px solid var(--cp)}
.sw.message{background:var(--acc);height:3px}.sw.artifact{background:var(--ok);height:3px}
.sw.invocation{background:var(--muted);height:3px}
.sw.inf{background:repeating-linear-gradient(90deg,var(--fg) 0 4px,transparent 4px 7px);height:2px}
.sw.dead{background:transparent;border:2px dashed var(--bad)}
"""

_FAILED = ("error", "injected", "timeout", "cancelled")
_KIND_ORDER = ("message", "artifact", "invocation")


def _e(x: Any) -> str:
    """HTML-escape (quotes included: the result also goes into attributes)."""
    return html.escape(str(x), quote=True)


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _tile(k: str, v: str, caption: str = "") -> str:
    return (f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div>'
            f'<div class="c">{_e(caption)}</div></div>')


def _tiles(tiles: Sequence[str]) -> str:
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _bar(frac: Optional[float]) -> str:
    if frac is None:
        return "unknown"
    w = max(0.0, min(1.0, frac)) * 100
    return f'<span class="barwrap"><span class="bar" style="width:{w:.0f}%"></span></span>{frac * 100:.0f}%'


def _legend(items: Sequence[Tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><span class="sw {c}"></span>{_e(t)}</span>'
                                            for c, t in items) + "</div>"


_DEF = re.compile(r"^(.*?) \((.*)\)$")


def _split_label(label: str) -> Tuple[str, str]:
    """``"Actual Parallelism (work ÷ wall)"`` -> name and its definition."""
    m = _DEF.match(label.strip())
    return (m.group(1), m.group(2)) if m else (label.strip(), "")


# --------------------------------------------------------------------------- #
# SVG: agent graph
# --------------------------------------------------------------------------- #
_NODE_W, _NODE_H, _COL_W, _ROW_H, _PAD = 160, 50, 230, 84, 24


def _first_start(traj: Trajectory, agents: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for a in agents:
        spans = traj.spans_for(a)
        out[a] = min(s.start for s in spans) if spans else math.inf
    return out


def _layout(graph: ExecutionGraph, first: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Columns by dependency rank: an agent sits one column right of the
    latest earlier-starting agent with an edge into it, so the flow reads
    left to right.  Edges into an agent from later-starting agents (replies)
    are ignored for ranking, which keeps the ranking acyclic."""
    order = sorted(graph.agents, key=lambda a: (first[a], a))
    preds: Dict[str, Set[str]] = {a: set() for a in order}
    for e in graph.agent_edges:
        s, t = e.source, e.target
        if s != t and s in preds and t in preds and (first[s], s) < (first[t], t):
            preds[t].add(s)
    rank: Dict[str, int] = {}
    for a in order:
        rank[a] = max([rank[p] + 1 for p in preds[a] if p in rank] or [0])
    cols: Dict[int, List[str]] = {}
    for a in order:
        cols.setdefault(rank[a], []).append(a)
    nrows = max(len(v) for v in cols.values())
    pos: Dict[str, Tuple[float, float]] = {}
    for r, names in cols.items():
        off = (nrows - len(names)) * _ROW_H / 2
        for i, a in enumerate(names):
            pos[a] = (_PAD + r * _COL_W, _PAD + off + i * _ROW_H)
    return pos


def _clip(cx: float, cy: float, tx: float, ty: float) -> Tuple[float, float]:
    """Point where the ray from a node centre towards (tx, ty) leaves the node box."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (_NODE_W / 2 + 3) / abs(dx) if dx else math.inf
    sy = (_NODE_H / 2 + 3) / abs(dy) if dy else math.inf
    s = min(sx, sy, 1.0)
    return cx + dx * s, cy + dy * s


def graph_svg(traj: Trajectory, graph: ExecutionGraph, cm: Optional["CoordinationMetrics"] = None) -> str:
    """Agent graph of one run as inline SVG plus its legend.  Edge style
    follows ``graph.export``: dashed when message delivery was inferred from
    timing (or never happened); stroke width grows with tokens."""
    agents = list(graph.agents)
    if not agents:
        return '<p class="note">no agents recorded</p>'
    first = _first_start(traj, agents)
    pos = _layout(graph, first)
    # bounding box of nodes and curve midpoints; the viewBox is fitted at the end
    xs = [x for x, _ in pos.values()] + [x + _NODE_W for x, _ in pos.values()]
    ys = [y for _, y in pos.values()] + [y + _NODE_H for _, y in pos.values()]
    inferred = _inferred(graph)
    undelivered: Dict[Tuple[str, str], int] = {}
    for m in graph.messages:
        if m.receive_node is None and m.sender is not None and m.receiver is not None:
            undelivered[(m.sender, m.receiver)] = undelivered.get((m.sender, m.receiver), 0) + 1
    hub = cm.communication.hub_agent if cm is not None and cm.communication is not None else None
    max_tok = max([e.tokens for e in graph.agent_edges] + [1])

    parts = ["", "<defs>"]
    for k in _KIND_ORDER:
        parts.append(f'<marker id="se-arr-{k}" markerWidth="8" markerHeight="8" refX="7" refY="4" '
                     f'orient="auto" markerUnits="userSpaceOnUse"><path class="mk {k}" d="M0,0 L8,4 L0,8 z"/></marker>')
    parts.append("</defs>")
    centre = {a: (x + _NODE_W / 2, y + _NODE_H / 2) for a, (x, y) in pos.items()}
    for e in graph.agent_edges:
        if e.source == e.target or e.source not in centre or e.target not in centre:
            continue
        kind = e.kind if e.kind in _KIND_ORDER else "invocation"
        (sx, sy), (tx, ty) = centre[e.source], centre[e.target]
        # Bend each edge to its left by an amount set by its kind: the reply
        # edge (opposite direction) bends the other way and parallel kinds
        # fan out, so no two edges of a pair overlap.
        # Edges that skip columns bend further so they pass around, not
        # through, the agents in between.
        dx, dy = tx - sx, ty - sy
        length = math.hypot(dx, dy) or 1.0
        skipped = max(0, int(round(abs(dx) / _COL_W)) - 1)
        bend = 16 * (1 + _KIND_ORDER.index(kind)) + 70 * skipped
        qx, qy = (sx + tx) / 2 - dy / length * bend, (sy + ty) / 2 + dx / length * bend
        x1, y1 = _clip(sx, sy, qx, qy)
        x2, y2 = _clip(tx, ty, qx, qy)
        xs.append(0.25 * x1 + 0.5 * qx + 0.25 * x2)
        ys.append(0.25 * y1 + 0.5 * qy + 0.25 * y2)
        n_inf = inferred.get((e.source, e.target, e.kind), 0)
        n_lost = undelivered.get((e.source, e.target), 0) if e.kind == "message" else 0
        width = min(6.0, 1.0 + 5.0 * e.tokens / max_tok)
        tip = _edge_label(e.kind, e.count, e.tokens, n_inf)
        tip = f"{e.source} → {e.target}: {tip}"
        if e.kind == "message":
            tip += f" · {e.used_count} used"
        if n_lost:
            tip += f" · {n_lost} never delivered"
        cls = "e " + kind + (" inf" if n_inf else "")
        parts.append(f'<path class="{cls}" d="M{x1:.1f},{y1:.1f} Q{qx:.1f},{qy:.1f} {x2:.1f},{y2:.1f}" '
                     f'stroke-width="{width:.1f}" marker-end="url(#se-arr-{kind})"><title>{_e(tip)}</title></path>')
        if e.count > 1:
            mx, my = 0.25 * x1 + 0.5 * qx + 0.25 * x2, 0.25 * y1 + 0.5 * qy + 0.25 * y2
            parts.append(f'<text class="m" x="{mx:.0f}" y="{my - 3:.0f}" text-anchor="middle">×{e.count}</text>')
    for a in agents:
        x, y = pos[a]
        u = cm.utilization.get(a) if cm is not None else None
        c = cm.contribution.get(a) if cm is not None else None
        dead = c is not None and c.dead is True
        share = u.token_share if u is not None else None
        role = (u.role if u is not None else None) or ""
        flags = []
        if dead:
            flags.append("dead")
        if a == hub:
            flags.append("hub")
        detail = " · ".join([p for p in (_short(role, 12), "" if share is None else f"{share * 100:.0f}% tok") if p]
                            + flags)
        tip = f"{a}" + (f" ({role})" if role else "")
        if share is not None:
            tip += f" · {share * 100:.1f}% of tokens"
        if c is not None:
            tip += f" · {c.status}"
            if c.provenance_coverage is not None:
                tip += f" · provenance coverage {c.provenance_coverage * 100:.0f}%"
        if a == hub and cm is not None and cm.communication is not None and cm.communication.hub_share is not None:
            tip += f" · hub: in {cm.communication.hub_share * 100:.0f}% of messages"
        cls = "n" + (" dead" if dead else "") + (" hub" if a == hub else "")
        parts.append(f'<g><title>{_e(tip)}</title>'
                     f'<rect class="{cls}" x="{x:.0f}" y="{y:.0f}" width="{_NODE_W}" height="{_NODE_H}" rx="8"/>'
                     f'<text x="{x + 10:.0f}" y="{y + 20:.0f}" font-weight="600">{_e(_short(a, 22))}</text>'
                     f'<text class="m{" dead" if dead else ""}" x="{x + 10:.0f}" y="{y + 37:.0f}">{_e(detail)}</text>'
                     "</g>")
    parts.append("</svg>")
    x0, y0 = int(min(xs) - _PAD), int(min(ys) - _PAD)
    W, H = int(max(xs) + _PAD) - x0, int(max(ys) + _PAD) - y0
    parts[0] = (f'<svg class="fit" width="{W}" height="{H}" viewBox="{x0} {y0} {W} {H}" role="img" '
                'aria-label="agent graph">')
    legend = _legend([("message", "message"), ("artifact", "artifact consumed"), ("invocation", "invocation"),
                      ("inf", "dashed: delivery inferred from timing or never delivered"),
                      ("dead", "dead agent (complete provenance, nothing it sent was used)")])
    notes = ['<p class="note">Columns follow first start time and who fed whom; stroke width grows with '
             'tokens. Hover an edge or agent for counts.</p>']
    if hub is not None:
        share = cm.communication.hub_share if cm is not None and cm.communication is not None else None
        notes.append(f'<p class="note">Hub agent: <b>{_e(hub)}</b>'
                     + (f", in {share * 100:.0f}% of messages" if share is not None else "") + ".</p>")
    return '<div class="panel">' + "".join(parts) + "</div>" + legend + "".join(notes)


# --------------------------------------------------------------------------- #
# SVG: timeline
# --------------------------------------------------------------------------- #
def _unit_tip(kind: str, ev: Optional[Event], start: float, end: float, t0: float) -> str:
    tip = f"{kind} @ {start - t0:.2f}s"
    if kind in ("llm", "tool", "wait"):
        tip += f", {end - start:.2f}s"
    if ev is None:
        return tip
    if kind == "llm":
        tip += f" · {ev.model or '?'} · {ev.prompt_tokens}→{ev.output_tokens} tok"
    elif kind == "tool":
        tip += f" · {ev.tool_name or '?'}"
    elif kind in ("send", "receive"):
        tip += f" · {ev.sender or '?'} → {ev.receiver or '?'} [{ev.kind or '-'}]"
    elif kind in ("produce", "use", "wait"):
        tip += f" · {ev.kind or '-'}"
    if ev.status != "success":
        tip += f" · {ev.status}"
    return tip


def timeline_svg(traj: Trajectory, graph: ExecutionGraph, critical: Optional[Set[str]] = None) -> str:
    """One row per span: LLM/tool calls as bars (nested calls thinner),
    waits hatched, message sends/receives as ticks; units on the critical
    path get ``class="cp"``."""
    critical = critical or set()
    spans = sorted(traj.spans.values(), key=lambda s: (s.start, s.agent_id, s.span_id))
    if not spans and not graph.units:
        return '<p class="note">no spans recorded</p>'
    starts = [s.start for s in spans] + [u.start for u in graph.units]
    ends = [s.end for s in spans if s.end is not None] + [u.end for u in graph.units]
    t0 = traj.started_at if traj.started_at is not None else min(starts)
    t_end = max(ends + [t0 + traj.duration])
    total = max(t_end - t0, 1e-6)
    by_span: Dict[Optional[str], List[Any]] = {}
    for u in graph.units:
        by_span.setdefault(u.span_id if u.span_id in traj.spans else None, []).append(u)
    rows: List[Tuple[str, str, Any]] = [(s.span_id, s.agent_id, s) for s in spans]
    if by_span.get(None):
        rows.append(("", "(no span)", None))
    events = {ev.event_id: ev for ev in traj.events}

    W, left, right, row_h, top = 960, 150, 16, 26, 22
    scale = (W - left - right) / total
    H = top + row_h * len(rows) + 6
    X = lambda t: left + (t - t0) * scale  # noqa: E731
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="timeline">',
             '<defs><pattern id="se-hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)"><rect class="hatch-bg" width="6" height="6"/>'
             '<line class="hatch-ln" x1="0" y1="0" x2="0" y2="6"/></pattern></defs>']
    for i in range(7):
        x = left + (W - left - right) * i / 6
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{H}"/>'
                     f'<text class="m" x="{x:.1f}" y="{top - 9}" text-anchor="middle">{total * i / 6:.1f}s</text>')
    for i, (sid, agent, span) in enumerate(rows):
        y = top + i * row_h
        role = f" ({span.role})" if span is not None and span.role else ""
        parts.append(f'<text x="6" y="{y + 15}"><title>{_e(agent + role)}</title>{_e(_short(agent, 22))}</text>')
        if span is not None:
            end = span.end if span.end is not None else t_end
            failed = " failed" if span.status in _FAILED else ""
            parts.append(f'<rect class="span{failed}" x="{X(span.start):.1f}" y="{y + 2}" '
                         f'width="{max(1.0, X(end) - X(span.start)):.1f}" height="{row_h - 6}" rx="3">'
                         f'<title>{_e(agent)} span {span.duration:.2f}s · {_e(span.status)}</title></rect>')
        for u in by_span.get(sid if span is not None else None, []):
            ev = events.get(u.event_id)
            cp = " cp" if u.node_id in critical else ""
            tip = _e(_unit_tip(u.kind, ev, u.start, u.end, t0) + (" · on critical path" if cp else ""))
            x0 = X(u.start)
            if u.kind in ("llm", "tool", "wait"):
                err = " err" if ev is not None and ev.status in _FAILED and u.kind != "wait" else ""
                nested = u.nested_in is not None
                by, bh = (y + 10, row_h - 14) if nested else (y + 5, row_h - 12)
                parts.append(f'<rect class="u {u.kind}{err}{cp}" x="{x0:.1f}" y="{by}" '
                             f'width="{max(1.5, X(u.end) - x0):.1f}" height="{bh}" rx="2"><title>{tip}</title></rect>')
            elif u.kind == "send":
                parts.append(f'<path class="tick send{cp}" d="M{x0:.1f},{y} l4,5 l-4,5 l-4,-5 z"><title>{tip}</title></path>')
            elif u.kind == "receive":
                yb = y + row_h - 4
                parts.append(f'<path class="tick receive{cp}" d="M{x0:.1f},{yb - 7} l4,7 l-8,0 z"><title>{tip}</title></path>')
            else:  # produce / use: thin tick
                parts.append(f'<line class="tick {u.kind}{cp}" x1="{x0:.1f}" y1="{y + 3}" x2="{x0:.1f}" '
                             f'y2="{y + row_h - 5}"><title>{tip}</title></line>')
    parts.append("</svg>")
    legend = _legend([("llm", "LLM call"), ("tool", "tool call"), ("wait", "wait"), ("err", "failed call"),
                      ("send", "message sent"), ("receive", "message received"),
                      ("cp", "on the critical path")])
    return '<div class="panel scroll">' + "".join(parts) + "</div>" + legend


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _phase1(result: Any) -> List[str]:
    perf, eff = result.performance, result.efficiency
    n_runs = len(result.runs)
    perf_tiles = [_tile("Task Success", _pct(perf.success_rate), f"{perf.n_successes}/{n_runs} runs"),
                  _tile("Mean Score", f"{perf.mean_score:.3f}")]
    if perf.constraint_pass_rate is not None:
        perf_tiles.append(_tile("Constraint Pass", _pct(perf.constraint_pass_rate)))
    if perf.run_errors:
        perf_tiles.append(_tile("Runs Crashed", f"{perf.run_errors}/{n_runs}"))
    if perf.trials > 1:
        perf_tiles.append(_tile("Per-trial Success", ", ".join(_pct(x) for x in perf.per_trial)))
    out = ["<h2>Performance</h2>", _tiles(perf_tiles)]
    complete = eff.cost_complete
    out += ["<h2>Efficiency</h2>", _tiles([
        _tile("Cost / Run", _money(result.cost["per_run"], complete)),
        _tile("Cost / Successful Task", _money(eff.cost_per_success, complete)),
        _tile("Total Cost", _money(eff.cost_total, complete)),
        _tile("Tokens / Run", f"{eff.tokens.mean:,.0f}",
              f"in {eff.input_tokens.mean:,.0f} / out {eff.output_tokens.mean:,.0f}"),
        _tile("LLM Calls / Run", f"{eff.llm_calls.mean:.1f}", f"tool calls {eff.tool_calls.mean:.1f}"),
        _tile("Messages / Run", f"{eff.messages.mean:.1f}"),
        _tile("Wall Clock / Run", _spread(eff.wall_clock_s, "s")),
        _tile("Agent Time / Run", f"{eff.agent_time_s.mean:.1f}s", f"waiting {eff.waiting_time_s.mean:.1f}s"),
    ] + ([_tile("Errors (all runs)", f"{eff.errors.total:.0f}")] if eff.errors.total else []))]
    if not complete:
        out.append('<p class="note">* cost incomplete: some models have no pricing entry (unpriced calls are '
                   "excluded, not counted as $0; re-price with <code>swarmeval report --pricing</code>)</p>")
    if eff.per_agent:
        out += ["<h3>Agents (per run, mean)</h3>", '<div class="tablewrap"><table><tr><th>Agent</th><th>Role</th>'
                '<th>Token share</th><th class="num">LLM</th><th class="num">Tools</th><th class="num">Msgs</th>'
                '<th class="num">Time</th><th class="num">Cost</th></tr>']
        for aid, u in sorted(eff.per_agent.items(), key=lambda kv: -kv[1]["token_share"]):
            out.append(f"<tr><td>{_e(aid)}</td><td>{_e(u.get('role') or '-')}</td><td>{_bar(u['token_share'])}</td>"
                       f'<td class="num">{u["llm_calls"]:.1f}</td><td class="num">{u["tool_calls"]:.1f}</td>'
                       f'<td class="num">{u["messages_sent"]:.1f}</td><td class="num">{u["self_time_s"]:.1f}s</td>'
                       f'<td class="num">{_e(_money(u["cost_usd"], u.get("cost_complete", True)))}</td></tr>')
        out.append("</table></div>")
    if perf.per_task:
        out += ["<h3>Tasks</h3>", '<div class="tablewrap"><table><tr><th>Task</th><th class="num">Success</th>'
                '<th class="num">Score</th><th class="num">Cost</th><th class="num">Latency</th></tr>']
        for tid, t in perf.per_task.items():
            out.append(f'<tr><td>{_e(tid)}</td><td class="num">{_pct(t["success_rate"])}</td>'
                       f'<td class="num">{t["score"]:.2f}</td>'
                       f'<td class="num">{_e(_money(t["cost_usd"], t.get("cost_complete", True)))}</td>'
                       f'<td class="num">{t["latency_s"]:.1f}s</td></tr>')
        out.append("</table></div>")
    return out


def _coordination_tiles(cs: Any) -> List[str]:
    out = ["<h2>Coordination</h2>",
           f'<p class="sub">{cs.n_runs} runs · values are means over runs (range)</p>']
    group: List[str] = []
    for row in coordination_rows(cs) + [None]:
        if row is None:
            if group:
                out.append(_tiles(group))
            group = []
            continue
        name, definition = _split_label(row[0])
        group.append(_tile(name, row[1], definition))
    for warn, text in coordination_caveats(cs):
        out.append(f'<p class="note{" warn" if warn else ""}">{"⚠ " if warn else ""}{_e(text)}</p>')
    return out


def _run_views(result: Any, run_index: int) -> List[str]:
    traj = result.runs.trajectories[run_index]
    per_run = result.per_run_coordination
    cm = per_run[run_index] if per_run is not None and run_index < len(per_run) else None
    graph = (cm.graph if cm is not None else None) or build_graph(traj)
    status = "succeeded" if traj.succeeded else "failed"
    head = (f'<span class="badge">run {run_index}: task {_e(traj.task_id)} · trial {traj.trial} · '
            f"{_e(status)}</span>")
    out: List[str] = []
    if cm is not None:
        out += [f"<h2>Agent graph {head}</h2>", graph_svg(traj, graph, cm)]
        cov = graph.coverage.message_receive_coverage
        if cov is not None and cov < 1.0:
            out.append(f'<p class="note warn">⚠ this run recorded receive events for {cov * 100:.0f}% of messages; '
                       "dashed message edges are inferred from timing</p>")
    critical = set(cm.dependency.critical_path.node_ids) if cm is not None and cm.dependency is not None else set()
    out += [f"<h2>Timeline {head}</h2>", timeline_svg(traj, graph, critical)]
    if cm is not None and cm.dependency is not None:
        dep = cm.dependency
        out.append(f'<p class="note">Critical path {dep.critical_path_s:.1f}s of {dep.wall_s:.1f}s wall clock'
                   + (f", {dep.inferred_edges_on_path} inferred edges on it" if dep.inferred_edges_on_path else "")
                   + ". Waits weigh 0 on the path.</p>")
    return out


def _contribution(cs: Any) -> List[str]:
    if not cs.contribution:
        return []
    out = ["<h2>Observed contribution (provenance, not marginal effect)</h2>",
           '<div class="tablewrap"><table><tr><th>Agent</th><th>Status (runs)</th><th class="num">Lineage</th>'
           '<th class="num">Consumed</th><th class="num">Dead</th><th class="num">Coverage</th>'
           "<th>Utilization (tokens)</th></tr>"]
    for aid, c in cs.contribution.items():
        status = ", ".join(f"{s.replace('_', ' ')} {n}" for s, n in c.get("status_counts", {}).items()) or "-"
        dead = c.get("dead_rate")
        cov = c.get("provenance_coverage")
        share = cs.utilization.get(aid, {}).get("token_share")
        dead_s = "unknown" if dead is None else f"{dead * 100:.0f}%"
        cov_s = "unknown" if cov is None else f"{cov['mean'] * 100:.0f}%"
        out.append(f"<tr><td>{_e(aid)}</td><td>{_e(status)}</td>"
                   f'<td class="num">{c["lineage_rate"] * 100:.0f}%</td>'
                   f'<td class="num">{c["consumed_rate"] * 100:.0f}%</td>'
                   f'<td class="num">{dead_s}</td><td class="num">{cov_s}</td>'
                   f"<td>{_bar(share)}</td></tr>")
    out.append("</table></div>")
    out += [f'<p class="note">{_e(n)}</p>' for n in CONTRIBUTION_NOTES]
    return out


def _diagnostics(cs: Any) -> List[str]:
    diag = cs.diagnostics
    if diag is None:
        return []
    out = ["<h2>Diagnostics — hypotheses to test, not verdicts</h2>"]
    if not diag.smells:
        out.append('<p class="note">no smells at the current thresholds</p>')
    for sp in diag.smells:
        sev = sp.max_severity if sp.max_severity in ("low", "medium", "high") else "low"
        agents = ", ".join(f"{a} ({n} runs)" for a, n in list(sp.agents.items())[:6])
        meta = f"{sp.runs_affected}/{sp.n_runs} runs ({sp.prevalence * 100:.0f}%)"
        if agents:
            meta += f" · agents: {agents}"
        if sp.inferred_runs:
            meta += f" · resting on inferred edges in {sp.inferred_runs} runs"
        out.append(f'<div class="diag {sev}"><div class="t"><span class="badge {sev}">{_e(sp.max_severity)}</span> '
                   f'{_e(sp.smell)}</div><div class="m">{_e(meta)}</div>'
                   f"<p><b>Observed:</b> {_e(sp.observation)}</p>"
                   f"<p><b>Hypothesis:</b> {_e(sp.hypothesis)}</p>"
                   f"<p><b>Test:</b> {_e(sp.intervention)}</p></div>")
    return out


# --------------------------------------------------------------------------- #
# Main renderer
# --------------------------------------------------------------------------- #
def render_html(result: Any, path: Optional[str] = None, run_index: int = 0) -> str:
    """The whole report as one HTML document; also written to ``path`` when
    given.  ``run_index`` picks the run whose agent graph and timeline are
    drawn (the other sections cover all runs)."""
    n_runs = len(result.runs)
    if n_runs and not 0 <= run_index < n_runs:
        raise ValueError(f"run_index {run_index} out of range (result has {n_runs} runs)")
    perf = result.performance
    title = f"SwarmEval · {result.label}"
    body: List[str] = [
        f"<h1>{_e(title)}</h1>",
        f'<div class="sub">{_e(result.benchmark)} · {perf.n_tasks} tasks × {perf.trials} trials = {n_runs} runs · '
        "health dimensions are reported separately; there is no overall score</div>",
    ]
    body += _phase1(result)
    cs = result.coordination
    if cs is None:
        body.append('<h2>Coordination</h2><p class="note">Coordination analysis was skipped for this report '
                    "(run without <code>--no-coordination</code> to see the agent graph, critical path, observed "
                    "contribution and diagnostics).</p>")
    else:
        body += _coordination_tiles(cs)
    if n_runs:
        body += _run_views(result, run_index)
    if cs is not None:
        body += _contribution(cs)
        body += _diagnostics(cs)
    body.append(f'<p class="note">Evaluators: {_e(", ".join(perf.evaluators) or "none")}. '
                "Generated by SwarmEval. Observed metrics describe the recorded trajectories; "
                "the effect of removing or changing an agent needs an ablation run.</p>")
    doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           '<meta name="color-scheme" content="light dark">'
           f"<title>{_e(title)}</title><style>{CSS}</style></head><body>\n"
           + "\n".join(body) + "\n</body></html>\n")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
    return doc


__all__ = ["render_html", "graph_svg", "timeline_svg"]
