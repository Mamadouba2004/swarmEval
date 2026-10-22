"""Execution graph export: Mermaid, Graphviz DOT and JSON (Phase 2: visualization).

Two views of one run's ``ExecutionGraph``:

* ``agents`` -- one node per agent (labelled with its token share when the
  coordination metrics are given), one edge per (source, target, kind) with
  ``kind ×count`` and tokens.  Message edges whose delivery was inferred from
  timing (no receive event, as in Phase 1 data) are dashed; dead agents
  (``dead is True``) are dashed and red.
* ``dag`` -- the work-unit DAG, units grouped by agent (Mermaid subgraph /
  DOT cluster), critical-path units highlighted, inferred edges dashed.

Agent ids are user data, so node ids are generated (``a0``, ``u12``) and
every label is escaped for its format.  Output is deterministic (graph order,
which is start order with stable tie-breaks), so it can be diffed and
snapshot-tested.  Paste Mermaid into GitHub Markdown or mermaid.live; render
DOT with ``dot -Tsvg``.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from .builder import ExecutionGraph

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.coordination import CoordinationMetrics

VIEWS = ("agents", "dag")
_CALL_KINDS = ("llm", "tool")


def _check_view(view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"unknown graph view {view!r}; known: {list(VIEWS)}")


def _agent_info(graph: ExecutionGraph, cm: Optional["CoordinationMetrics"]) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for a in graph.agents:
        u = cm.utilization.get(a) if cm is not None else None
        c = cm.contribution.get(a) if cm is not None else None
        info[a] = {"token_share": u.token_share if u is not None else None,
                   "dead": c.dead if c is not None else None}
    return info


def _inferred(graph: ExecutionGraph) -> Dict[Any, int]:
    """(source, target, kind) -> how many of the edge's messages had their
    *delivery* inferred (no receive event).  Use judged by timing alone does
    not make the edge itself inferred: the message was recorded arriving."""
    out: Dict[Any, int] = {}
    for m in graph.messages:
        if not m.received and m.sender is not None and m.receiver is not None:
            k = (m.sender, m.receiver, "message")
            out[k] = out.get(k, 0) + 1
    return out


def _cp_nodes(cm: Optional["CoordinationMetrics"]) -> Set[str]:
    if cm is None or cm.dependency is None:
        return set()
    return set(cm.dependency.critical_path.node_ids)


def _edge_label(kind: str, count: int, tokens: int, inferred: int) -> str:
    label = f"{kind} ×{count}"
    if tokens:
        label += f" · {tokens:,} tok"
    if inferred:
        label += f" · {inferred} inferred"
    return label


def _unit_label(u: Any) -> str:
    if u.kind in _CALL_KINDS:
        return f"{u.kind} {u.weight:.1f}s"
    if u.kind == "wait":
        return f"wait {u.end - u.start:.1f}s"
    return u.kind


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #
_MERMAID_ESC = {"&": "&amp;", '"': "#quot;", "<": "&lt;", ">": "&gt;", "[": "#91;", "]": "#93;",
                "{": "#123;", "}": "#125;", "|": "#124;", "`": "#96;", "\n": " ", "\r": " "}


def mermaid_escape(text: str) -> str:
    return "".join(_MERMAID_ESC.get(ch, ch) for ch in str(text))


def to_mermaid(graph: ExecutionGraph, view: str = "agents",
               coordination: Optional["CoordinationMetrics"] = None) -> str:
    _check_view(view)
    lines = ["flowchart LR" if view == "agents" else "flowchart TB"]
    if view == "agents":
        info = _agent_info(graph, coordination)
        inferred = _inferred(graph)
        ids = {a: f"a{i}" for i, a in enumerate(graph.agents)}
        for a in graph.agents:
            share = info[a]["token_share"]
            label = mermaid_escape(a) + (f"<br/>{share * 100:.0f}% tokens" if share is not None else "")
            lines.append(f'    {ids[a]}["{label}"]' + (":::dead" if info[a]["dead"] is True else ""))
        for e in graph.agent_edges:
            if e.source not in ids or e.target not in ids:
                continue
            n_inf = inferred.get((e.source, e.target, e.kind), 0)
            arrow = "-.->" if n_inf else "-->"
            label = mermaid_escape(_edge_label(e.kind, e.count, e.tokens, n_inf))
            lines.append(f'    {ids[e.source]} {arrow}|"{label}"| {ids[e.target]}')
        lines.append("    classDef dead stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 5,color:#c0392b")
        return "\n".join(lines) + "\n"
    units = graph.dag_units
    uid = {u.node_id: f"u{i}" for i, u in enumerate(units)}
    cp = _cp_nodes(coordination)
    agents = list(graph.agents) + sorted({u.agent_id or "-" for u in units} - set(graph.agents))
    for gi, a in enumerate(agents):
        members = [u for u in units if (u.agent_id or "-") == a]
        if not members:
            continue
        lines.append(f'    subgraph g{gi}["{mermaid_escape(a)}"]')
        for u in members:
            lines.append(f'        {uid[u.node_id]}["{mermaid_escape(_unit_label(u))}"]'
                         + (":::cp" if u.node_id in cp else ""))
        lines.append("    end")
    for e in graph.edges:
        if e.src not in uid or e.dst not in uid:
            continue
        arrow = "-.->" if e.inferred else "-->"
        label = "" if e.type == "sequence" else f'|"{mermaid_escape(e.type)}"|'
        lines.append(f"    {uid[e.src]} {arrow}{label} {uid[e.dst]}")
    lines.append("    classDef cp fill:#fde68a,stroke:#b45309,stroke-width:2px")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# DOT
# --------------------------------------------------------------------------- #
def dot_escape(text: str) -> str:
    """Body of a double-quoted DOT string: backslashes and quotes escaped,
    line breaks as ``\\n``, angle brackets as entities so nothing reads as
    markup in an SVG viewer."""
    out = str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "\\n")
    return out.replace("<", "&lt;").replace(">", "&gt;")


def to_dot(graph: ExecutionGraph, view: str = "agents",
           coordination: Optional["CoordinationMetrics"] = None) -> str:
    _check_view(view)
    name = dot_escape(f"{graph.run_id} {view}")
    lines = [f'digraph "{name}" {{', "    rankdir=LR;" if view == "agents" else "    rankdir=TB;",
             '    node [shape=box, fontname="Helvetica"];', '    edge [fontname="Helvetica", fontsize=10];']
    if view == "agents":
        info = _agent_info(graph, coordination)
        inferred = _inferred(graph)
        ids = {a: f"a{i}" for i, a in enumerate(graph.agents)}
        for a in graph.agents:
            share = info[a]["token_share"]
            label = dot_escape(a) + (f"\\n{share * 100:.0f}% tokens" if share is not None else "")
            style = ", style=dashed, color=red, fontcolor=red" if info[a]["dead"] is True else ""
            lines.append(f'    {ids[a]} [label="{label}"{style}];')
        for e in graph.agent_edges:
            if e.source not in ids or e.target not in ids:
                continue
            n_inf = inferred.get((e.source, e.target, e.kind), 0)
            style = ", style=dashed" if n_inf else ""
            label = dot_escape(_edge_label(e.kind, e.count, e.tokens, n_inf))
            lines.append(f'    {ids[e.source]} -> {ids[e.target]} [label="{label}"{style}];')
        lines.append("}")
        return "\n".join(lines) + "\n"
    units = graph.dag_units
    uid = {u.node_id: f"u{i}" for i, u in enumerate(units)}
    cp = _cp_nodes(coordination)
    agents = list(graph.agents) + sorted({u.agent_id or "-" for u in units} - set(graph.agents))
    for gi, a in enumerate(agents):
        members = [u for u in units if (u.agent_id or "-") == a]
        if not members:
            continue
        lines.append(f"    subgraph cluster_{gi} {{")
        lines.append(f'        label="{dot_escape(a)}";')
        for u in members:
            style = ', style=filled, fillcolor="#fde68a", penwidth=2' if u.node_id in cp else ""
            lines.append(f'        {uid[u.node_id]} [label="{dot_escape(_unit_label(u))}"{style}];')
        lines.append("    }")
    for e in graph.edges:
        if e.src not in uid or e.dst not in uid:
            continue
        attrs = [] if e.type == "sequence" else [f'label="{dot_escape(e.type)}"']
        if e.inferred:
            attrs.append("style=dashed")
        lines.append(f"    {uid[e.src]} -> {uid[e.dst]}" + (f" [{', '.join(attrs)}]" if attrs else "") + ";")
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def to_json(graph: ExecutionGraph, view: str = "agents",
            coordination: Optional["CoordinationMetrics"] = None) -> str:
    _check_view(view)
    if view == "agents":
        info = _agent_info(graph, coordination)
        inferred = _inferred(graph)
        d: Dict[str, Any] = {
            "run_id": graph.run_id, "view": view,
            "nodes": [{"id": a, **info[a]} for a in graph.agents],
            "edges": [dict(e.to_dict(), delivery_inferred=inferred.get((e.source, e.target, e.kind), 0))
                      for e in graph.agent_edges],
            "coverage": graph.coverage.to_dict(),
        }
    else:
        cp = _cp_nodes(coordination)
        d = {"run_id": graph.run_id, "view": view,
             "nodes": [dict(u.to_dict(), critical=u.node_id in cp) for u in graph.dag_units],
             "edges": [e.to_dict() for e in graph.edges],
             "critical_path": list(coordination.dependency.critical_path.node_ids)
             if coordination is not None and coordination.dependency is not None else None,
             "coverage": graph.coverage.to_dict()}
    return json.dumps(d, indent=1, sort_keys=True, default=str) + "\n"


GRAPH_FORMATS: Dict[str, Callable[..., str]] = {"mermaid": to_mermaid, "dot": to_dot, "json": to_json}


def export_graph(graph: ExecutionGraph, fmt: str = "mermaid", view: str = "agents",
                 coordination: Optional["CoordinationMetrics"] = None) -> str:
    if fmt not in GRAPH_FORMATS:
        raise ValueError(f"unknown graph format {fmt!r}; known: {sorted(GRAPH_FORMATS)}")
    return GRAPH_FORMATS[fmt](graph, view=view, coordination=coordination)


__all__: List[str] = ["GRAPH_FORMATS", "VIEWS", "export_graph", "to_mermaid", "to_dot", "to_json",
                      "mermaid_escape", "dot_escape"]
