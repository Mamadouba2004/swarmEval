"""One-row-per-event Parquet export (needs ``pyarrow``: ``pip install swarmeval[parquet]``).

Columns match the JSONL export.  Fields that hold arbitrary values
(``content``, ``tool_args``, ``metadata`` and the artifact lists) are stored
as JSON strings so every column has a single type.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..schema import Event, Trajectory

_JSON_COLUMNS = ("content", "tool_args", "metadata", "input_artifacts", "output_artifacts")


def _pyarrow():
    """Import pyarrow, explaining how to install it if missing."""
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError("Parquet export needs pyarrow: pip install 'swarmeval[parquet]'") from None
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def _row(ev: Event) -> dict[str, Any]:
    """One event as a Parquet row, with variable-type fields as JSON strings."""
    d = ev.to_dict()
    for k in _JSON_COLUMNS:
        d[k] = json.dumps(d[k], default=str) if d[k] is not None else None
    return d


def write_events_parquet(source: Trajectory | Iterable[Trajectory], path: str) -> int:
    """Write every event as one Parquet row; returns the number written."""
    pa, pq = _pyarrow()
    trajs = [source] if isinstance(source, Trajectory) else list(source)
    rows = [_row(ev) for t in trajs for ev in t.events]
    columns = list(Event.__dataclass_fields__)
    table = pa.Table.from_pylist(rows) if rows else pa.table({c: [] for c in columns})
    pq.write_table(table, path)
    return len(rows)


def read_events_parquet(path: str) -> list[Event]:
    """Read events from a Parquet file written by ``write_events_parquet``."""
    _, pq = _pyarrow()
    out: list[Event] = []
    for d in pq.read_table(path).to_pylist():
        for k in _JSON_COLUMNS:
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        d["input_artifacts"] = d.get("input_artifacts") or []
        d["output_artifacts"] = d.get("output_artifacts") or []
        d["metadata"] = d.get("metadata") or {}
        out.append(Event.from_dict(d))
    return out
