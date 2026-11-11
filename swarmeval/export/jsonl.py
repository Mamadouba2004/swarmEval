"""One-event-per-line export: easy to load into pandas, DuckDB or a warehouse."""
from __future__ import annotations

import json
from collections.abc import Iterable

from ..schema import Event, Trajectory


def write_events_jsonl(source: Trajectory | Iterable[Trajectory], path: str) -> int:
    """Write every event as one JSON line; returns the number written."""
    trajs = [source] if isinstance(source, Trajectory) else list(source)
    n = 0
    with open(path, "w") as f:
        for t in trajs:
            for ev in t.events:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")
                n += 1
    return n


def read_events_jsonl(path: str) -> list[Event]:
    """Read events from a JSONL file."""
    out: list[Event] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Event.from_dict(json.loads(line)))
    return out
