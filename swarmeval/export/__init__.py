"""Event-stream export: one record per event as JSONL or Parquet, picked by format or file extension."""
from collections.abc import Iterable

from ..schema import Trajectory
from .jsonl import read_events_jsonl, write_events_jsonl
from .parquet import read_events_parquet, write_events_parquet

FORMATS = {"jsonl": write_events_jsonl, "parquet": write_events_parquet}


def export_events(source: Trajectory | Iterable[Trajectory], path: str, fmt: str = "") -> int:
    """Write one record per event in ``fmt`` ("jsonl" or "parquet"; inferred
    from the file extension when empty).  Returns the number of events."""
    fmt = fmt or ("parquet" if path.endswith(".parquet") else "jsonl")
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; known: {sorted(FORMATS)}")
    return FORMATS[fmt](source, path)


__all__ = ["export_events", "read_events_jsonl", "write_events_jsonl", "read_events_parquet",
           "write_events_parquet"]
