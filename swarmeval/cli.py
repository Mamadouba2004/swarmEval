"""Command-line interface.

    swarmeval demo [--trials N] [--out DIR]
    swarmeval run --swarm pkg.module:swarm --benchmark bench.json [--trials N] [--out DIR]
    swarmeval report runs/baseline.json [--trace N] [--pricing prices.json]
    swarmeval export runs/baseline.json events.parquet [--format jsonl|parquet]
    swarmeval compare runs/baseline.json runs/candidate.json [--pricing prices.json] [--out cmp.json]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, List, Optional

from . import __version__
from .evaluate import AnalysisFailed, EvaluationResult, analyze_runs, evaluate, load_runset
from .export import export_events
from .export.jsonl import write_events_jsonl
from .pricing import PricingTable
from .recorder import SimClock
from .report.text import render_trajectory
from .schema import Benchmark


def _load_obj(spec: str) -> Any:
    if ":" not in spec:
        raise SystemExit(f"expected package.module:attr (got {spec!r})")
    mod, attr = spec.split(":", 1)
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), attr)


def _load_benchmark(spec: str) -> Benchmark:
    if ":" in spec and not os.path.exists(spec):
        obj = _load_obj(spec)
        return obj() if callable(obj) and not isinstance(obj, Benchmark) else obj
    return Benchmark.load(spec)


def _clock_factory(name: Optional[str]) -> Optional[Callable[[], Any]]:
    if name == "sim":
        return SimClock
    if name in (None, "wall"):
        return None
    return _load_obj(name)


def _write_outputs(result: EvaluationResult, out: Optional[str]) -> None:
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    result.save(os.path.join(out, f"{result.label}.json"))
    write_events_jsonl(result.runs.trajectories, os.path.join(out, f"{result.label}.events.jsonl"))
    print(f"\nwrote {out}/{result.label}.json and {out}/{result.label}.events.jsonl")


def _write_plain_evaluation(exc: AnalysisFailed, out: Optional[str]) -> int:
    print(exc.result.report())
    _write_outputs(exc.result, out)
    cause = exc.__cause__
    done = f"written to {out}" if out else "shown"
    print(f"swarmeval: coordination analysis failed: {type(cause).__name__}: {cause}; plain evaluation {done}",
          file=sys.stderr)
    return 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .examples.research_swarm import build_benchmark, make_swarm
    try:
        result = evaluate(make_swarm, build_benchmark(), trials=args.trials, clock_factory=SimClock, progress=True)
    except AnalysisFailed as exc:
        return _write_plain_evaluation(exc, args.out)
    print(result.report())
    if args.trace:
        print()
        print(render_trajectory(result.runs.trajectories[0]))
    _write_outputs(result, args.out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    swarm = _load_obj(args.swarm)
    bench = _load_benchmark(args.benchmark)
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    try:
        result = evaluate(swarm, bench, trials=args.trials, seed=args.seed, clock_factory=_clock_factory(args.clock),
                          label=args.label, progress=True, pricing=pricing, capture_text=not args.no_capture_text)
    except AnalysisFailed as exc:
        return _write_plain_evaluation(exc, args.out)
    print(result.report())
    _write_outputs(result, args.out)
    return 0


def _load_runs_or_exit(path: str):
    try:
        return load_runset(path)
    except ValueError as exc:
        raise SystemExit(f"swarmeval: {exc}")


def _analysis_failed(exc: Exception) -> None:
    print(f"swarmeval: coordination analysis failed: {type(exc).__name__}: {exc}; showing the plain evaluation "
          "(rerun with --no-coordination to skip it)", file=sys.stderr)


def cmd_report(args: argparse.Namespace) -> int:
    runs = _load_runs_or_exit(args.path)
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    code = 0
    try:
        result = analyze_runs(runs, pricing=pricing)
    except Exception as exc:  # every run failed the same way: the runs are on disk, show what still holds
        _analysis_failed(exc)
        result, code = analyze_runs(runs, pricing=pricing, coordination=False), 1
    print(result.report())
    if args.trace is not None:
        if not 0 <= args.trace < len(runs):
            raise SystemExit(f"swarmeval: --trace {args.trace} out of range (file has {len(runs)} runs)")
        print()
        print(render_trajectory(runs.trajectories[args.trace]))
    return code


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare
    base, cand = _load_runs_or_exit(args.baseline), _load_runs_or_exit(args.candidate)
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    try:
        cmp = compare(base, cand, pricing=pricing)
    except ValueError as exc:
        raise SystemExit(f"swarmeval: {exc}")
    print(cmp.report())
    if args.out:
        cmp.save(args.out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    runs = _load_runs_or_exit(args.path)
    n = export_events(runs.trajectories, args.out, args.format or "")
    print(f"wrote {n} events to {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="swarmeval", description="Record and evaluate multi-agent systems.")
    p.add_argument("--version", action="version", version=f"swarmeval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the built-in simulated research swarm")
    d.add_argument("--trials", type=int, default=3)
    d.add_argument("--out", default=None, help="directory for JSON/JSONL outputs")
    d.add_argument("--trace", action="store_true", help="also print the first trajectory")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="run a swarm on a benchmark and print the report")
    r.add_argument("--swarm", required=True, help="package.module:swarm_or_factory_or_run_fn")
    r.add_argument("--benchmark", required=True, help="path to benchmark JSON/JSONL or package.module:builder")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--clock", default=None, help="'wall' (default), 'sim', or package.module:factory")
    r.add_argument("--label", default="baseline")
    r.add_argument("--out", default=None)
    r.add_argument("--pricing", default=None, help='JSON {"model": [input, output], ...} in USD per 1M tokens')
    r.add_argument("--no-capture-text", action="store_true",
                   help="keep only short previews of prompts, outputs and artifacts")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="re-analyse a saved run set")
    rp.add_argument("path")
    rp.add_argument("--trace", type=int, default=None, metavar="N", help="also print trajectory N")
    rp.add_argument("--pricing", default=None, help="re-price every LLM call with this JSON pricing table")
    rp.set_defaults(fn=cmd_report)

    ex = sub.add_parser("export", help="convert a saved run set to an event stream (JSONL or Parquet)")
    ex.add_argument("path", help="saved RunSet or EvaluationResult JSON")
    ex.add_argument("out", help="output file (.jsonl or .parquet)")
    ex.add_argument("--format", choices=["jsonl", "parquet"], default=None,
                    help="default: inferred from the output extension")
    ex.set_defaults(fn=cmd_export)

    cp = sub.add_parser("compare", help="compare a candidate run set against a baseline")
    cp.add_argument("baseline", help="saved RunSet or EvaluationResult JSON")
    cp.add_argument("candidate", help="saved RunSet or EvaluationResult JSON")
    cp.add_argument("--pricing", default=None, help="price both sides with this JSON pricing table")
    cp.add_argument("--out", default=None, help="also write the comparison as JSON")
    cp.set_defaults(fn=cmd_compare)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
