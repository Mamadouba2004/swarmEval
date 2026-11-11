"""Command-line interface.

    swarmeval demo [--trials N] [--out DIR] [--html FILE [--run N]]
    swarmeval run --swarm pkg.module:swarm --benchmark bench.json [--trials N] [--out DIR]
    swarmeval report runs/baseline.json [--trace N] [--pricing prices.json] [--html FILE [--run N]]
    swarmeval export runs/baseline.json events.parquet [--format jsonl|parquet]
    swarmeval compare runs/baseline.json runs/candidate.json [--pricing prices.json] [--out cmp.json]
    swarmeval graph runs/baseline.json [--run N] [--view agents|dag] [--format mermaid|dot|json] [--out F]

demo, run, report and compare also take --no-coordination, --similarity NAME|pkg.mod:attr and
--smells-config FILE (JSON object of SmellConfig threshold overrides).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from . import __version__
from .evaluate import AnalysisFailed, EvaluationResult, analyze_runs, evaluate, failure_errors, load_runset
from .export import export_events
from .diagnostics.smells import load_smell_config
from .export.jsonl import write_events_jsonl
from .graph.builder import build_graph
from .graph.export import GRAPH_FORMATS, VIEWS, export_graph
from .metrics.coordination import compute_coordination
from .runner import RunInterrupted
from .pricing import PricingTable
from .recorder import SimClock
from .report.html import render_html
from .report.text import render_trajectory
from .schema import Benchmark
from .similarity import resolve_similarity


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


def _analysis_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Coordination options shared by demo/run/report/compare."""
    smell_config = None
    if getattr(args, "smells_config", None):
        try:
            smell_config = load_smell_config(args.smells_config)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"swarmeval: --smells-config: {exc}")
    similarity = getattr(args, "similarity", None)
    if similarity:
        try:
            similarity = resolve_similarity(similarity)
        except (ImportError, ValueError, AttributeError, TypeError) as exc:
            raise SystemExit(f"swarmeval: --similarity: {exc}")
    return {"coordination": not getattr(args, "no_coordination", False), "similarity": similarity,
            "smell_config": smell_config}


def _add_analysis_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-coordination", action="store_true",
                   help="skip the coordination analysis (execution graph, smells)")
    p.add_argument("--similarity", default=None, metavar="NAME|pkg.mod:attr",
                   help="artifact redundancy backend: lexical (default), embeddings, or an import path")
    p.add_argument("--smells-config", default=None, metavar="FILE",
                   help="JSON object overriding smell thresholds (see SmellConfig)")


def _write_outputs(result: EvaluationResult, out: Optional[str]) -> None:
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    result.save(os.path.join(out, f"{result.label}.json"))
    write_events_jsonl(result.runs.trajectories, os.path.join(out, f"{result.label}.events.jsonl"))
    print(f"\nwrote {out}/{result.label}.json and {out}/{result.label}.events.jsonl")


def _add_html_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--html", default=None, metavar="FILE", help="also write a standalone HTML report")
    p.add_argument("--run", type=int, default=None, metavar="N",
                   help="which run the HTML agent graph and timeline show (default 0; needs --html)")


def _check_html_run(args: argparse.Namespace, n_runs: int) -> None:
    """Reject a bad --run with a clean message (before evaluating where possible)."""
    if args.html and n_runs and not 0 <= (args.run or 0) < n_runs:   # no runs: render_html writes the summary
        raise SystemExit(f"swarmeval: --run {args.run or 0} out of range (result has {n_runs} runs)")


def _write_html(result: EvaluationResult, args: argparse.Namespace) -> None:
    if not args.html:
        return
    render_html(result, path=args.html, run_index=args.run or 0)
    print(f"\nwrote HTML report to {args.html}")


def _evaluate(*args: Any, out: Optional[str] = None, **kwargs: Any) -> Any:
    """``evaluate()`` and an exit code: Ctrl-C gives the finished runs' result
    and 130; a failed coordination analysis gives the plain evaluation and 1."""
    try:
        return evaluate(*args, **kwargs), 0
    except RunInterrupted as exc:
        return exc.result, 130
    except AnalysisFailed as exc:
        cause = exc.__cause__
        done = f"written to {out}" if out else "shown"
        print(f"swarmeval: coordination analysis failed: {type(cause).__name__}: {cause}; plain evaluation {done}",
              file=sys.stderr)
        return exc.result, 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .examples.research_swarm import build_benchmark, make_swarm
    bench = build_benchmark()
    _check_html_run(args, len(bench) * args.trials)
    result, code = _evaluate(make_swarm, bench, out=args.out, trials=args.trials, clock_factory=SimClock,
                             progress=True, **_analysis_kwargs(args))
    print(result.report())
    _write_outputs(result, args.out)
    if args.trace and result.runs.trajectories:
        print()
        print(render_trajectory(result.runs.trajectories[0]))
    if args.html and 0 < len(result.runs) <= (args.run or 0):
        # only an interrupt leaves fewer runs than the upfront check allowed; keep its exit code
        # (with 0 runs the HTML is a summary-only page, so it is still written)
        print(f"swarmeval: --run {args.run or 0} out of range (result has {len(result.runs)} runs); "
              "HTML report not written", file=sys.stderr)
        return code
    if args.html and args.run is not None and not len(result.runs):
        print(f"swarmeval: --run {args.run} ignored (no runs to show); wrote a summary-only HTML report",
              file=sys.stderr)
    _write_html(result, args)
    return code


def cmd_run(args: argparse.Namespace) -> int:
    swarm = _load_obj(args.swarm)
    bench = _load_benchmark(args.benchmark)
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    result, code = _evaluate(swarm, bench, out=args.out, trials=args.trials, seed=args.seed,
                             clock_factory=_clock_factory(args.clock), label=args.label, progress=True,
                             pricing=pricing, capture_text=not args.no_capture_text, **_analysis_kwargs(args))
    print(result.report())
    _write_outputs(result, args.out)
    return code


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
    _check_html_run(args, len(runs))
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    code = 0
    try:
        result = analyze_runs(runs, pricing=pricing, **_analysis_kwargs(args))
    except Exception as exc:  # every run failed the same way: the runs are on disk, show what still holds
        _analysis_failed(exc)
        result, code = analyze_runs(runs, pricing=pricing, coordination=False), 1
        result.coordination_errors = failure_errors(exc, runs)
    print(result.report())
    if args.trace is not None:
        if not 0 <= args.trace < len(runs):
            raise SystemExit(f"swarmeval: --trace {args.trace} out of range (file has {len(runs)} runs)")
        print()
        print(render_trajectory(runs.trajectories[args.trace]))
    _write_html(result, args)
    return code


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare
    base, cand = _load_runs_or_exit(args.baseline), _load_runs_or_exit(args.candidate)
    pricing = PricingTable.load(args.pricing) if args.pricing else None
    kw = _analysis_kwargs(args)
    code = 0
    try:
        cmp = compare(base, cand, pricing=pricing, **kw)
    except ValueError as exc:
        raise SystemExit(f"swarmeval: {exc}")
    except Exception as exc:  # every run failed the same way: compare what still holds
        _analysis_failed(exc)
        cmp, code = compare(base, cand, pricing=pricing, **{**kw, "coordination": False}), 1
    print(cmp.report())
    if args.out:
        cmp.save(args.out)
        print(f"\nwrote {args.out}")
    return code


def cmd_export(args: argparse.Namespace) -> int:
    runs = _load_runs_or_exit(args.path)
    n = export_events(runs.trajectories, args.out, args.format or "")
    print(f"wrote {n} events to {args.out}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    runs = _load_runs_or_exit(args.path)
    if not 0 <= args.run < len(runs):
        raise SystemExit(f"swarmeval: --run {args.run} out of range (file has {len(runs)} runs)")
    traj = runs.trajectories[args.run]
    graph = build_graph(traj)
    kw = _analysis_kwargs(args)
    cm = None
    if kw["coordination"]:
        cm = compute_coordination(traj, graph=graph, similarity=kw["similarity"],
                                  smell_config=kw["smell_config"])
    text = export_graph(graph, args.format, view=args.view, coordination=cm)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.format} ({args.view} view) of run {traj.run_id} to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="swarmeval", description="Record and evaluate multi-agent systems.")
    p.add_argument("--version", action="version", version=f"swarmeval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the built-in simulated research swarm")
    d.add_argument("--trials", type=int, default=3)
    d.add_argument("--out", default=None, help="directory for JSON/JSONL outputs")
    d.add_argument("--trace", action="store_true", help="also print the first trajectory")
    _add_html_args(d)
    _add_analysis_args(d)
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
    _add_analysis_args(r)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="re-analyse a saved run set")
    rp.add_argument("path")
    rp.add_argument("--trace", type=int, default=None, metavar="N", help="also print trajectory N")
    rp.add_argument("--pricing", default=None, help="re-price every LLM call with this JSON pricing table")
    _add_html_args(rp)
    _add_analysis_args(rp)
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
    _add_analysis_args(cp)
    cp.set_defaults(fn=cmd_compare)

    g = sub.add_parser("graph", help="export one run's execution graph (Mermaid, DOT or JSON)")
    g.add_argument("path", help="saved RunSet or EvaluationResult JSON")
    g.add_argument("--run", type=int, default=0, metavar="N", help="which run (default 0)")
    g.add_argument("--view", choices=list(VIEWS), default="agents",
                   help="agents: who talked to whom; dag: work units and the critical path")
    g.add_argument("--format", choices=sorted(GRAPH_FORMATS), default="mermaid")
    g.add_argument("--out", default=None, help="output file (default: stdout)")
    _add_analysis_args(g)
    g.set_defaults(fn=cmd_graph)

    args = p.parse_args(argv)
    if args.cmd in ("demo", "report") and args.run is not None and not args.html:
        (d if args.cmd == "demo" else rp).error("--run requires --html")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
