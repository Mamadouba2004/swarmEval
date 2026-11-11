"""SwarmEval -- evaluation and observability for multi-agent AI systems.

Phase 1: trajectory recording, standardized task format, basic
performance/efficiency metrics, run comparison, text reports and
JSON/JSONL/Parquet export.  Phase 2 (in progress): execution graph,
utilization and observed contribution (``build_graph``, ``compute_coordination``).

Quick start::

    from swarmeval import evaluate
    result = evaluate(my_swarm, benchmark, trials=3)
    print(result.report())
"""
from . import evaluators
from .evaluate import AnalysisFailed, EfficiencySummary, EvaluationResult, analyze_runs, evaluate
from .recorder import AgentSpan, LLMCall, LLMResult, Message, RunContext, SimClock, ToolCall, WallClock
from .runner import FunctionSwarm, RunOptions, RunSet, Swarm, SwarmRunner
from .schema import Artifact, Benchmark, Evaluation, Event, EventType, Span, Task, Trajectory
from .metrics import (CoordinationMetrics, CoordinationSummary, EfficiencyMetrics, PerformanceMetrics,
                      compute_coordination, compute_efficiency, compute_performance, summarize_coordination)
from .graph import ExecutionGraph, build_graph
from .export import read_events_jsonl, write_events_jsonl
from .report import render_comparison, render_text, render_trajectory
from .compare import Comparison, compare

__version__ = "0.1.0"

__all__ = [
    "evaluate", "analyze_runs", "EvaluationResult", "AnalysisFailed", "EfficiencySummary",
    "Task", "Benchmark", "Trajectory", "Event", "EventType", "Span", "Artifact", "Evaluation",
    "RunContext", "AgentSpan", "Message", "LLMResult", "LLMCall", "ToolCall", "SimClock", "WallClock",
    "Swarm", "FunctionSwarm", "SwarmRunner", "RunOptions", "RunSet",
    "PerformanceMetrics", "EfficiencyMetrics", "compute_performance", "compute_efficiency",
    "CoordinationMetrics", "CoordinationSummary", "compute_coordination", "summarize_coordination",
    "ExecutionGraph", "build_graph",
    "read_events_jsonl", "write_events_jsonl", "render_text", "render_trajectory", "render_comparison",
    "compare", "Comparison",
    "evaluators", "__version__",
]
