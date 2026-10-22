"""SwarmEval -- evaluation and observability for multi-agent AI systems.

Phase 1: trajectory recording, standardized task format, basic
performance/efficiency metrics, run comparison, text reports and
JSON/JSONL/Parquet export.  Phase 2 (in progress): execution graph,
utilization, observed contribution, communication overhead, redundancy,
critical path and parallelism, context bloat and swarm smells
(``build_graph``, ``compute_coordination``, ``detect_smells``), and graph
export (Mermaid / DOT / JSON).

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
from .metrics import (CommunicationMetrics, ContextMetrics, CoordinationMetrics, CoordinationSummary,
                      DependencyMetrics, EfficiencyMetrics, PerformanceMetrics, RedundancyMetrics,
                      compute_communication, compute_context, compute_coordination, compute_dependency,
                      compute_efficiency, compute_performance, compute_redundancy, summarize_coordination)
from .diagnostics import (Diagnostic, DiagnosticsReport, SmellConfig, SmellPrevalence, detect_smells,
                          load_smell_config, summarize_diagnostics)
from .similarity import EmbeddingSimilarity, LexicalSimilarity, Similarity, resolve_similarity
from .graph import ExecutionGraph, build_graph, export_graph, to_dot, to_json, to_mermaid
from .export import read_events_jsonl, write_events_jsonl
from .report import render_comparison, render_coordination, render_text, render_trajectory
from .compare import Comparison, compare

__version__ = "0.1.0"

__all__ = [
    "evaluate", "analyze_runs", "EvaluationResult", "AnalysisFailed", "EfficiencySummary",
    "Task", "Benchmark", "Trajectory", "Event", "EventType", "Span", "Artifact", "Evaluation",
    "RunContext", "AgentSpan", "Message", "LLMResult", "LLMCall", "ToolCall", "SimClock", "WallClock",
    "Swarm", "FunctionSwarm", "SwarmRunner", "RunOptions", "RunSet",
    "PerformanceMetrics", "EfficiencyMetrics", "compute_performance", "compute_efficiency",
    "CoordinationMetrics", "CoordinationSummary", "compute_coordination", "summarize_coordination",
    "CommunicationMetrics", "compute_communication", "RedundancyMetrics", "compute_redundancy",
    "Similarity", "LexicalSimilarity", "EmbeddingSimilarity", "resolve_similarity",
    "DependencyMetrics", "compute_dependency", "ContextMetrics", "compute_context",
    "Diagnostic", "DiagnosticsReport", "SmellConfig", "SmellPrevalence", "detect_smells", "load_smell_config",
    "summarize_diagnostics",
    "ExecutionGraph", "build_graph", "export_graph", "to_mermaid", "to_dot", "to_json",
    "read_events_jsonl", "write_events_jsonl", "render_text", "render_trajectory", "render_comparison",
    "render_coordination",
    "compare", "Comparison",
    "evaluators", "__version__",
]
