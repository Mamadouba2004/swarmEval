"""Swarm smells and their prevalence over runs (Phase 2: coordination analysis)."""
from .engine import DiagnosticsReport, SmellPrevalence, summarize_diagnostics
from .smells import (
    SEVERITIES,
    SMELLS,
    AgentPingPongConfig,
    CommunicationOverheadConfig,
    ContextBloatConfig,
    CoordinatorBottleneckConfig,
    DeadAgentConfig,
    Diagnostic,
    OverAgentizationConfig,
    RedundantDelegationConfig,
    SerializedExecutionConfig,
    SmellConfig,
    SmellContext,
    UnderSpecializationConfig,
    detect_smells,
    load_smell_config,
    register_smell,
    resolve_smell_config,
)

__all__ = ["Diagnostic", "SmellConfig", "SmellContext", "SEVERITIES", "SMELLS", "detect_smells", "load_smell_config",
           "register_smell", "resolve_smell_config", "DiagnosticsReport", "SmellPrevalence", "summarize_diagnostics",
           "RedundantDelegationConfig", "AgentPingPongConfig", "OverAgentizationConfig", "UnderSpecializationConfig",
           "CoordinatorBottleneckConfig", "ContextBloatConfig", "DeadAgentConfig", "SerializedExecutionConfig",
           "CommunicationOverheadConfig"]
