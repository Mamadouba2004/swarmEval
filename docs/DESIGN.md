# SwarmEval — Design Document

> An evaluation, observability, diagnostic, and causal analysis framework for multi-agent AI systems.

This document is the project's charter: what SwarmEval measures, why, and in what order it is being built. The README covers usage; this covers the reasoning. Module references below name where each piece lives or will live; see §11 for what is implemented so far.

## 1. Core idea

Given a task and a multi-agent system, SwarmEval records the execution as a structured trajectory and analyses it at multiple levels:

```
Task → Outcome → Trajectory → Agent behaviour → Communication → Dependencies → Coordination → Architecture
```

producing quantitative evaluation (what happened), qualitative diagnosis (why), and counterfactual analysis (what would happen if the system changed).

## 2. Foundational principle

Trajectory observability and causal ablation are the foundation; everything else builds on them.

```
Trajectory Observability + Causal Ablation
              ↓
      Reliable Understanding
              ↓
      Coordination Analysis → Diagnostics → Resilience Analysis → Architecture Evaluation → Optimization
```

**Observability** answers *what happened*: agent invocations, LLM calls, tool calls, inputs, outputs, messages, dependencies, timing, tokens, costs, errors, artifacts, downstream usage, evaluation signals.

> If the execution cannot be represented as structured events, it cannot be reliably analysed.

**Causal ablation** answers *what changed because a component was present*: remove an agent, tool, channel, model, context, ordering or budget; re-run; compare. This separates *observed activity* from *causal contribution*. An agent can generate thousands of tokens and contribute nothing; another can do very little and prevent a major downstream failure.

Implementation: `swarmeval.schema` (events) and `swarmeval.recorder` (capture) exist today; `swarmeval.causal` (ablation) is Phase 3.

## 3. Measurement layers

```
┌─────────────────────────────┐
│      TASK PERFORMANCE       │  success, correctness, constraints, quality
├─────────────────────────────┤
│         EFFICIENCY          │  cost, latency, critical path
├─────────────────────────────┤
│    COORDINATION QUALITY     │  utilization, communication, redundancy, dependencies, parallelism
├─────────────────────────────┤
│   ROBUSTNESS & CAUSALITY    │  ablation, failure injection, resilience
└─────────────────────────────┘
```

Separate layers prevent a system from scoring "good" just because it solved the task while being expensive or badly coordinated.

### Layer 1 — Task performance (`metrics/performance.py`, `evaluators.py`)
Success rate, correctness score, constraint satisfaction, tool-use correctness, groundedness, task-specific criteria. Deterministic evaluators and LLM judges; LLM judging supports multiple judgments, agreement and confidence rather than treating one model's verdict as truth.

### Layer 2 — Efficiency (`metrics/efficiency.py`)
Cost: input/output/total tokens, LLM and tool call counts, estimated API cost, cost per successful task. Latency: wall clock, critical path, agent compute time, waiting time, total work. Wall-clock time is not total work: a swarm may do 60s of aggregate work in 15s through parallelism.

### Layer 3 — Coordination quality (`metrics/coordination.py`, `graph/`)
The core analytical layer: did multiple agents actually produce useful coordination, or only complexity?

* **Agent utilization** — actions, tokens, tools, artifacts, messages, downstream references. Utilization is *not* contribution.
* **Agent contribution** — *observed* (provenance: artifacts consumed downstream, messages used, lineage of the final output) vs *marginal* (ablation). SwarmEval never says "Agent X contributed 12%"; it says "removing Agent X reduced task success by 12 points under this configuration".
* **Redundant work** (`metrics/redundancy.py`) — duplicate tool calls, similar artifacts, shared sources; *accidental* redundancy distinguished from *intentional verification* (critic/verifier roles, `intent="verification"`) and retries. Tool redundancy = accidental duplicate calls (same tool, same args; strings exact, key order ignored) ÷ tool calls; artifact redundancy (a separate number) = artifacts matching an earlier other-producer artifact by hash or similarity ÷ artifacts. Similarity is pluggable (`swarmeval/similarity.py`): lexical shingle Jaccard by default (stdlib), embeddings via the optional `swarmeval[embeddings]` extra or a user `embed_fn`.
* **Communication overhead** (`metrics/communication.py`) — message counts and body tokens (attached artifacts excluded: counted once, at production), category split (work product / information / coordination), repeated and broadcast messages, fan-in/out, relay fraction, hub share, used-message ratio. **Communication / Work** = message body tokens as a share of LLM tokens (Σ body ÷ Σ prompt + output). Communication is not automatically waste.
* **Dependency efficiency** (`metrics/dependency.py`) — critical path (heaviest DAG path, LLM/tool durations), depth (calls on it), cycles (strongly connected groups of ≥3 agents once replies are set aside: a request/response round trip is never a cycle), isolated agents.
* **Parallelism** (`metrics/dependency.py`) — actual = work ÷ wall, potential = work ÷ critical path, headroom = wall − critical path; *unrecorded* time (wall minus the union of call and wait intervals) is reported separately so idle time is not read as headroom. The gap is an optimization opportunity, subject to rate limits and deliberate synchronisation.
* **Context bloat** (`metrics/context.py`) — per LLM call, prompt tokens vs tokens traceable to recorded inputs (task + used artifacts + received message bodies + earlier outputs of the span + `system_tokens`). Untracked inputs show up as excess, so the ratio reads "untraceable", not "useless".

### Layer 4 — Robustness & causality (`causal/`)
Agent, tool and architecture ablation; failure injection (crash, delay, dropped/corrupted messages, malformed tool output, tool errors, LLM errors, budgets); resilience = success under failure ÷ normal success; quality retention, recovery cost and latency.

## 4. Trajectory graph (`graph/builder.py`)

Every execution is a graph. Nodes carry timing, model, tokens, tools, inputs, outputs, errors and cost; edges carry sender, receiver, timestamp, tokens, dependency type, and whether downstream behaviour used them. Two views:

* **Agent graph** — message, artifact and invocation edges; supports fan-in/out, cycles, density, isolated agents, bottlenecks.
* **Work-unit DAG** — LLM and tool calls with inferred dependencies (sequential order within a span, messages, artifact provenance, nesting); supports critical path, depth, theoretical parallelism.

Dependencies are inferred from what was recorded. Unrecorded channels make the DAG under-count dependencies and over-estimate parallelism.

Provenance recorded since schema v2 (`trajectory.metadata["schema_version"] = 2`; absent = Phase 1 data): message ids on `message` events and a `message_receive` event per message read (`reply_to` links answers), `Artifact.derived_from` (`None` = not recorded, `[]` = no inputs), `wait(..., on=)` targets, and `metadata["within_call"]` on a call made inside another call of its span (so nesting is not confused with gathered calls), and in `span.metadata` on an agent span opened inside a call (an agent used as a tool: its calls count once in `work_s`). DAG rules: nested calls are excluded; within a span each unit depends on the *finished frontier* before it, so overlapping calls are not chained; child spans connect to their parent's frontier and back; messages link send → receive (or, without a receive event — Phase 1 data, or a message only read with `inbox()` — → the receiver's next unit, flagged inferred); artifacts link producer → genuine consumption only (forwarding is not consumption); waits link to what they waited on. `Coverage` reports how much of this was recorded.

## 5. Swarm smells (`diagnostics/smells.py`)

ESLint for agent architectures. Each smell is a *hypothesis from trajectory evidence*, reported as observation → hypothesis → intervention to test:

| Smell | Signal |
|---|---|
| Redundant delegation | several agents send one agent similar requests |
| Agent ping-pong | repeated alternating exchanges between two agents (a new, unanswering request starts a fresh exchange, so answered delegations are not ping-pong) |
| Over-agentization | many agents for a simple task; many low-contribution agents |
| Under-specialization | "specialised" agents doing overlapping work |
| Coordinator bottleneck | most messages pass through one agent; high token share; on the critical path; no bypass |
| Premature synthesis / late information | an agent reasons before its inputs exist, or receives them after its last call |
| Context bloat | input tokens far exceed relevant tokens |
| Dead / zero-value agent | output never used downstream and absent from the final output's lineage |
| Serialized execution | large gap between potential and actual parallelism |
| Communication overhead | message tokens are a large share of LLM tokens |

Cross-run aggregation (`diagnostics/engine.py`) reports prevalence: each run counts once per smell, agents once per run, grouped by smell, sorted by severity then prevalence. Thresholds (`SmellConfig`, JSON-overridable) are initial guesses; they are to be validated against Phase 3 ablations (does removing the flagged agent / capping the flagged exchange change success and cost as the hypothesis predicts?). Evidence that rests on timing-inferred edges is marked `inferred` and its severity capped. Wording is checked by a test: no causal claims, no "contributed N%".

## 6. Counterfactual evaluation (`causal/`)

Instead of "did the swarm succeed?" ask "what happens when we intervene?". Interventions: remove agent/tool, disable channel, change model, restrict context/budget, change ordering or parallelism (swarm params), introduce failures. The counterfactual runner re-runs the same benchmark with the same seeds and compares pairwise with bootstrap confidence intervals. Each agent additionally gets its own random stream so removing one agent does not perturb the others (experimental control).

**Intelligence vs coordination**: compare strong single agent, strong swarm, weak swarm, weak agents without coordination to separate capability from coordination.

**Efficiency frontier**: expose the success/cost trade-off as a Pareto frontier instead of a composite score; an architecture is interesting when it moves the frontier.

## 7. Reports (`report/`)

Health report with separate dimensions (no universal score), agent table, diagnostics with observation / hypothesis / recommendation kept explicit, and causal tables. HTML output adds the execution graph, a timeline with the critical path highlighted, and frontier plots. Machine-readable: JSON, JSONL events, OpenTelemetry-compatible spans.

## 8. From evaluation to optimization (`optimize/`)

Observe → diagnose → generate candidate architectures → predict → benchmark → compare → validate → deploy → observe again. Candidates are produced from diagnostics with heuristic predictions, labelled hypotheses, then run as counterfactuals and accepted, rejected or marked inconclusive on measured deltas.

## 9. Evaluation validity (`validity.py`, `stats.py`)

Repeated trials and distributions rather than single runs; Wilson and bootstrap confidence intervals; evaluator agreement; deterministic seeds derived from (seed, task, trial); experimental controls; benchmark leakage detection (answer fragments in prompts or tool arguments before any tool result supplied them).

## 10. Benchmark design (`schema.Task`, `schema.Benchmark`)

Task = input, expected behaviour, success criteria, constraints, evaluator spec, optional reference, difficulty, category. Benchmarks are plain JSON/JSONL so different architectures are evaluated under identical conditions.

## 11. Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 Observability & evaluation | reproducible trace + evaluation report | **done** (`schema`, `recorder`, `runner`, `evaluators`, `metrics/performance`, `metrics/efficiency`, `report/text`, `export/jsonl`, `export/parquet`, `compare` (run-to-run comparison, research.md §27), `integrations/anthropic`, CLI incl. `export` and `compare`); sync and async swarms |
| 2 Coordination analysis | explain where the swarm is inefficient and why | in progress: provenance schema v2; Graph Analyzer (Layer 3: `graph/builder`, `graph/export` Mermaid/DOT/JSON, `metrics/dependency`); utilization, observed contribution, communication, redundancy, context (`metrics/*`, `similarity`); Diagnostic Engine (Layer 3 → report: `diagnostics/smells`, `diagnostics/engine`); COORDINATION / OBSERVED CONTRIBUTION / DIAGNOSTICS text report, coordination rows in `compare`, `swarmeval graph`; standalone HTML report planned |
| 3 Causal & research features | estimate which components actually matter | planned (`causal/`, confidence intervals in `stats`, `validity`) |
| 4 Optimization | experimentally validated architecture improvements | planned (`optimize/`) |

Phase 1 records everything later phases need (messages with kinds, artifact provenance via `input_artifacts` and `consume`, span nesting, timestamps) so trajectories captured now remain analysable when those layers land.

## 12. Central research question

> When does adding more agents actually make an AI system better?

SwarmEval investigates capability + coordination + parallelism + specialization − communication − redundancy − cost − latency, and the deeper questions of agent value, coordination value, scaling, specialization, redundancy, robustness, architecture generalisation and economics.

## 13. Principles

1. Observe before optimizing.
2. Activity is not contribution.
3. Correlation is not causation.
4. Recommendations are hypotheses.
5. Preserve underlying measurements.
6. Compare against strong baselines.
7. Measure the trade-offs.
8. Reproduce experiments.

## 14. Thesis

Trajectory observability provides the evidence. Causal ablation provides the experimental mechanism. Coordination analysis provides the explanation. Diagnostics provide the hypotheses. Benchmarking provides the validation. Optimization provides the improvement.

Multi-agent engineering moves from *build → run → look at success rate → guess* to *observe → measure → understand → hypothesise → intervene → measure causal effect → validate → optimize → repeat*.
