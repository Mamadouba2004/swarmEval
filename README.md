# SwarmEval

> Evaluation and observability for multi-agent AI systems.

SwarmEval evaluates a multi-agent system not only on whether it produced the right answer, but on **how the agents worked together to produce it**. The foundation for that is trajectory observability: every agent invocation, LLM call, tool call, message and artifact is recorded as a structured event, and every metric is computed from those events.

This is **Phase 1** of the [roadmap](ROADMAP.md): observability and basic evaluation. Coordination analysis, causal ablation and optimization are later phases and are not in this release.

```
Phase 1  Trajectory recorder · task format · metrics · report · run comparison · JSON/Parquet export · agent interface   ← this release
Phase 2  Execution graph · utilization · communication · redundancy · critical path · swarm smells
Phase 3  Ablation · failure injection · counterfactuals · confidence intervals
Phase 4  Architecture recommendations · validated optimization
```

Python 3.12+, no runtime dependencies. Optional Anthropic SDK adapter. The design charter is in [docs/DESIGN.md](docs/DESIGN.md); the full research write-up is in [research.md](research.md).

---

## Install

```bash
pip install -e ".[dev]"        # from this repo
pip install -e ".[anthropic]"  # optional: Anthropic SDK adapter
```

## 60-second demo

The repo ships a simulated research swarm (no API keys) with a coordinator, three researchers, a critic and a synthesizer:

```bash
swarmeval demo --trials 3 --trace --out runs/demo
```

```
╭──────────────────────────────────────────────────────────────╮
│                       SWARM EVALUATION                       │
├──────────────────────────────────────────────────────────────┤
│ baseline  ·  research_briefs  ·  8 tasks × 3 trials = 24 runs│
├──────────────────────────────────────────────────────────────┤
│ Task Success                                19/24  (79.2%)   │
│ Mean Score                                           0.931   │
│ Constraint Pass                                     100.0%   │
├──────────────────────────────────────────────────────────────┤
│ Cost / Run                                           $0.12   │
│ Cost / Successful Task                               $0.16   │
│ Tokens / Run                16,996  (in 13,916 / out 3,080)  │
│ LLM Calls / Run                                       11.0   │
│ Tool Calls / Run                                       6.0   │
├──────────────────────────────────────────────────────────────┤
│ Wall Clock / Run                                40.0s ±0.0   │
│ Agent Time / Run                                     54.6s   │
╰──────────────────────────────────────────────────────────────╯

AGENTS (per run, mean)
  agent           role        tokens                    llm tools  msgs    time      cost
  synthesizer     synthesizer █████████░░░░░░░   58%  1.0   0.0   0.0   13.4s   $0.0629
  coordinator     coordinator █████░░░░░░░░░░░   30%  3.0   0.0   5.0   11.6s   $0.0370
  ...
```

`--trace` prints one trajectory event by event (timestamps, models, tokens, cost, messages, artifacts). `--out` writes `baseline.json` (the run set plus metrics) and `baseline.events.jsonl` (one event per line).

---

## Instrumenting a swarm (the agent interface)

A swarm is anything with `run(task, ctx)`. Everything the swarm does goes through the `RunContext`, which is the recorder.

```python
from swarmeval import Swarm, Task, Benchmark, evaluate

class MySwarm(Swarm):
    def run(self, task, ctx):
        with ctx.agent("planner", role="coordinator", model="claude-opus-5") as planner:
            plan = planner.llm(call_model, task.input, prompt=task.input)   # LLMResult carries real usage
            planner.send("researcher", plan, kind="task_assignment")

        with ctx.agent("researcher", role="research", model="claude-sonnet-5") as r:
            assignment = r.receive()[0]
            hits = r.call_tool("web_search", search, query=assignment.content)
            notes = r.produce("notes", summarise(hits))
            r.send("writer", notes.content, kind="result", artifacts=[notes])

        with ctx.agent("writer", role="synthesizer") as w:
            msg = w.receive()[0]
            with w.llm_call(model="claude-opus-5", input_artifacts=msg.artifacts, prompt=msg.content) as call:
                answer = call_model(msg.content)               # any client
                call.input_tokens, call.output_tokens = answer.usage.input, answer.usage.output
                call.output = answer.text
            final = w.produce("answer", answer.text)
        ctx.set_output(answer.text, artifacts=[final])
        return answer.text

bench = Benchmark("qa", [Task("q1", "What is …?", expected=["keyword"],
                              evaluator={"type": "contains_all", "keywords": ["keyword"]})])
result = evaluate(MySwarm(), bench, trials=3)
print(result.report())
result.save("runs/qa.json")
```

| Call | Records |
|---|---|
| `ctx.agent(id, role, model)` | an agent invocation span (nest them for hierarchies; `async with` works too) |
| `ctx.parallel(*fns)` / `await ctx.gather(*coros)` | concurrent branches: each gets its own `SimClock` timeline and seeded `ctx.rng`, and the parent resumes at the latest branch end |
| `span.llm(fn, …)` / `span.llm_call(...)` | an LLM call with model, tokens (incl. prompt-cache reads/writes), cost, latency, prompt, `input_artifacts` |
| `span.call_tool(name, fn, **args)` / `span.tool_call(...)` | a tool call with arguments, result, latency and status |
| `await span.allm(...)` / `await span.acall_tool(...)` | the same for async functions (the sync versions raise if given one) |
| `span.send(to, content, kind, artifacts, reply_to=)` / `span.receive()` | messages through a per-agent mailbox; each message gets an id, and `receive()` records a `message_receive` event per message read (`inbox()` only peeks) |
| `span.produce(name, content, derived_from=[...])` / `span.consume(artifact)` | artifacts, what they were made from, and who used them |
| `span.wait(reason, seconds, on=...)` / `span.waiting(reason, on=...)` | explicit waiting, optionally on Messages, Artifacts or agent ids |
| `span.note(text)` / `span.error(msg)` | annotations and errors |

Return an `LLMResult(text, input_tokens, output_tokens, model)` from the function you pass to `span.llm` to record real usage; otherwise tokens are estimated from the prompt (`prompt=`, or the function's arguments) and the output, and flagged as estimated. A lone `prompt=` is passed to a function that takes an argument, so `span.llm(fn, prompt=p)` works. Models missing from the pricing table get a cost of `unknown`: they are left out of totals (marked `*`), never counted as `$0`. Only dated or provider-prefixed forms of a listed model id share its price.

**Anthropic SDK**: `swarmeval.integrations.anthropic.instrument_client(client)` records every `messages.create` / `beta.messages.create` made inside an agent span with real usage (sync or async client); inside `span.llm` it fills in that call rather than recording it twice. `anthropic_llm(client, model)` returns a prompt → `LLMResult` callable for `span.llm` (async for an async client, for `span.allm`).

**Provenance** (schema v2, `trajectory.metadata["schema_version"] == 2`): the coordination analysis follows messages from send to receive and artifacts from producer to consumer. Carrying an artifact in a message is transport, not use; use is `consume(...)`, an LLM/tool call's `input_artifacts`, or another artifact's `derived_from`. `derived_from` left out means "not recorded" (`[]` means "made from nothing"). Phase 1 run files still load; their message edges are marked inferred.

`evaluate()` also accepts a `Swarm` class, a bare `run(task, ctx)` function (sync or `async`) or a zero-argument factory. Use `clock_factory=SimClock` for simulated swarms that advance a virtual clock. An `evaluator=` passed to `evaluate()` overrides the task and benchmark evaluators. `capture_text=False` keeps only short previews of prompts, outputs, tool arguments and artifacts in everything recorded and saved.

---

## Task format

A `Task` has `task_id`, `input`, optional `expected`, `success_criteria`, `constraints`, `reference`, `category`, `difficulty` and an `evaluator`. Evaluators are JSON specs so benchmarks are plain files:

```json
{"name": "qa", "tasks": [
  {"task_id": "q1", "input": "…", "evaluator": {"type": "contains_all", "keywords": ["a", "b"], "forbidden": ["c"]}},
  {"task_id": "q2", "input": "…", "expected": "42", "evaluator": {"type": "exact_match"}}
]}
```

Built-in evaluators: `exact_match`, `contains_all` / `contains_any` (with `forbidden`), `regex`, `numeric`, `tool_use` (checks the trajectory), `constraints`, `composite`, or any callable `fn(task, output, trajectory)`. `Benchmark.load(path)` accepts JSON or JSONL. `Benchmark.validate()` (run before any swarm run) checks ids and fields and builds every evaluator, so an unknown task field, an unknown evaluator type or key, a missing required key, or an evaluator with nothing to check (e.g. `contains_all` without keywords) fails up front. `Benchmark.save()` refuses evaluators that cannot be rebuilt from JSON (plain callables); register a spec type for them.

## What gets measured

| Layer | Metrics |
|---|---|
| **Performance** | success rate, mean score, constraint pass rate, per-task and per-trial breakdown, crashed runs |
| **Efficiency** | input/output tokens, LLM and tool call counts, messages, cost (pricing table), cost per successful task, wall clock, aggregate agent time, waiting time, per-agent tokens/cost/calls/time |
| **Coordination** (Phase 2, in progress) | execution graph (`build_graph`: agent graph + work-unit DAG), provenance coverage, per-agent utilization shares, observed contribution (final-output lineage, artifacts consumed by others, `dead` only when provenance coverage is complete) -- `result.per_run_coordination`, `result.coordination` |

Repeated trials use seeds derived from `(seed, task, trial)` so a run set is reproducible. Means and spreads are reported; confidence intervals are Phase 3.

## Outputs

* `EvaluationResult.report()` – text report (above); `.summary()` / `.to_dict()` / `.save(path)` – JSON including the full run set.
* `swarmeval.export.write_events_jsonl(trajectories, path)` / `write_events_parquet(...)` (needs `swarmeval[parquet]`) – flat event stream for pandas / DuckDB; `swarmeval export runs.json events.parquet` converts a saved run set.
* `RunSet.save/load`, `Trajectory.save/load` – trajectories are the source of truth; `swarmeval report runs.json` recomputes everything from them, and `--pricing prices.json` (or `analyze_runs(runs, pricing=...)`) re-prices every call from its recorded model and tokens.
* `render_trajectory(traj)` – the event-by-event text trace.

## Comparing runs

`compare(baseline, candidate)` takes two `EvaluationResult`s (or `RunSet`s) and reports how the candidate differs:

```python
from swarmeval import compare, evaluate
base = evaluate(flat_swarm, bench, trials=5, label="flat")
cand = evaluate(hierarchical_swarm, bench, trials=5, label="hierarchical")
cmp = compare(base, cand)          # pricing=PricingTable(...) prices both sides alike
print(cmp.report())
cmp.improved, cmp.regressed, cmp.is_tradeoff, cmp.flips["lost"]
```

* Both sides are re-analysed on the tasks they share; tasks present on only one side are listed and left out.
* Every metric (success, score, crashes, cost per run / per success, tokens, calls, messages, wall clock, agent time, errors) gets its own baseline, candidate, delta and verdict (`improved` / `regressed` / `unchanged`), so a cheaper but less accurate candidate shows up as a **tradeoff** rather than one blended score.
* Runs are paired by task and trial (same derived seed), giving fail→pass and pass→fail flips; per-task and per-agent rows show where the change happened, including agents that were added or removed.
* A cost that is incomplete on either side is reported as unknown, never compared as $0.
* Warnings flag different benchmarks or trial counts, mismatched seeds, and fewer than 3 trials. Differences are measured, not tested for significance (that is Phase 3).

## CLI

```bash
swarmeval demo [--trials N] [--out DIR] [--trace]
swarmeval run    --swarm pkg.mod:factory --benchmark bench.json --trials 3 --out runs [--clock sim] [--pricing p.json] [--no-capture-text]
swarmeval report runs/baseline.json [--trace N] [--pricing p.json]
swarmeval export runs/baseline.json events.parquet [--format jsonl|parquet]
swarmeval compare runs/flat.json runs/hierarchical.json [--pricing p.json] [--out cmp.json]
```

## Layout

```
swarmeval/
  schema.py        events, artifacts, spans, tasks, benchmarks, trajectories
  recorder.py      RunContext / AgentSpan: the agent interface and recorder
  runner.py        Swarm, SwarmRunner, RunSet, repeated trials, seeds
  evaluators.py    deterministic evaluators with JSON specs
  graph/           execution graph: agent graph and work-unit DAG
  metrics/         performance, efficiency, contribution, coordination
  compare.py       run-to-run comparison
  report/          text report, comparison report and trajectory trace
  export/          JSONL and Parquet events
  integrations/    Anthropic SDK adapter
  examples/        simulated research swarm + benchmark used by the demo and tests
  pricing.py       model pricing table
  stats.py         descriptive statistics
  cli.py
tests/
```

## Development

```bash
docker run --rm -v "$PWD":/w -w /w python:3.12-slim \
  sh -c 'pip install -q -e ".[dev,parquet]" && pytest -q'
```

The Parquet tests skip when `pyarrow` is not installed.
