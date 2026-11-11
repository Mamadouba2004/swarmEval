"""A simulated multi-agent research swarm -- no API keys required.

Used by ``swarmeval demo`` and the tests.  It shows how a swarm is
instrumented through the ``RunContext`` / ``AgentSpan`` interface: agent
spans, LLM calls with token usage, tool calls, messages and artifacts.

The swarm has a coordinator, three researchers, a critic and a synthesizer.
It is deliberately imperfect (the coordinator relays everything, two
researchers share a scope, the synthesizer receives far more context than it
needs) so that the coordination analysis planned for Phase 2 has something
to find.  Phase 1 only records and measures it.

The simulated clock (``SimClock``) is advanced explicitly so runs are instant
but timings look real; ``ctx.parallel`` gives each researcher its own
timeline, so their work overlaps under both SimClock and WallClock.
"""
from __future__ import annotations

import random
from functools import partial
from typing import Any, Dict, List, Optional

from swarmeval import Benchmark, RunContext, SimClock, Swarm, Task
from swarmeval.schema import estimate_tokens

MODEL_QUALITY = {"claude-opus-5": 0.985, "claude-sonnet-5": 0.96, "claude-haiku-4-5": 0.85}
MODEL_SPEED = {"claude-opus-5": 55.0, "claude-sonnet-5": 90.0, "claude-haiku-4-5": 160.0}   # output tok/s
WRONG_FACT_PICKUP = 0.5   # how often a researcher records the planted wrong fact

TOPICS = [
    ("Aurora protocol", "aurora"), ("Helix compiler", "helix"), ("Quasar storage", "quasar"),
    ("Lumen scheduler", "lumen"), ("Tidal cache", "tidal"), ("Basalt runtime", "basalt"),
    ("Meridian mesh", "meridian"), ("Cinder allocator", "cinder"), ("Sable index", "sable"),
    ("Orchid planner", "orchid"),
]

_FACT_TEMPLATES = [
    "{name} was first released in {year}",
    "{name} is maintained by the {org} working group",
    "{name} version {ver} added {feature}",
    "{name} is licensed under the {lic} license",
    "{name} processes roughly {n} requests per second per node",
    "{name} depends on the {dep} library",
]
_WRONG_TEMPLATE = "{name} was deprecated in {year}"


def build_world(n_tasks: int = 8, seed: int = 7) -> Dict[str, Dict[str, Any]]:
    """Deterministic knowledge base: each task has five sources with facts and
    one planted wrong fact in source s3."""
    rng = random.Random(seed)
    world: Dict[str, Dict[str, Any]] = {}
    for i in range(n_tasks):
        name, slug = TOPICS[i % len(TOPICS)]
        year = rng.randint(2012, 2023)
        vals = {"name": name, "year": year, "org": rng.choice(["Tern", "Kestrel", "Ibis", "Heron"]),
                "ver": f"{rng.randint(1, 5)}.{rng.randint(0, 9)}",
                "feature": rng.choice(["streaming replication", "incremental snapshots", "zero-copy IO", "adaptive batching"]),
                "lic": rng.choice(["Apache-2.0", "MIT", "BSD-3", "MPL-2.0"]),
                "n": rng.choice([12000, 48000, 90000, 250000]),
                "dep": rng.choice(["libwarp", "zephyrio", "corundum", "nimbus-rt"])}
        facts = [t.format(**vals) for t in _FACT_TEMPLATES]
        wrong = _WRONG_TEMPLATE.format(name=name, year=year + rng.randint(1, 3))
        extra = f"{name} has an experimental {rng.choice(['WASM', 'GPU', 'RISC-V'])} backend"
        sources = {"s1": [facts[0], facts[1]], "s2": [facts[2]], "s3": [facts[3], wrong], "s4": [facts[4], facts[5]],
                   "s5": [extra]}
        keywords = [vals["org"], str(vals["ver"]), vals["feature"], vals["lic"], str(vals["n"]), vals["dep"]]
        world[f"task_{i+1:02d}"] = {"name": name, "slug": slug, "sources": sources, "required": facts,
                                    "keywords": keywords, "wrong": wrong,
                                    "wrong_keyword": "deprecated", "extra": extra}
    return world


WORLD = build_world()


def build_benchmark(n_tasks: int = 8) -> Benchmark:
    """The standardized task format in use: input, expected keywords, success
    criteria, constraints and a JSON evaluator spec per task."""
    tasks = []
    for tid, w in list(WORLD.items())[:n_tasks]:
        tasks.append(Task(
            task_id=tid, category="research",
            input=f"Write a factual brief on the {w['name']}: release year, maintainer, notable version, "
                  f"license, throughput and dependencies.",
            expected=w["keywords"],
            success_criteria="All six facts present, no deprecated claim.",
            constraints=["Do not claim the project is deprecated."],
            evaluator={"type": "contains_all", "keywords": w["keywords"], "forbidden": [w["wrong_keyword"]]},
            difficulty="moderate",
        ))
    return Benchmark("research_briefs", tasks, metadata={"simulated": True})


class ResearchSwarm(Swarm):
    """coordinator -> researchers (parallel) -> coordinator -> critic -> coordinator -> synthesizer"""

    # -- simulation helpers ---------------------------------------------------
    @staticmethod
    def _advance(ctx: RunContext, seconds: float) -> None:
        clock = ctx.clock
        if isinstance(clock, SimClock):
            clock.advance(seconds)
        else:
            clock.sleep(seconds * 0.01)  # real clock: keep demos fast

    def _llm(self, ctx: RunContext, agent, prompt: str, out_tokens: int, extra_context_tokens: int = 0,
             input_artifacts=(), output: Optional[str] = None) -> str:
        """Simulate one LLM call and record it with realistic token counts and latency."""
        model = agent.model or "claude-sonnet-5"
        in_tokens = estimate_tokens(prompt) + extra_context_tokens
        with agent.llm_call(model=model, input_artifacts=input_artifacts, prompt=prompt) as call:
            self._advance(ctx, 0.35 + out_tokens / MODEL_SPEED.get(model, 80.0) + in_tokens / 25000.0)
            call.input_tokens = in_tokens
            call.output_tokens = out_tokens
            call.output = output if output is not None else f"[{agent.agent_id} output for: {prompt[:60]}]"
        return call.output

    def _search(self, ctx: RunContext, world: Dict[str, Any], query: str) -> List[str]:
        self._advance(ctx, 0.4 + 0.15 * len(query))
        return list(world["sources"].get(query, []))

    @staticmethod
    def _quality(agent) -> float:
        return MODEL_QUALITY.get(agent.model or "", 0.8)

    # -- main flow ------------------------------------------------------------
    def run(self, task: Task, ctx: RunContext) -> Any:
        world = WORLD[task.task_id]
        researchers = ["researcher_a", "researcher_b", "researcher_c"]
        scopes = {"researcher_a": ["s1", "s2"], "researcher_b": ["s3", "s4"], "researcher_c": ["s1", "s2"]}

        # 1. coordinator plans and delegates -------------------------------
        with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
            self._llm(ctx, coord, f"Plan the research for: {task.input}", 300,
                      output="Plan: split sources between researchers, verify with critic, synthesize.")
            for r in researchers:
                coord.send(r, f"Research {world['name']}. Cover sources {', '.join(scopes[r])} and report facts.",
                           kind="task_assignment", scope=scopes[r])

        # 2. researchers work in parallel ---------------------------------------
        def research(r: str) -> None:
            rng = ctx.rng  # per-branch generator: draws do not depend on thread scheduling
            with ctx.agent(r, role="research", model="claude-sonnet-5") as agent:
                msg = agent.receive()[0]
                scope = msg.metadata.get("scope", [])
                q = self._quality(agent)
                self._llm(ctx, agent, f"Assignment: {msg.content}. Decide search queries.", 120)
                found: List[str] = []
                for s in scope:
                    hits = agent.call_tool("web_search", partial(self._search, ctx, world), query=s)
                    for fact in hits:
                        p_keep = WRONG_FACT_PICKUP if fact == world["wrong"] else q
                        if rng.random() < p_keep:
                            found.append(fact)
                notes = "\n".join(found) if found else "No reliable information found."
                self._llm(ctx, agent, f"Summarise findings:\n{notes}", 350, output=notes)
                art = agent.produce(f"notes_{r[-1]}", notes, derived_from=[])  # from search results only
                agent.send("coordinator", notes, kind="result", artifacts=[art])

        ctx.parallel(*(partial(research, r) for r in researchers))

        # 3. coordinator relays notes to the critic ------------------------
        with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
            notes_msgs = {m.sender: m for m in coord.receive()}
            chosen = [notes_msgs[a] for a in ("researcher_a", "researcher_b") if a in notes_msgs]
            arts = [a for m in chosen for a in m.artifacts]
            combined = "\n".join(str(m.content) for m in chosen)
            self._llm(ctx, coord, f"Forward these findings for review:\n{combined}", 150,
                      extra_context_tokens=1800, input_artifacts=arts, output="Forwarding findings to the critic.")
            coord.send("critic", combined, kind="request", artifacts=arts)

        # 4. critic verifies ---------------------------------------------------
        with ctx.agent("critic", role="critic", model="claude-opus-5") as critic:
            msgs = critic.receive()
            arts = [a for m in msgs for a in m.artifacts]
            for a in arts:
                critic.consume(a, purpose="review")
            facts = [ln for m in msgs for ln in str(m.content).splitlines() if ln.strip()]
            q = self._quality(critic)
            flagged = [f for f in facts if f == world["wrong"] and ctx.rng.random() < q]
            critique = ("Flagged as unsupported: " + "; ".join(flagged)) if flagged else "No issues found."
            self._llm(ctx, critic, "Review these findings for errors:\n" + "\n".join(facts), 400,
                      input_artifacts=arts, output=critique)
            critique_art = critic.produce("critique", critique, derived_from=arts)
            critic.send("coordinator", critique, kind="feedback", artifacts=arts + [critique_art])

        # 5. coordinator relays to synthesizer --------------------------------
        with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
            msgs = coord.receive()
            arts = [a for m in msgs for a in m.artifacts]
            content = "\n".join(str(m.content) for m in msgs)
            self._llm(ctx, coord, f"Hand off to synthesizer:\n{content}", 120, extra_context_tokens=2600,
                      input_artifacts=arts, output="Handing off to synthesizer.")
            coord.send("synthesizer", content, kind="request", artifacts=arts)

        # 6. synthesizer writes the brief -------------------------------------
        with ctx.agent("synthesizer", role="synthesizer", model="claude-opus-5") as synth:
            msgs = synth.receive()
            arts = [a for m in msgs for a in m.artifacts]
            gathered: List[str] = []
            for a in arts:
                art = synth.consume(a, purpose="synthesis")
                if art is not None and art.name.startswith("notes"):
                    gathered.extend(ln for ln in str(art.content).splitlines() if ln.strip())
            flagged_set = set(flagged)
            kept = [f for f in gathered if f not in flagged_set and f != "No reliable information found."]
            answer = f"Brief on {world['name']}:\n" + "\n".join(f"- {f}" for f in dict.fromkeys(kept))
            self._llm(ctx, synth, "Write the brief from:\n" + "\n".join(kept), 700, extra_context_tokens=9000,
                      input_artifacts=arts, output=answer)
            final_art = synth.produce("final_brief", answer, derived_from=arts)
        ctx.set_output(answer, artifacts=[final_art])
        return answer


def make_swarm() -> ResearchSwarm:
    """Factory used by ``swarmeval.evaluate`` and the CLI."""
    return ResearchSwarm()


def sim_clock() -> SimClock:
    return SimClock()


if __name__ == "__main__":  # pragma: no cover
    from swarmeval import evaluate
    res = evaluate(make_swarm, build_benchmark(), trials=3, clock_factory=sim_clock)
    print(res.report())
