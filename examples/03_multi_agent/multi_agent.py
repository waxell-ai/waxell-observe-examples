#!/usr/bin/env python3
"""Multi-agent coordinator demo.

Demonstrates waxell-observe's modern decorator pattern with a multi-agent
system. A coordinator dispatches tasks to three specialized sub-agents
(planner, researcher, executor), each decorated with @waxell.observe for
automatic trace correlation via parent-child context lineage.

Architecture::

  demo-coordinator (parent)
  +-- @step: delegate_to_planner
  +-- demo-planner (child)
  |   +-- @step: analyze_task
  |   +-- @decision: choose_strategy
  |   +-- @step: generate_plan
  +-- @step: delegate_to_researchers
  +-- demo-researcher (child, x3)
  |   +-- @step: search
  |   +-- @step: compile_findings
  +-- @step: delegate_to_executor
  +-- demo-executor (child)
      +-- @step: evaluate_findings
      +-- @step: produce_output

Usage::

    # Dry-run (no OpenAI API key needed)
    python examples/03_multi_agent/multi_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/03_multi_agent/multi_agent.py

    # Custom task
    python examples/03_multi_agent/multi_agent.py --dry-run --query "How should I deploy AI agents?"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import provider clients (auto-instrumentors have patched them)
import waxell_observe as waxell
from waxell_observe import generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_TASK = "What are the key considerations for deploying AI agents in production?"


# ---------------------------------------------------------------------------
# @step decorators -- auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="analyze_task")
def analyze_task(task: str) -> dict:
    """Record the task being analyzed."""
    return {"task": task[:100]}


@waxell.step_dec(name="generate_plan")
def generate_plan(queries: list) -> dict:
    """Record the generated plan."""
    return {"num_queries": len(queries), "strategy": "parallel-research"}


@waxell.step_dec(name="search")
def search(query: str) -> dict:
    """Record a research search step."""
    return {"query": query[:100]}


@waxell.step_dec(name="compile_findings")
def compile_findings(finding_length: int) -> dict:
    """Record compilation of a finding."""
    return {"length": finding_length}


@waxell.step_dec(name="evaluate_findings")
def evaluate_findings(num_findings: int) -> dict:
    """Record evaluation of collected findings."""
    return {"num_findings": num_findings}


@waxell.step_dec(name="produce_output")
def produce_output(answer_length: int) -> dict:
    """Record final output production."""
    return {"answer_length": answer_length}


@waxell.step_dec(name="delegate_to_planner")
def delegate_to_planner() -> dict:
    """Record delegation to the planner sub-agent."""
    return {}


@waxell.step_dec(name="delegate_to_researchers")
def delegate_to_researchers() -> dict:
    """Record delegation to the researcher sub-agents."""
    return {}


@waxell.step_dec(name="delegate_to_executor")
def delegate_to_executor() -> dict:
    """Record delegation to the executor sub-agent."""
    return {}


# ---------------------------------------------------------------------------
# @decision decorator -- choose research strategy
# ---------------------------------------------------------------------------


@waxell.decision(name="choose_strategy", options=["parallel-research", "sequential-research", "focused-deep-dive"])
async def choose_strategy(queries: list) -> dict:
    """Decide the research strategy based on the planned queries."""
    return {
        "chosen": "parallel-research",
        "reasoning": f"With {len(queries)} independent queries, parallel execution is optimal",
    }


# ---------------------------------------------------------------------------
# Child agent 1: Planner -- breaks task into research queries
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="demo-planner", workflow_name="plan-task", capture_io=True)
async def plan_task(task_description: str, *, client=None, waxell_ctx=None) -> dict:
    """Break a task into research queries."""
    print("  [Planner] Analyzing task...")

    analyze_task(task_description)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. Break the following task "
                    "into exactly 3 specific research queries, one per line."
                ),
            },
            {"role": "user", "content": f"Break down this task into a plan: {task_description}"},
        ],
    )
    content = response.choices[0].message.content

    # Parse the 3 research queries from the response
    lines = [
        line.strip().lstrip("0123456789.-) ")
        for line in content.strip().splitlines()
        if line.strip()
    ]
    queries = lines[:3] if len(lines) >= 3 else [content] * 3

    await choose_strategy(queries)
    generate_plan(queries)

    print(f"  [Planner] Generated {len(queries)} research queries")
    return {"queries": queries, "strategy": "parallel-research"}


# ---------------------------------------------------------------------------
# Child agent 2: Researcher -- researches a single query
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="demo-researcher", workflow_name="research-query", capture_io=True)
async def research_query(
    query: str, *, query_index: int = 0, client=None, waxell_ctx=None
) -> str:
    """Research a single query and return findings."""
    waxell.tag("query_index", str(query_index))

    print(f'  [Researcher] Researching: "{query[:60]}..."')

    search(query)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Provide a concise 2-3 "
                    "sentence finding for this research query."
                ),
            },
            {"role": "user", "content": f"Investigate the following topic: {query}"},
        ],
    )
    finding = response.choices[0].message.content

    compile_findings(len(finding))

    print("  [Researcher] Finding compiled")
    return finding


# ---------------------------------------------------------------------------
# Child agent 3: Executor -- synthesizes findings into a final answer
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="demo-executor", workflow_name="synthesize-findings", capture_io=True)
async def synthesize_findings(
    findings: list[str], original_task: str, *, client=None, waxell_ctx=None
) -> str:
    """Synthesize research findings into a final answer."""
    waxell.metadata("num_findings", len(findings))

    print(f"  [Executor] Synthesizing {len(findings)} findings...")

    evaluate_findings(len(findings))

    findings_text = "\n".join(
        f"Finding {i + 1}: {f}" for i, f in enumerate(findings)
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert synthesizer. Combine these research "
                    "findings into a clear, comprehensive answer to the "
                    "original task."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original task: {original_task}\n\n"
                    f"Collected findings:\n{findings_text}\n\n"
                    "Synthesize a comprehensive answer from these findings."
                ),
            },
        ],
    )
    answer = response.choices[0].message.content

    produce_output(len(answer))

    print("  [Executor] Output produced")
    return answer


# ---------------------------------------------------------------------------
# Parent coordinator -- orchestrates the sub-agents
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="demo-coordinator", workflow_name="multi-agent-task", capture_io=True)
async def run_coordinator(
    task: str, *, dry_run: bool = False, policy_triggers: bool = False, waxell_ctx=None
) -> dict:
    """Coordinate the multi-agent task across planner, researchers, and executor."""
    waxell.tag("demo", "multi-agent")
    waxell.tag("num_agents", "3")

    client = get_openai_client(dry_run=dry_run)

    try:
        # ----------------------------------------------------------
        # Policy trigger: extra pre-check steps (safety category)
        # ----------------------------------------------------------
        if policy_triggers:
            print("[Coordinator] [Policy Trigger] Recording extra pre-check steps...")

            @waxell.step_dec(name="pre_check_1")
            def _pre_check_1():
                return {"check": "input_validation"}

            @waxell.step_dec(name="pre_check_2")
            def _pre_check_2():
                return {"check": "resource_availability"}

            @waxell.step_dec(name="security_scan")
            def _security_scan():
                return {"check": "threat_assessment"}

            @waxell.step_dec(name="approval_gate")
            def _approval_gate():
                return {"check": "authorization"}

            _pre_check_1()
            print("           Recorded pre_check_1 (step 1/4 pre-checks)")
            _pre_check_2()
            print("           Recorded pre_check_2 (step 2/4 pre-checks)")
            _security_scan()
            print("           Recorded security_scan (step 3/4 pre-checks)")
            _approval_gate()
            print("           Recorded approval_gate (step 4/4 pre-checks, exceeds max_steps=3)")
            print()

        # ----------------------------------------------------------
        # Phase 1: Planning
        # ----------------------------------------------------------
        print("[Coordinator] Phase 1: Planning...")
        delegate_to_planner()

        print("[Coordinator] Delegating to Planner...")
        plan_result = await plan_task(task, client=client)
        queries = plan_result["queries"]

        for i, q in enumerate(queries, 1):
            print(f"  Query {i}: {q[:80]}")
        print()

        # ----------------------------------------------------------
        # Phase 2: Research
        # ----------------------------------------------------------
        print("[Coordinator] Phase 2: Research...")
        delegate_to_researchers()

        findings: list[str] = []
        for i, query in enumerate(queries):
            print(f"[Coordinator] Dispatching Researcher for query {i + 1}/{len(queries)}...")
            finding = await research_query(query, query_index=i, client=client)
            findings.append(finding)
        print()

        # ----------------------------------------------------------
        # Phase 3: Synthesis
        # ----------------------------------------------------------
        print("[Coordinator] Phase 3: Synthesis...")
        delegate_to_executor()

        print("[Coordinator] Delegating to Executor...")
        final_answer = await synthesize_findings(findings, task, client=client)
        print()

        # ----------------------------------------------------------
        # Completion
        # ----------------------------------------------------------
        answer_display = (
            final_answer[:300] + "..."
            if len(final_answer) > 300
            else final_answer
        )
        print(f"[Multi-Agent Demo] Final Answer: {answer_display}")
        print(
            f"[Multi-Agent Demo] Complete. "
            f"(3 sub-agents, 5 LLM calls, {len(queries)} research queries)"
        )

        return {
            "answer": final_answer,
            "num_queries": len(queries),
            "num_findings": len(findings),
        }

    except PolicyViolationError as e:
        print(f"\n[Multi-Agent Demo] POLICY VIOLATION: {e}")
        print("[Multi-Agent Demo] Agent halted by governance policy.")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    task = args.query or DEFAULT_TASK
    policy_triggers = args.policy_triggers

    session = generate_session_id()

    observe_active = not is_observe_disabled()
    obs_client = get_observe_client()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    if policy_triggers:
        print("[Multi-Agent Demo] POLICY TRIGGER MODE -- intentionally crossing policy thresholds")
        print("[Multi-Agent Demo] Expected triggers: safety (step limit exceeded from multiple agents)")
        print()

    print("[Multi-Agent Demo] Starting multi-agent task coordination...")
    print(f"[Multi-Agent Demo] Session: {session}")
    print(f"[Multi-Agent Demo] End user: {user_id} ({user_group})")
    print(f'[Multi-Agent Demo] Task: "{task}"')
    print()

    result = await run_coordinator(
        task,
        dry_run=args.dry_run,
        policy_triggers=policy_triggers,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=obs_client,
    )

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
