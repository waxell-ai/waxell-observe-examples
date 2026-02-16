#!/usr/bin/env python3
"""Multi-agent coordinator demo.

Demonstrates waxell-observe's @waxell_agent decorator with a multi-agent
system. A coordinator dispatches tasks to three specialized sub-agents
(planner, researcher, executor), each creating its own WaxellContext with
a shared session_id for full trace correlation.

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
import os

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import openai (auto-instrumentor has patched it)
import waxell_observe
from waxell_observe import WaxellContext, waxell_agent, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_TASK = "What are the key considerations for deploying AI agents in production?"

_client = None


async def main() -> None:
    args = parse_args()
    global _client
    _client = get_openai_client(dry_run=args.dry_run)
    task = args.query or DEFAULT_TASK
    policy_triggers = args.policy_triggers

    session = generate_session_id()

    # When observe is disabled, skip policy checks and mid-execution governance
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

    # ------------------------------------------------------------------
    # Sub-agent functions defined INSIDE main() so @waxell_agent captures
    # the correct session_id from the enclosing scope.
    # ------------------------------------------------------------------

    @waxell_agent(
        agent_name="demo-planner",
        workflow_name="plan-task",
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=obs_client,
    )
    async def plan_task(task_description: str, waxell_ctx=None) -> dict:
        """Break a task into research queries."""
        print("  [Planner] Analyzing task...")

        waxell_ctx.record_step("analyze_task", output={"task": task_description[:100]})

        response = await _client.chat.completions.create(
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

        waxell_ctx.record_step(
            "generate_plan",
            output={"num_queries": len(queries), "strategy": "parallel-research"},
        )
        waxell_ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="plan_task",
            prompt_preview=task_description[:200],
            response_preview=content[:200],
        )

        print(f"  [Planner] Generated {len(queries)} research queries")
        return {"queries": queries, "strategy": "parallel-research"}

    @waxell_agent(
        agent_name="demo-researcher",
        workflow_name="research-query",
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=obs_client,
    )
    async def research_query(
        query: str, query_index: int = 0, waxell_ctx=None
    ) -> str:
        """Research a single query and return findings."""
        waxell_ctx.set_tag("query_index", str(query_index))

        print(f'  [Researcher] Researching: "{query[:60]}..."')

        waxell_ctx.record_step("search", output={"query": query[:100]})

        response = await _client.chat.completions.create(
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

        waxell_ctx.record_step("compile_findings", output={"length": len(finding)})
        waxell_ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="research_query",
            prompt_preview=query[:200],
            response_preview=finding[:200],
        )

        print("  [Researcher] Finding compiled")
        return finding

    @waxell_agent(
        agent_name="demo-executor",
        workflow_name="synthesize-findings",
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=obs_client,
    )
    async def synthesize_findings(
        findings: list[str], original_task: str, waxell_ctx=None
    ) -> str:
        """Synthesize research findings into a final answer."""
        waxell_ctx.set_metadata("num_findings", len(findings))

        print(f"  [Executor] Synthesizing {len(findings)} findings...")

        waxell_ctx.record_step(
            "evaluate_findings",
            output={"num_findings": len(findings)},
        )

        findings_text = "\n".join(
            f"Finding {i + 1}: {f}" for i, f in enumerate(findings)
        )
        response = await _client.chat.completions.create(
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

        waxell_ctx.record_step("produce_output", output={"answer_length": len(answer)})
        waxell_ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="synthesize_findings",
            prompt_preview=original_task[:200],
            response_preview=answer[:200],
        )

        print("  [Executor] Output produced")
        return answer

    # ------------------------------------------------------------------
    # Coordinator: orchestrate the sub-agents via WaxellContext
    # ------------------------------------------------------------------

    async with WaxellContext(
        agent_name="demo-coordinator",
        workflow_name="multi-agent-task",
        inputs={"task": task},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=obs_client,
    ) as ctx:
        ctx.set_tag("demo", "multi-agent")
        ctx.set_tag("num_agents", "3")

        try:
            # ----------------------------------------------------------
            # Policy trigger: extra pre-check steps (safety category)
            # ----------------------------------------------------------
            if policy_triggers:
                print("[Coordinator] [Policy Trigger] Recording extra pre-check steps...")
                ctx.record_step("pre_check_1", output={"check": "input_validation"})
                print("           Recorded pre_check_1 (step 1/4 pre-checks)")
                ctx.record_step("pre_check_2", output={"check": "resource_availability"})
                print("           Recorded pre_check_2 (step 2/4 pre-checks)")
                ctx.record_step("security_scan", output={"check": "threat_assessment"})
                print("           Recorded security_scan (step 3/4 pre-checks)")
                ctx.record_step("approval_gate", output={"check": "authorization"})
                print("           Recorded approval_gate (step 4/4 pre-checks, exceeds max_steps=3)")
                print()

            # ----------------------------------------------------------
            # Phase 1: Planning
            # ----------------------------------------------------------
            print("[Coordinator] Phase 1: Planning...")
            ctx.record_step("delegate_to_planner")

            print("[Coordinator] Delegating to Planner...")
            plan_result = await plan_task(task)
            queries = plan_result["queries"]

            for i, q in enumerate(queries, 1):
                print(f"  Query {i}: {q[:80]}")
            print()

            # ----------------------------------------------------------
            # Phase 2: Research
            # ----------------------------------------------------------
            print("[Coordinator] Phase 2: Research...")
            ctx.record_step("delegate_to_researchers")

            findings: list[str] = []
            for i, query in enumerate(queries):
                print(f"[Coordinator] Dispatching Researcher for query {i + 1}/{len(queries)}...")
                finding = await research_query(query, query_index=i)
                findings.append(finding)
            print()

            # ----------------------------------------------------------
            # Phase 3: Synthesis
            # ----------------------------------------------------------
            print("[Coordinator] Phase 3: Synthesis...")
            ctx.record_step("delegate_to_executor")

            print("[Coordinator] Delegating to Executor...")
            final_answer = await synthesize_findings(findings, task)
            print()

            # ----------------------------------------------------------
            # Completion
            # ----------------------------------------------------------
            ctx.set_result(
                {
                    "answer": final_answer,
                    "num_queries": len(queries),
                    "num_findings": len(findings),
                }
            )

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

        except PolicyViolationError as e:
            print(f"\n[Multi-Agent Demo] POLICY VIOLATION: {e}")
            print("[Multi-Agent Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
