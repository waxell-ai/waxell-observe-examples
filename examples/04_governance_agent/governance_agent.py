#!/usr/bin/env python3
"""Governance & policy deep dive demo.

Demonstrates all governance-related SDK features: record_events(),
record_policy_check(), manual check_policy(), RunCompleteResult retry
feedback loop, and sync client wrappers.

Usage::

    # Dry-run (no API key needed)
    python examples/04_governance_agent/governance_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/04_governance_agent/governance_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os
import time

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Evaluate the risk profile of deploying an autonomous AI trading agent"


async def run_governance_demo(
    query: str,
    client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
) -> None:
    """Execute the governance showcase across 4 phases."""

    observe_client = get_observe_client()

    # ==================================================================
    # PHASE 1: PRE-EXECUTION GOVERNANCE SPANS
    # ==================================================================
    print("=" * 70)
    print("PHASE 1: PRE-EXECUTION GOVERNANCE SPANS")
    print("  Records policy evaluation results as governance spans on the trace.")
    print("  Uses ctx.record_policy_check() to capture each policy decision.")
    print("=" * 70)
    print()

    async with WaxellContext(
        agent_name="governance-demo",
        workflow_name="policy-evaluation",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        # Simulate pre-execution policy evaluations
        policies = [
            ("Budget Policy", "allow", "budget", "Within daily token budget (used 12K of 100K)"),
            ("Rate Limit Policy", "allow", "rate-limit", "3 of 100 hourly requests used"),
            ("Scheduling Policy", "allow", "scheduling", "Within allowed hours (Mon-Fri 6am-10pm)"),
            ("Safety Policy", "warn", "safety", "High-risk domain detected: financial trading"),
            ("Kill Switch", "allow", "kill", "No active kill switches"),
        ]

        for policy_name, action, category, reason in policies:
            ctx.record_policy_check(
                policy_name=policy_name,
                action=action,
                category=category,
                reason=reason if action != "allow" else "",
                duration_ms=round(1.5 + (hash(policy_name) % 10) * 0.3, 1),
                phase="pre_execution",
                priority=policies.index((policy_name, action, category, reason)) * 10 + 10,
            )
            symbol = "PASS" if action == "allow" else "WARN" if action == "warn" else "BLOCK"
            print(f"  [{symbol}] {policy_name} ({category}): {reason}")

        print()
        print("[Phase 1] Making LLM call after pre-execution checks...")

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a risk assessment analyst for AI systems."},
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content
        ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="risk_assessment",
        )
        ctx.record_step("pre_execution_evaluation", output={"policies_checked": len(policies)})
        print(f"  LLM response: {answer[:120]}...")

    print()

    # ==================================================================
    # PHASE 2: MID-EXECUTION MANUAL POLICY CHECK
    # ==================================================================
    print("=" * 70)
    print("PHASE 2: MID-EXECUTION MANUAL POLICY CHECK")
    print("  Uses ctx.check_policy() for on-demand policy evaluation between")
    print("  steps. Also shows mid_execution_governance=True auto-flush.")
    print("=" * 70)
    print()

    async with WaxellContext(
        agent_name="governance-demo",
        workflow_name="mid-execution-check",
        inputs={"query": query, "phase": "mid-execution"},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=observe_client,
    ) as ctx:
        # Step 1: Tool call (triggers auto governance flush with mid_execution_governance)
        print("[Phase 2] Step 1: Executing risk_analysis tool...")
        ctx.record_tool_call(
            name="risk_analysis",
            input={"domain": "financial_trading", "risk_level": "high"},
            output={"risk_score": 0.78, "recommendation": "requires_review"},
            status="ok",
            duration_ms=150,
            tool_type="function",
        )
        print("  Tool output: risk_score=0.78, recommendation=requires_review")
        print("  (mid_execution_governance: auto-flushed data and checked governance)")
        print()

        # Manual mid-execution policy check
        print("[Phase 2] Step 2: Manual policy check between steps...")
        try:
            policy_result = await ctx.check_policy()
            print(f"  PolicyCheckResult:")
            print(f"    action: {policy_result.action}")
            print(f"    allowed: {policy_result.allowed}")
            print(f"    blocked: {policy_result.blocked}")
            print(f"    should_retry: {policy_result.should_retry}")
            if policy_result.reason:
                print(f"    reason: {policy_result.reason}")
        except Exception as e:
            print(f"  Policy check error (expected if no policies configured): {e}")
        print()

        # Step 3: Decision based on policy result
        print("[Phase 2] Step 3: Making routing decision...")
        ctx.record_decision(
            name="execution_route",
            options=["continue", "escalate_to_human", "abort"],
            chosen="continue",
            reasoning="Policy check returned allow, risk score within acceptable range",
            confidence=0.85,
        )
        print("  Decision: continue (confidence: 0.85)")
        print()

        # Step 4: Another tool call
        print("[Phase 2] Step 4: Running compliance check tool...")
        ctx.record_tool_call(
            name="compliance_check",
            input={"agent": "governance-demo", "domain": "trading"},
            output={"compliant": True, "frameworks": ["SOX", "MiFID II"]},
            status="ok",
            duration_ms=95,
            tool_type="api",
        )
        print("  Compliance: passed (SOX, MiFID II)")

        ctx.record_step("mid_execution_complete", output={"steps": 4, "policy_checks": 1})
        ctx.set_result({"phase": "mid-execution", "risk_score": 0.78, "compliant": True})

    print()

    # ==================================================================
    # PHASE 3: RETRY FEEDBACK LOOP (SYNC WRAPPERS)
    # ==================================================================
    print("=" * 70)
    print("PHASE 3: RETRY FEEDBACK LOOP (SYNC WRAPPERS)")
    print("  Uses WaxellObserveClient directly with sync methods:")
    print("  start_run_sync → record_llm_calls_sync → complete_run_sync")
    print("  Inspects RunCompleteResult for should_retry and retry_feedback.")
    print("=" * 70)
    print()

    max_attempts = 3
    attempt = 0
    retry_prompt = query

    while attempt < max_attempts:
        attempt += 1
        print(f"[Phase 3] Attempt {attempt}/{max_attempts}")

        # Start run using sync wrapper
        run_info = observe_client.start_run_sync(
            agent_name="governance-retry-demo",
            workflow_name="retry-feedback",
            inputs={"query": retry_prompt[:200], "attempt": attempt},
            metadata={"max_retries": max_attempts},
            user_id=user_id,
            user_group=user_group,
            session_id=session,
        )
        print(f"  Run ID: {run_info.run_id or '(empty - server may be down)'}")

        # Make LLM call
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide a brief risk assessment."},
                {"role": "user", "content": retry_prompt},
            ],
        )
        answer = response.choices[0].message.content

        # Record LLM call using sync wrapper
        observe_client.record_llm_calls_sync(
            run_id=run_info.run_id,
            calls=[{
                "model": response.model,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "task": f"attempt_{attempt}",
            }],
        )
        print(f"  LLM response: {answer[:100]}...")

        # Complete run and check governance response
        complete_result = observe_client.complete_run_sync(
            run_id=run_info.run_id,
            result={"answer": answer[:200], "attempt": attempt},
            status="success",
        )

        print(f"  RunCompleteResult:")
        print(f"    governance_action: {complete_result.governance_action}")
        print(f"    should_retry: {complete_result.should_retry}")
        print(f"    retry_feedback: \"{complete_result.retry_feedback or '(none)'}\"")
        print(f"    max_retries: {complete_result.max_retries}")

        if complete_result.should_retry and complete_result.retry_feedback:
            print(f"  → Retrying with feedback: {complete_result.retry_feedback[:100]}")
            retry_prompt = f"{query}\n\nPrevious feedback: {complete_result.retry_feedback}"
        else:
            print(f"  → No retry needed (action={complete_result.governance_action})")
            break
        print()

    print()

    # ==================================================================
    # PHASE 4: GOVERNANCE EVENT RECORDING
    # ==================================================================
    print("=" * 70)
    print("PHASE 4: GOVERNANCE EVENT RECORDING")
    print("  Uses client.record_events() to record governance events that")
    print("  appear in the controlplane's event timeline.")
    print("=" * 70)
    print()

    events = [
        {
            "event": "policy_warn",
            "agent_name": "governance-demo",
            "metadata": {
                "policy_name": "Safety Policy",
                "category": "safety",
                "reason": "High-risk domain: financial trading",
                "action": "warn",
            },
        },
        {
            "event": "budget_threshold",
            "agent_name": "governance-demo",
            "metadata": {
                "policy_name": "Budget Policy",
                "category": "budget",
                "tokens_used": 12450,
                "daily_limit": 100000,
                "percent_used": 12.45,
            },
        },
        {
            "event": "compliance_check_passed",
            "agent_name": "governance-demo",
            "metadata": {
                "frameworks": ["SOX", "MiFID II"],
                "domain": "financial_trading",
                "risk_score": 0.78,
            },
        },
    ]

    print(f"  Recording {len(events)} governance events...")
    for evt in events:
        print(f"    - {evt['event']}: {evt['metadata'].get('reason', evt['metadata'].get('policy_name', ''))}")

    # Use sync wrapper for event recording
    observe_client.record_events_sync(events=events)
    print("  Events recorded successfully")
    print()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 70)
    print("GOVERNANCE DEMO SUMMARY")
    print("=" * 70)
    print("  Phase 1: Pre-execution governance spans (5 policy evaluations)")
    print("  Phase 2: Mid-execution manual check_policy() + auto-flush")
    print("  Phase 3: Retry feedback loop with sync wrappers")
    print("  Phase 4: Governance event recording (3 events)")
    print()
    print("  SDK features exercised:")
    print("    ctx.record_policy_check()      — governance span recording")
    print("    ctx.check_policy()             — manual mid-execution policy check")
    print("    client.start_run_sync()        — sync wrapper for start_run")
    print("    client.record_llm_calls_sync() — sync wrapper for record_llm_calls")
    print("    client.complete_run_sync()     — sync wrapper for complete_run")
    print("    client.record_events_sync()    — sync wrapper for record_events")
    print("    RunCompleteResult.should_retry  — retry feedback inspection")
    print("    RunCompleteResult.retry_feedback — governance retry guidance")
    print()
    print("[Governance Demo] Complete. (3 LLM calls, 8 steps, 5 policy checks, 3 events)")


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Governance Agent] Starting governance & policy deep dive...")
    print(f"[Governance Agent] Session: {session}")
    print(f"[Governance Agent] End user: {user_id} ({user_group})")
    print(f"[Governance Agent] Query: {query[:80]}...")
    print()

    await run_governance_demo(
        query=query,
        client=client,
        observe_active=observe_active,
        session=session,
        user_id=user_id,
        user_group=user_group,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
