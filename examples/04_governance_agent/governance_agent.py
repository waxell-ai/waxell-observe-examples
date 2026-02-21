#!/usr/bin/env python3
"""Governance & policy deep dive demo.

Demonstrates all governance-related SDK features: record_events(),
record_policy_check(), manual check_policy(), RunCompleteResult retry
feedback loop, and sync client wrappers.

Uses the modern decorator-based SDK pattern (@waxell.observe, @waxell.tool,
@waxell.decision, @waxell.step_dec) while preserving governance-specific
APIs that require direct waxell_ctx access (record_policy_check,
check_policy) and sync wrapper demos (start_run_sync, etc.).

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

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

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

DEFAULT_QUERY = "Evaluate the risk profile of deploying an autonomous AI trading agent"


# ---------------------------------------------------------------------------
# @tool decorators -- auto-record tool calls
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="function")
def risk_analysis(domain: str, risk_level: str) -> dict:
    """Simulate a risk analysis tool call."""
    return {"risk_score": 0.78, "recommendation": "requires_review"}


@waxell.tool(tool_type="api")
def compliance_check(agent: str, domain: str) -> dict:
    """Simulate a compliance check tool call."""
    return {"compliant": True, "frameworks": ["SOX", "MiFID II"]}


# ---------------------------------------------------------------------------
# @decision decorator -- auto-record routing decisions
# ---------------------------------------------------------------------------


@waxell.decision(
    name="execution_route",
    options=["continue", "escalate_to_human", "abort"],
)
def make_execution_decision(risk_score: float) -> dict:
    """Decide execution route based on risk score."""
    return {
        "chosen": "continue",
        "reasoning": "Policy check returned allow, risk score within acceptable range",
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# @step decorators -- auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="pre_execution_evaluation")
def record_pre_execution_step(policies_checked: int) -> dict:
    """Record the pre-execution evaluation step."""
    return {"policies_checked": policies_checked}


@waxell.step_dec(name="mid_execution_complete")
def record_mid_execution_step(steps: int, policy_checks: int) -> dict:
    """Record the mid-execution completion step."""
    return {"steps": steps, "policy_checks": policy_checks}


# ==================================================================
# PHASE 1: PRE-EXECUTION GOVERNANCE SPANS
# ==================================================================


@waxell.observe(
    agent_name="governance-evaluator",
    workflow_name="policy-evaluation",
    capture_io=True,
)
async def run_pre_execution(query: str, client, *, waxell_ctx=None) -> dict:
    """Phase 1: Pre-execution governance spans.

    Records policy evaluation results as governance spans on the trace.
    Uses waxell_ctx.record_policy_check() to capture each policy decision.
    LLM call is auto-captured by the instrumentor.
    """
    print("=" * 70)
    print("PHASE 1: PRE-EXECUTION GOVERNANCE SPANS")
    print("  Records policy evaluation results as governance spans on the trace.")
    print("  Uses waxell_ctx.record_policy_check() to capture each policy decision.")
    print("=" * 70)
    print()

    # Simulate pre-execution policy evaluations
    policies = [
        ("Budget Policy", "allow", "budget", "Within daily token budget (used 12K of 100K)"),
        ("Rate Limit Policy", "allow", "rate-limit", "3 of 100 hourly requests used"),
        ("Scheduling Policy", "allow", "scheduling", "Within allowed hours (Mon-Fri 6am-10pm)"),
        ("Safety Policy", "warn", "safety", "High-risk domain detected: financial trading"),
        ("Kill Switch", "allow", "kill", "No active kill switches"),
    ]

    # GOVERNANCE-SPECIFIC: record_policy_check via waxell_ctx
    for policy_name, action, category, reason in policies:
        waxell_ctx.record_policy_check(
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

    # LLM call -- auto-captured by instrumentor (no manual record_llm_call needed)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a risk assessment analyst for AI systems."},
            {"role": "user", "content": query},
        ],
    )
    answer = response.choices[0].message.content

    # @step_dec records the step automatically
    record_pre_execution_step(len(policies))
    print(f"  LLM response: {answer[:120]}...")

    print()

    return {"answer": answer, "policies_checked": len(policies)}


# ==================================================================
# PHASE 2: MID-EXECUTION MANUAL POLICY CHECK
# ==================================================================


@waxell.observe(
    agent_name="governance-mid-execution",
    workflow_name="mid-execution-check",
    capture_io=True,
)
async def run_mid_execution(query: str, *, waxell_ctx=None) -> dict:
    """Phase 2: Mid-execution manual policy check.

    Uses waxell_ctx.check_policy() for on-demand policy evaluation between
    steps. Also shows @waxell.tool and @waxell.decision decorators.
    """
    print("=" * 70)
    print("PHASE 2: MID-EXECUTION MANUAL POLICY CHECK")
    print("  Uses waxell_ctx.check_policy() for on-demand policy evaluation between")
    print("  steps. Also shows @waxell.tool and @waxell.decision decorators.")
    print("=" * 70)
    print()

    # Step 1: @tool decorator auto-records the tool call
    print("[Phase 2] Step 1: Executing risk_analysis tool...")
    result = risk_analysis("financial_trading", "high")
    print(f"  Tool output: risk_score={result['risk_score']}, recommendation={result['recommendation']}")
    print("  (mid_execution_governance: auto-flushed data and checked governance)")
    print()

    # Step 2: GOVERNANCE-SPECIFIC: manual mid-execution policy check via waxell_ctx
    print("[Phase 2] Step 2: Manual policy check between steps...")
    try:
        policy_result = await waxell_ctx.check_policy()
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

    # Step 3: @decision decorator auto-records the routing decision
    print("[Phase 2] Step 3: Making routing decision...")
    decision_result = make_execution_decision(result["risk_score"])
    print(f"  Decision: {decision_result['chosen']} (confidence: {decision_result['confidence']})")
    print()

    # Step 4: @tool decorator auto-records the tool call
    print("[Phase 2] Step 4: Running compliance check tool...")
    compliance_result = compliance_check("governance-demo", "trading")
    print(f"  Compliance: passed ({', '.join(compliance_result['frameworks'])})")

    # @step_dec records the step automatically
    record_mid_execution_step(steps=4, policy_checks=1)

    return {"phase": "mid-execution", "risk_score": 0.78, "compliant": True}


# ==================================================================
# PHASE 3: RETRY FEEDBACK LOOP (SYNC WRAPPERS)
# ==================================================================


async def run_sync_wrapper_demo(
    query: str,
    client,
    observe_client,
    session: str,
    user_id: str,
    user_group: str,
) -> None:
    """Phase 3: Retry feedback loop using sync wrappers.

    Uses WaxellObserveClient directly with sync methods:
    start_run_sync -> complete_run_sync
    Inspects RunCompleteResult for should_retry and retry_feedback.

    This demonstrates the low-level sync API surface -- kept as-is to
    showcase this alternative API pattern.
    """
    print("=" * 70)
    print("PHASE 3: RETRY FEEDBACK LOOP (SYNC WRAPPERS)")
    print("  Uses WaxellObserveClient directly with sync methods:")
    print("  start_run_sync -> complete_run_sync")
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

        # Make LLM call -- auto-captured by instrumentor
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide a brief risk assessment."},
                {"role": "user", "content": retry_prompt},
            ],
        )
        answer = response.choices[0].message.content
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
            print(f"  -> Retrying with feedback: {complete_result.retry_feedback[:100]}")
            retry_prompt = f"{query}\n\nPrevious feedback: {complete_result.retry_feedback}"
        else:
            print(f"  -> No retry needed (action={complete_result.governance_action})")
            break
        print()

    print()


# ==================================================================
# PHASE 4: GOVERNANCE EVENT RECORDING
# ==================================================================


def run_event_recording(observe_client) -> None:
    """Phase 4: Governance event recording.

    Uses client.record_events_sync() to record governance events that
    appear in the controlplane's event timeline.
    """
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
# ORCHESTRATOR
# ==================================================================


@waxell.observe(
    agent_name="governance-orchestrator",
    workflow_name="governance-deep-dive",
    capture_io=True,
)
async def run_governance_demo(
    query: str,
    *,
    dry_run: bool = False,
    waxell_ctx=None,
) -> dict:
    """Execute the governance showcase across 4 phases."""
    waxell.tag("demo", "governance")

    client = get_openai_client(dry_run=dry_run)
    observe_client = get_observe_client()

    # Phase 1: Pre-execution governance spans
    phase1 = await run_pre_execution(query, client, waxell_ctx=waxell_ctx)

    # Phase 2: Mid-execution manual policy check
    phase2 = await run_mid_execution(query, waxell_ctx=waxell_ctx)

    # Phase 3: Sync wrapper retry feedback loop (kept as low-level API demo)
    # Uses observe_client directly -- these sync wrappers are the POINT of this phase
    session_id = waxell_ctx.session_id if waxell_ctx else ""
    user_id = waxell_ctx.user_id if waxell_ctx else ""
    user_group = waxell_ctx.user_group if waxell_ctx else ""
    await run_sync_wrapper_demo(
        query=query,
        client=client,
        observe_client=observe_client,
        session=session_id,
        user_id=user_id,
        user_group=user_group,
    )

    # Phase 4: Governance event recording (kept as-is)
    run_event_recording(observe_client)

    # Summary
    print("=" * 70)
    print("GOVERNANCE DEMO SUMMARY")
    print("=" * 70)
    print("  Phase 1: Pre-execution governance spans (5 policy evaluations)")
    print("  Phase 2: Mid-execution manual check_policy() + @tool + @decision")
    print("  Phase 3: Retry feedback loop with sync wrappers")
    print("  Phase 4: Governance event recording (3 events)")
    print()
    print("  SDK features exercised:")
    print("    @waxell.observe                — agent/workflow context")
    print("    @waxell.tool                   — auto-recorded tool calls")
    print("    @waxell.decision               — auto-recorded routing decisions")
    print("    @waxell.step_dec               — auto-recorded execution steps")
    print("    waxell_ctx.record_policy_check — governance span recording")
    print("    waxell_ctx.check_policy()      — manual mid-execution policy check")
    print("    client.start_run_sync()        — sync wrapper for start_run")
    print("    client.complete_run_sync()     — sync wrapper for complete_run")
    print("    client.record_events_sync()    — sync wrapper for record_events")
    print("    RunCompleteResult.should_retry  — retry feedback inspection")
    print("    RunCompleteResult.retry_feedback — governance retry guidance")
    print()
    print("[Governance Demo] Complete. (3 LLM calls, 8 steps, 5 policy checks, 3 events)")

    return {
        "phase1": phase1,
        "phase2": phase2,
        "phases_completed": 4,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

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
        dry_run=args.dry_run,
        # Context overrides -- intercepted by @waxell.observe decorator
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    )

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
