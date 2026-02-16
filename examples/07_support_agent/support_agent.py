#!/usr/bin/env python3
"""Customer support agent demo with agentic behavior tracking.

Demonstrates waxell-observe's behavior tracking methods through a customer
support pipeline that classifies intent, looks up orders, checks inventory,
makes routing decisions, handles retries/fallbacks, and generates responses.

Usage::

    # Dry-run (no OpenAI API key needed)
    python examples/07_support_agent/support_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/07_support_agent/support_agent.py

    # Custom query
    python examples/07_support_agent/support_agent.py --dry-run --query "Where is my order ORD-12345?"
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
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    MockFailingOpenAI,
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Where is my order ORD-12345? I placed it last week and haven't received any updates."


async def run_support_agent(
    query: str,
    client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
    dry_run: bool = True,
) -> dict:
    """Execute the customer support agent pipeline.

    Args:
        query: The customer support query.
        client: OpenAI-compatible async client (real or mock).
        observe_active: Whether observe/governance is active.
        session: Session ID for trace correlation.
        user_id: End-user identifier.
        user_group: End-user group/tier.
        dry_run: Whether to use mock clients.

    Returns:
        Dict with support interaction results.
    """
    async with WaxellContext(
        agent_name="support-agent",
        workflow_name="customer-support",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "support")
        ctx.set_tag("pipeline", "customer-support")
        ctx.set_metadata("channel", "chat")

        try:
            # ------------------------------------------------------------------
            # Step 1: Classify Intent (LLM + reasoning)
            # ------------------------------------------------------------------
            print("[Support] Step 1: Classifying customer intent...")

            classify_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer support intent classifier. "
                            "Classify the query into: order_status, refund_request, "
                            "product_inquiry, complaint, general_help."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            classification = classify_response.choices[0].message.content

            ctx.record_step("classify_intent", output={"intent": "order_status"})
            ctx.record_llm_call(
                model=classify_response.model,
                tokens_in=classify_response.usage.prompt_tokens,
                tokens_out=classify_response.usage.completion_tokens,
                task="classify_intent",
                prompt_preview=query[:200],
                response_preview=classification[:200],
            )

            # Reasoning about the classification
            ctx.record_reasoning(
                step="analyze_intent",
                thought=(
                    "Customer mentions 'order ORD-12345' and asks 'where is my order', "
                    "strongly indicating an order status inquiry. No mention of refund, "
                    "return, or complaint keywords. The phrase 'haven't received any updates' "
                    "suggests frustration but the primary intent is tracking information."
                ),
                evidence=["keyword:order", "keyword:where", "phrase:haven't received updates"],
                conclusion="Intent classified as order_status with high confidence",
            )
            print(f"           Intent: order_status (via reasoning chain)")

            # ------------------------------------------------------------------
            # Step 2: Order Lookup (tool call)
            # ------------------------------------------------------------------
            print("[Support] Step 2: Looking up order...")

            order_data = {
                "order_id": "ORD-12345",
                "status": "shipped",
                "tracking_number": "1Z999AA10123456784",
                "carrier": "UPS",
                "estimated_delivery": "2024-03-15",
                "items": [
                    {"name": "Wireless Headphones", "quantity": 1, "price": 79.99},
                    {"name": "USB-C Cable", "quantity": 2, "price": 12.99},
                ],
            }

            ctx.record_tool_call(
                name="order_lookup",
                input={"order_id": "ORD-12345"},
                output=order_data,
                duration_ms=120,
                status="ok",
                tool_type="database",
            )
            print(f"           Order found: {order_data['status']} via {order_data['carrier']}")

            # ------------------------------------------------------------------
            # Step 3: Inventory Check (tool call)
            # ------------------------------------------------------------------
            print("[Support] Step 3: Checking inventory for related items...")

            inventory_data = {
                "product_id": "PROD-789",
                "product_name": "Wireless Headphones",
                "in_stock": True,
                "quantity_available": 42,
                "warehouse": "US-WEST-1",
                "restock_date": None,
            }

            ctx.record_tool_call(
                name="inventory_check",
                input={"product_id": "PROD-789"},
                output=inventory_data,
                duration_ms=85,
                status="ok",
                tool_type="database",
            )
            print(f"           Inventory: {inventory_data['quantity_available']} units in stock")

            # ------------------------------------------------------------------
            # Step 4: Decision Tree (3 levels)
            # ------------------------------------------------------------------
            print("[Support] Step 4: Processing decision tree...")

            # Decision 1: Order found?
            ctx.record_decision(
                name="order_found",
                options=["process_order", "order_not_found"],
                chosen="process_order",
                reasoning="Order ORD-12345 found in database with valid status",
                confidence=1.0,
            )
            print("           [1/3] order_found -> process_order (confidence: 1.0)")

            # Decision 2: What does the customer want?
            ctx.record_decision(
                name="refund_requested",
                options=["process_refund", "provide_status", "escalate"],
                chosen="provide_status",
                reasoning="Customer asking about shipping status, not requesting refund or escalation",
                confidence=0.85,
            )
            print("           [2/3] refund_requested -> provide_status (confidence: 0.85)")

            # Decision 3: Can we auto-respond?
            ctx.record_decision(
                name="response_type",
                options=["automated_response", "human_handoff"],
                chosen="automated_response",
                reasoning="Order status query with clear tracking data available; no ambiguity requiring human judgment",
                confidence=0.91,
            )
            print("           [3/3] response_type -> automated_response (confidence: 0.91)")

            # ------------------------------------------------------------------
            # Step 5: Generate Response with Retry
            # ------------------------------------------------------------------
            print("[Support] Step 5: Generating response (with retry demo)...")

            # Simulate rate limit on first attempt
            ctx.record_retry(
                attempt=1,
                reason="Rate limit exceeded",
                strategy="retry",
                original_error="429 Too Many Requests",
                fallback_to="gpt-4o-mini",
                max_attempts=3,
            )
            print("           Attempt 1: 429 Too Many Requests")

            # Simulate fallback to smaller model
            ctx.record_retry(
                attempt=2,
                reason="Fallback to smaller model after rate limit",
                strategy="fallback",
                fallback_to="gpt-4o-mini",
                max_attempts=3,
            )
            print("           Attempt 2: Falling back to gpt-4o-mini")

            # Use a failing client for the first attempt, then succeed
            failing_client = MockFailingOpenAI(
                fail_count=1,
                error_message="429 Too Many Requests",
            )

            # First attempt fails
            try:
                await failing_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                )
            except Exception:
                pass  # Expected failure, already recorded retry above

            # Second attempt succeeds with fallback model
            response_prompt = (
                f"Customer query: {query}\n\n"
                f"Order details:\n"
                f"- Order ID: {order_data['order_id']}\n"
                f"- Status: {order_data['status']}\n"
                f"- Carrier: {order_data['carrier']}\n"
                f"- Tracking: {order_data['tracking_number']}\n"
                f"- Estimated delivery: {order_data['estimated_delivery']}\n\n"
                "Generate a friendly, helpful customer support response."
            )

            gen_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly customer support agent. Provide "
                            "helpful, empathetic responses with specific order details."
                        ),
                    },
                    {"role": "user", "content": response_prompt},
                ],
            )
            support_reply = gen_response.choices[0].message.content

            ctx.record_step("generate_response", output={"response_length": len(support_reply)})
            ctx.record_llm_call(
                model="gpt-4o-mini",
                tokens_in=gen_response.usage.prompt_tokens,
                tokens_out=gen_response.usage.completion_tokens,
                task="generate_response",
                prompt_preview=response_prompt[:200],
                response_preview=support_reply[:200],
            )
            print(f"           Response generated ({len(support_reply)} chars) via gpt-4o-mini")

            # ------------------------------------------------------------------
            # Step 6: Confidence Routing
            # ------------------------------------------------------------------
            print("[Support] Step 6: Confidence routing...")

            ctx.record_decision(
                name="confidence_routing",
                options=["send_response", "add_disclaimer", "escalate_to_human"],
                chosen="send_response",
                reasoning=(
                    "All decision points resolved with confidence >= 0.85. "
                    "Order data is unambiguous. Response generated successfully "
                    "after fallback. No escalation triggers detected."
                ),
                confidence=0.92,
            )
            print("           Routing: send_response (confidence: 0.92)")

            # ------------------------------------------------------------------
            # Set final result
            # ------------------------------------------------------------------
            result = {
                "response": support_reply,
                "intent": "order_status",
                "order_id": "ORD-12345",
                "resolution": "automated_response",
                "model_used": "gpt-4o-mini",
                "retries": 2,
            }
            ctx.set_result(result)

            print()
            print(f"[Support] Response: {support_reply[:200]}...")
            print(
                f"[Support] Complete. "
                f"(2 LLM calls, 2 tool calls, 1 reasoning step, "
                f"4 decisions, 2 retries)"
            )
            return result

        except PolicyViolationError as e:
            print(f"\n[Support] POLICY VIOLATION: {e}")
            print("[Support] Agent halted by governance policy.")
            return {"error": str(e)}


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Support Agent] Starting customer support pipeline...")
    print(f"[Support Agent] Session: {session}")
    print(f"[Support Agent] End user: {user_id} ({user_group})")
    print(f"[Support Agent] Query: {query[:80]}...")
    print()

    await run_support_agent(
        query=query,
        client=client,
        observe_active=observe_active,
        session=session,
        user_id=user_id,
        user_group=user_group,
        dry_run=args.dry_run,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
