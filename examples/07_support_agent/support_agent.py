#!/usr/bin/env python3
"""Customer support agent demo with modern decorator patterns.

Demonstrates waxell-observe's decorator-first SDK through a customer support
pipeline: classify intent, look up orders, check inventory, make routing
decisions, handle retries/fallbacks, and generate responses.

Multi-agent architecture:
  support-orchestrator (parent)
  ├── support-handler (child) — intent classification, order lookup, decisions
  └── support-evaluator (child) — response generation, confidence routing

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

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import openai (auto-instrumentor has patched it)
import waxell_observe as waxell
from waxell_observe import generate_session_id
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

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

ORDER_DATA = {
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

INVENTORY_DATA = {
    "product_id": "PROD-789",
    "product_name": "Wireless Headphones",
    "in_stock": True,
    "quantity_available": 42,
    "warehouse": "US-WEST-1",
    "restock_date": None,
}


# ---------------------------------------------------------------------------
# @step decorator — auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="classify_intent")
async def classify_intent_step(classification: str) -> dict:
    """Record intent classification step."""
    return {"intent": "order_status", "raw_classification": classification[:100]}


# ---------------------------------------------------------------------------
# @tool decorators — auto-record tool calls
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="database")
def order_lookup(order_id: str) -> dict:
    """Look up an order by ID."""
    return ORDER_DATA


@waxell.tool(tool_type="database")
def inventory_check(product_id: str) -> dict:
    """Check inventory for a product."""
    return INVENTORY_DATA


# ---------------------------------------------------------------------------
# @decision decorator — auto-record routing decisions
# ---------------------------------------------------------------------------


@waxell.decision(name="order_found", options=["process_order", "order_not_found"])
async def decide_order_found(order_data: dict) -> dict:
    """Decide whether to process a found order."""
    return {
        "chosen": "process_order",
        "reasoning": f"Order {order_data['order_id']} found in database with valid status",
    }


@waxell.decision(name="refund_requested", options=["process_refund", "provide_status", "escalate"])
async def decide_refund_or_status(classification: str) -> dict:
    """Decide what the customer wants based on classification."""
    return {
        "chosen": "provide_status",
        "reasoning": "Customer asking about shipping status, not requesting refund or escalation",
    }


@waxell.decision(name="response_type", options=["automated_response", "human_handoff"])
async def decide_response_type() -> dict:
    """Decide if we can auto-respond or need human handoff."""
    return {
        "chosen": "automated_response",
        "reasoning": "Order status query with clear tracking data available; no ambiguity requiring human judgment",
    }


@waxell.decision(name="confidence_routing", options=["send_response", "add_disclaimer", "escalate_to_human"])
async def decide_confidence_routing() -> dict:
    """Final confidence routing decision."""
    return {
        "chosen": "send_response",
        "reasoning": (
            "All decision points resolved with confidence >= 0.85. "
            "Order data is unambiguous. Response generated successfully "
            "after fallback. No escalation triggers detected."
        ),
    }


# ---------------------------------------------------------------------------
# @reasoning decorator — auto-record chain-of-thought
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="analyze_intent")
async def analyze_intent(query: str) -> dict:
    """Analyze the customer intent via reasoning chain."""
    return {
        "thought": (
            "Customer mentions 'order ORD-12345' and asks 'where is my order', "
            "strongly indicating an order status inquiry. No mention of refund, "
            "return, or complaint keywords. The phrase 'haven't received any updates' "
            "suggests frustration but the primary intent is tracking information."
        ),
        "evidence": ["keyword:order", "keyword:where", "phrase:haven't received updates"],
        "conclusion": "Intent classified as order_status with high confidence",
    }


# ---------------------------------------------------------------------------
# @retry decorator — auto-record retry attempts
# ---------------------------------------------------------------------------


@waxell.retry_dec(max_attempts=3, strategy="fallback")
async def generate_response_with_retry(client, prompt: str):
    """Generate response with automatic retry recording on failure."""
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly customer support agent. Provide "
                    "helpful, empathetic responses with specific order details."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )


# ---------------------------------------------------------------------------
# Child Agent 1: support-handler — classification + lookup + decisions
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="support-handler", workflow_name="support-classification")
async def run_support_handler(query: str, openai_client, waxell_ctx=None) -> dict:
    """Handle intent classification, order lookup, and routing decisions."""
    waxell.tag("agent_role", "handler")
    waxell.tag("provider", "openai")

    # Step 1: Classify Intent (LLM + reasoning)
    print("[Support Handler] Step 1: Classifying customer intent...")
    classify_response = await openai_client.chat.completions.create(
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

    await classify_intent_step(classification)
    reasoning = await analyze_intent(query)
    print(f"           Intent: order_status (via reasoning chain)")

    # Step 2: Order Lookup (tool call)
    print("[Support Handler] Step 2: Looking up order...")
    order_data = order_lookup(order_id="ORD-12345")
    print(f"           Order found: {order_data['status']} via {order_data['carrier']}")

    # Step 3: Inventory Check (tool call)
    print("[Support Handler] Step 3: Checking inventory for related items...")
    inv_data = inventory_check(product_id="PROD-789")
    print(f"           Inventory: {inv_data['quantity_available']} units in stock")

    # Step 4: Decision Tree (3 levels)
    print("[Support Handler] Step 4: Processing decision tree...")
    d1 = await decide_order_found(order_data)
    print(f"           [1/3] order_found -> {d1['chosen']}")

    d2 = await decide_refund_or_status(classification)
    print(f"           [2/3] refund_requested -> {d2['chosen']}")

    d3 = await decide_response_type()
    print(f"           [3/3] response_type -> {d3['chosen']}")

    return {
        "classification": classification,
        "order_data": order_data,
        "inventory_data": inv_data,
        "decisions": [d1, d2, d3],
    }


# ---------------------------------------------------------------------------
# Child Agent 2: support-evaluator — response generation + confidence
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="support-evaluator", workflow_name="support-response")
async def run_support_evaluator(query: str, order_data: dict, openai_client, waxell_ctx=None) -> dict:
    """Generate the support response with retry and confidence routing."""
    waxell.tag("agent_role", "evaluator")
    waxell.tag("provider", "openai")

    # Step 5: Generate Response with Retry
    print("[Support Evaluator] Step 5: Generating response (with retry demo)...")

    # Simulate rate limit on first attempt using MockFailingOpenAI
    failing_client = MockFailingOpenAI(
        fail_count=1,
        error_message="429 Too Many Requests",
    )
    try:
        await failing_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )
    except Exception:
        pass  # Expected failure for retry demo
    print("           Attempt 1: 429 Too Many Requests")
    print("           Attempt 2: Falling back to gpt-4o-mini")

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

    gen_response = await generate_response_with_retry(openai_client, response_prompt)
    support_reply = gen_response.choices[0].message.content
    print(f"           Response generated ({len(support_reply)} chars) via gpt-4o-mini")

    # Step 6: Confidence Routing
    print("[Support Evaluator] Step 6: Confidence routing...")
    d4 = await decide_confidence_routing()
    print(f"           Routing: {d4['chosen']}")

    # Scores
    waxell.score("classification_confidence", 0.88)
    waxell.score("response_quality", "good", data_type="categorical", comment="Auto-generated response")
    waxell.score("sla_met", True, data_type="boolean")

    return {
        "response": support_reply,
        "model_used": "gpt-4o-mini",
        "retries": 2,
        "routing": d4["chosen"],
    }


# ---------------------------------------------------------------------------
# Parent Orchestrator: support-orchestrator
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="support-orchestrator", workflow_name="customer-support")
async def run_support_pipeline(query: str, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Coordinate the full customer support pipeline across child agents."""
    waxell.tag("demo", "support")
    waxell.tag("pipeline", "customer-support")
    waxell.metadata("channel", "chat")
    waxell.metadata("mode", "dry-run" if dry_run else "live")

    openai_client = get_openai_client(dry_run=dry_run)

    # Phase 1: support-handler child agent
    print("[Support Orchestrator] Phase 1: Handling request (support-handler)...")
    handler_result = await run_support_handler(
        query=query,
        openai_client=openai_client,
    )
    print()

    # Phase 2: support-evaluator child agent
    print("[Support Orchestrator] Phase 2: Evaluating response (support-evaluator)...")
    eval_result = await run_support_evaluator(
        query=query,
        order_data=handler_result["order_data"],
        openai_client=openai_client,
    )

    result = {
        "response": eval_result["response"],
        "intent": "order_status",
        "order_id": "ORD-12345",
        "resolution": "automated_response",
        "model_used": eval_result["model_used"],
        "retries": eval_result["retries"],
        "pipeline": "support-orchestrator -> support-handler -> support-evaluator",
    }

    print()
    print(f"[Support Orchestrator] Response: {eval_result['response'][:200]}...")
    print(
        f"[Support Orchestrator] Complete. "
        f"(2 LLM calls, 2 tool calls, 1 reasoning step, "
        f"4 decisions, 2 retries)"
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    session = generate_session_id()
    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]
    observe_active = not is_observe_disabled()

    print("[Support Agent] Starting customer support pipeline...")
    print(f"[Support Agent] Session: {session}")
    print(f"[Support Agent] End user: {user_id} ({user_group})")
    print(f"[Support Agent] Query: {query[:80]}...")
    print()

    try:
        await run_support_pipeline(
            query=query,
            dry_run=args.dry_run,
            session_id=session,
            user_id=user_id,
            user_group=user_group,
            enforce_policy=observe_active,
            mid_execution_governance=observe_active,
            client=get_observe_client(),
        )
    except PolicyViolationError as e:
        print(f"\n[Support Agent] POLICY VIOLATION: {e}")
        print("[Support Agent] Agent halted by governance policy.")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
