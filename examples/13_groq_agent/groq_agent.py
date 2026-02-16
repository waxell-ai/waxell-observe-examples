#!/usr/bin/env python3
"""Groq auto-instrumentor & OpenAI function calling demo.

Demonstrates the Groq auto-instrumentor with Llama 3 models and OpenAI
function calling with the tools parameter and auto-captured tool_calls.

Usage::

    # Dry-run (no API key needed)
    python examples/13_groq_agent/groq_agent.py --dry-run

    # With real Groq + OpenAI calls
    GROQ_API_KEY=gsk_... OPENAI_API_KEY=sk-... python examples/13_groq_agent/groq_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import os

# CRITICAL: init() BEFORE importing openai/groq so auto-instrumentor can patch them
from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_groq_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "What are the latest advances in AI safety research and their practical applications?"


# ---------------------------------------------------------------------------
# OpenAI function definitions for tool calling demo
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "average"],
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["operation", "values"],
            },
        },
    },
]


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool function and return the result as a string."""
    args = json.loads(arguments)
    if name == "web_search":
        return json.dumps({
            "results": [
                {"title": "AI Safety Research Overview 2025", "url": "https://example.com/safety", "snippet": "Recent advances include RLHF improvements, constitutional AI methods, and automated red-teaming."},
                {"title": "Practical AI Governance", "url": "https://example.com/governance", "snippet": "Organizations adopting policy-based governance for agent deployment."},
            ],
            "total": 2,
        })
    elif name == "calculator":
        values = args.get("values", [])
        op = args.get("operation", "add")
        if op == "average":
            result = sum(values) / len(values) if values else 0
        elif op == "add":
            result = sum(values)
        elif op == "multiply":
            result = 1
            for v in values:
                result *= v
        else:
            result = values[0] if values else 0
        return json.dumps({"result": result})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def run_groq_demo(
    query: str,
    groq_client: object,
    openai_client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
) -> None:
    """Execute the Groq + function calling showcase."""

    observe_client = get_observe_client()

    # ==================================================================
    # PHASE 1: GROQ API CALLS
    # ==================================================================
    print("=" * 70)
    print("PHASE 1: GROQ AUTO-INSTRUMENTOR")
    print("  Demonstrates the Groq instrumentor with Llama 3 models.")
    print("  When using real Groq SDK, calls are auto-captured by the")
    print("  waxell-observe Groq instrumentor.")
    print("=" * 70)
    print()

    async with WaxellContext(
        agent_name="groq-demo",
        workflow_name="groq-analysis",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        ctx.set_tag("demo", "groq")
        ctx.set_tag("provider", "groq")

        # Call 1: Classify query using Llama 3.3 70B
        print("[Groq] Step 1: Classifying query via llama-3.3-70b-versatile...")
        classify_response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the query into: technical, general, opinion, or research. "
                        "Also identify the primary topic area."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        classification = classify_response.choices[0].message.content
        ctx.record_llm_call(
            model=classify_response.model,
            tokens_in=classify_response.usage.prompt_tokens,
            tokens_out=classify_response.usage.completion_tokens,
            task="classify_query",
            prompt_preview=query[:200],
            response_preview=classification[:200],
        )
        ctx.record_step("groq_classify", output={"classification": classification[:200]})
        print(f"  Model: {classify_response.model}")
        print(f"  Classification: {classification[:120]}...")
        print()

        # Call 2: Synthesize answer using Llama 3.3 70B
        print("[Groq] Step 2: Synthesizing answer via llama-3.3-70b-versatile...")
        synthesis_response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable AI safety researcher. Provide a "
                        "thorough, well-structured answer with specific examples."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        synthesis = synthesis_response.choices[0].message.content
        ctx.record_llm_call(
            model=synthesis_response.model,
            tokens_in=synthesis_response.usage.prompt_tokens,
            tokens_out=synthesis_response.usage.completion_tokens,
            task="synthesize_answer",
            prompt_preview=query[:200],
            response_preview=synthesis[:200],
        )
        ctx.record_step("groq_synthesize", output={"answer_length": len(synthesis)})
        ctx.set_result({"classification": classification[:200], "answer": synthesis[:200]})
        print(f"  Model: {synthesis_response.model}")
        print(f"  Answer: {synthesis[:120]}...")

    print()

    # ==================================================================
    # PHASE 2: OPENAI FUNCTION CALLING WITH TOOLS PARAM
    # ==================================================================
    print("=" * 70)
    print("PHASE 2: OPENAI FUNCTION CALLING")
    print("  Demonstrates OpenAI's tools parameter with auto-captured")
    print("  tool_calls from the response. Shows the full loop:")
    print("  call with tools → tool_calls response → execute tool →")
    print("  send tool result → final text response.")
    print("=" * 70)
    print()

    async with WaxellContext(
        agent_name="groq-demo",
        workflow_name="function-calling",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        ctx.set_tag("demo", "function-calling")
        ctx.set_tag("provider", "openai")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant with access to web search and "
                    "calculator tools. Use them to answer the user's question."
                ),
            },
            {"role": "user", "content": query},
        ]

        # Call 1: Send with tools — expect tool_calls in response
        print("[Function Calling] Step 1: Sending query with tool definitions...")
        print(f"  Tools available: {', '.join(t['function']['name'] for t in TOOL_DEFINITIONS)}")
        tool_response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        assistant_msg = tool_response.choices[0].message
        ctx.record_llm_call(
            model=tool_response.model,
            tokens_in=tool_response.usage.prompt_tokens,
            tokens_out=tool_response.usage.completion_tokens,
            task="tool_selection",
        )
        ctx.record_step("tool_request", output={
            "finish_reason": tool_response.choices[0].finish_reason,
            "has_tool_calls": assistant_msg.tool_calls is not None,
        })

        if assistant_msg.tool_calls:
            print(f"  Finish reason: {tool_response.choices[0].finish_reason}")
            print(f"  Tool calls requested: {len(assistant_msg.tool_calls)}")

            # Execute each tool call
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            })

            for tc in assistant_msg.tool_calls:
                print(f"\n[Function Calling] Step 2: Executing tool '{tc.function.name}'...")
                print(f"  Arguments: {tc.function.arguments}")

                tool_result = execute_tool(tc.function.name, tc.function.arguments)
                print(f"  Result: {tool_result[:120]}...")

                # Record tool call in waxell-observe
                ctx.record_tool_call(
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                    output=json.loads(tool_result),
                    status="ok",
                    duration_ms=45,
                    tool_type="function",
                )

                # Add tool result to messages for the follow-up call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            # Call 2: Send tool results back for final answer
            print(f"\n[Function Calling] Step 3: Sending tool results for final answer...")
            final_response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            final_answer = final_response.choices[0].message.content
            ctx.record_llm_call(
                model=final_response.model,
                tokens_in=final_response.usage.prompt_tokens,
                tokens_out=final_response.usage.completion_tokens,
                task="final_synthesis",
            )
            ctx.record_step("final_answer", output={"answer_length": len(final_answer or "")})
            ctx.set_result({"answer": (final_answer or "")[:200], "tools_used": len(assistant_msg.tool_calls)})
            print(f"  Final answer: {(final_answer or 'No content')[:120]}...")
        else:
            # No tool calls — direct response
            print("  No tool calls (direct response)")
            ctx.record_step("direct_answer", output={"answer_length": len(assistant_msg.content or "")})
            ctx.set_result({"answer": (assistant_msg.content or "")[:200]})
            print(f"  Answer: {(assistant_msg.content or 'No content')[:120]}...")

    print()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 70)
    print("GROQ & FUNCTION CALLING DEMO SUMMARY")
    print("=" * 70)
    print("  Phase 1: Groq Llama 3.3 70B (2 calls: classify + synthesize)")
    print("  Phase 2: OpenAI function calling (tools → execute → final)")
    print()
    print("  SDK features exercised:")
    print("    Groq auto-instrumentor      — auto-captures Groq SDK calls")
    print("    OpenAI tools parameter       — function calling with tool_calls")
    print("    record_tool_call()           — manual tool execution recording")
    print("    Multi-provider in one session — Groq + OpenAI in same trace")
    print()
    print("  With real API keys, the auto-instrumentors capture:")
    print("    Groq: model, tokens, latency via groq_instrumentor.py")
    print("    OpenAI: model, tokens, tool_calls via openai_instrumentor.py")
    print()
    print("[Groq Demo] Complete. (4 LLM calls, 4 steps, 1+ tool calls)")


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    groq_client = get_groq_client(dry_run=args.dry_run)
    openai_client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Groq Agent] Starting Groq & function calling showcase...")
    print(f"[Groq Agent] Session: {session}")
    print(f"[Groq Agent] End user: {user_id} ({user_group})")
    print(f"[Groq Agent] Query: {query[:80]}...")
    print()

    await run_groq_demo(
        query=query,
        groq_client=groq_client,
        openai_client=openai_client,
        observe_active=observe_active,
        session=session,
        user_id=user_id,
        user_group=user_group,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
