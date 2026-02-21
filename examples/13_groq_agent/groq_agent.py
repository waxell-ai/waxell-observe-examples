#!/usr/bin/env python3
"""Groq + OpenAI function calling demo.

Demonstrates waxell-observe SDK decorators with Groq's Llama models and
OpenAI function calling in a multi-agent pipeline:

  groq-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: choose_provider_strategy
  ├── groq-analyzer (child agent)
  │   ├── Auto-instrumented Groq LLM calls (classify + synthesize)
  │   └── @reasoning: assess_answer_quality
  ├── function-caller (child agent)
  │   ├── @tool: web_search, calculator
  │   ├── Auto-instrumented OpenAI LLM calls (tool selection + final)
  │   └── @decision: choose_tool_response_style
  └── scores, tags, metadata

Usage::

    # Dry-run (no API keys needed)
    python examples/13_groq_agent/groq_agent.py --dry-run

    # With real Groq + OpenAI calls
    python examples/13_groq_agent/groq_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

# CRITICAL: init() BEFORE importing openai/groq so auto-instrumentors can patch them
from _common import setup_observe

setup_observe()

import waxell_observe as waxell
from waxell_observe import generate_session_id

from _common import (
    get_groq_client,
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "What are the latest advances in AI safety research and their practical applications?"


# ---------------------------------------------------------------------------
# Tool definitions for OpenAI function calling
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
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Maximum results"},
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
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide", "average"]},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["operation", "values"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Decorated helper functions
# ---------------------------------------------------------------------------

@waxell.step_dec(name="preprocess_query")
def preprocess_query(query: str) -> dict:
    """Normalize and prepare the query."""
    cleaned = query.strip()
    word_count = len(cleaned.split())
    has_question = "?" in cleaned
    return {"cleaned_query": cleaned, "word_count": word_count, "has_question": has_question}


@waxell.decision(name="choose_provider_strategy", options=["groq_only", "groq_then_openai", "openai_only"])
def choose_provider_strategy(query_info: dict) -> dict:
    """Decide which providers to use based on query characteristics."""
    if query_info.get("word_count", 0) > 20:
        chosen = "groq_then_openai"
        reasoning = "Complex query — use Groq for fast analysis, OpenAI for tool-augmented follow-up"
    else:
        chosen = "groq_then_openai"
        reasoning = "Standard query — demonstrate both providers in pipeline"
    return {"chosen": chosen, "reasoning": reasoning, "confidence": 0.9}


@waxell.tool(tool_type="function", name="web_search")
def web_search(query: str, max_results: int = 2) -> dict:
    """Search the web for information on a topic."""
    return {
        "results": [
            {"title": "AI Safety Research Overview 2025", "url": "https://example.com/safety",
             "snippet": "Recent advances include RLHF improvements, constitutional AI methods, and automated red-teaming."},
            {"title": "Practical AI Governance", "url": "https://example.com/governance",
             "snippet": "Organizations adopting policy-based governance for agent deployment."},
        ],
        "total": 2,
    }


@waxell.tool(tool_type="function", name="calculator")
def calculator(operation: str, values: list) -> dict:
    """Perform mathematical calculations."""
    if operation == "average":
        result = sum(values) / len(values) if values else 0
    elif operation == "add":
        result = sum(values)
    elif operation == "multiply":
        r = 1
        for v in values:
            r *= v
        result = r
    elif operation == "subtract":
        result = values[0] - sum(values[1:]) if values else 0
    elif operation == "divide":
        result = values[0] / values[1] if len(values) >= 2 and values[1] != 0 else 0
    else:
        result = values[0] if values else 0
    return {"result": result}


@waxell.reasoning_dec(step="assess_answer_quality")
def assess_answer_quality(answer: str) -> dict:
    """Assess the quality of the Groq-generated answer."""
    word_count = len(answer.split())
    has_examples = any(w in answer.lower() for w in ["example", "for instance", "such as"])
    has_structure = any(w in answer for w in ["1.", "2.", "-", "First", "Second"])
    quality = 0.6
    if has_examples:
        quality += 0.15
    if has_structure:
        quality += 0.15
    if word_count > 50:
        quality += 0.1
    return {
        "word_count": word_count,
        "has_examples": has_examples,
        "has_structure": has_structure,
        "quality_score": round(min(quality, 1.0), 2),
        "reasoning": f"Answer has {word_count} words, {'includes' if has_examples else 'lacks'} examples, {'structured' if has_structure else 'unstructured'}",
    }


@waxell.decision(name="choose_tool_response_style", options=["concise", "detailed", "bullet_points"])
def choose_tool_response_style(tool_results: list) -> dict:
    """Decide how to format the final response based on tool results."""
    total_content = sum(len(json.dumps(r)) for r in tool_results)
    if total_content > 500:
        chosen = "bullet_points"
        reasoning = "Large tool output — bullet points for readability"
    elif total_content > 200:
        chosen = "detailed"
        reasoning = "Moderate tool output — detailed synthesis"
    else:
        chosen = "concise"
        reasoning = "Small tool output — concise response"
    return {"chosen": chosen, "reasoning": reasoning, "confidence": 0.85}


# ---------------------------------------------------------------------------
# Child agents
# ---------------------------------------------------------------------------

@waxell.observe(agent_name="groq-analyzer", workflow_name="groq-analysis", capture_io=True)
async def run_groq_analyzer(query: str, groq_client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Analyze query using Groq's Llama models."""
    waxell.tag("provider", "groq")
    waxell.tag("task", "analysis")

    # Call 1: Classify
    print("[Groq Analyzer] Classifying query via llama-3.3-70b-versatile...")
    classify_response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Classify the query into: technical, general, opinion, or research. Also identify the primary topic area."},
            {"role": "user", "content": query},
        ],
    )
    classification = classify_response.choices[0].message.content
    print(f"  Classification: {classification[:120]}...")

    # Call 2: Synthesize
    print("[Groq Analyzer] Synthesizing answer via llama-3.3-70b-versatile...")
    synthesis_response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a knowledgeable AI safety researcher. Provide a thorough, well-structured answer with specific examples."},
            {"role": "user", "content": query},
        ],
    )
    answer = synthesis_response.choices[0].message.content

    # Assess quality
    quality = assess_answer_quality(answer)

    waxell.score("groq_answer_quality", quality["quality_score"], comment="Groq synthesis quality")
    print(f"  Answer quality: {quality['quality_score']}")

    return {"classification": classification, "answer": answer, "quality": quality}


@waxell.observe(agent_name="function-caller", workflow_name="function-calling", capture_io=True)
async def run_function_caller(query: str, openai_client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Execute OpenAI function calling with tool execution loop."""
    waxell.tag("provider", "openai")
    waxell.tag("task", "function_calling")

    messages = [
        {"role": "system", "content": "You are a research assistant with access to web search and calculator tools. Use them to answer the user's question."},
        {"role": "user", "content": query},
    ]

    # Call 1: Send with tools
    print("[Function Caller] Sending query with tool definitions...")
    tool_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_DEFINITIONS,
    )

    assistant_msg = tool_response.choices[0].message
    tool_results = []

    if assistant_msg.tool_calls:
        print(f"  Tool calls requested: {len(assistant_msg.tool_calls)}")

        messages.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in assistant_msg.tool_calls
            ],
        })

        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  Executing tool '{tc.function.name}' with {args}")

            # Execute via @waxell.tool decorated functions
            if tc.function.name == "web_search":
                result = web_search(**args)
            elif tc.function.name == "calculator":
                result = calculator(**args)
            else:
                result = {"error": f"Unknown tool: {tc.function.name}"}

            tool_results.append(result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

        # Decide response style
        style = choose_tool_response_style(tool_results)
        print(f"  Response style: {style['chosen']}")

        # Call 2: Final synthesis with tool results
        print("[Function Caller] Synthesizing final answer with tool results...")
        final_response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        final_answer = final_response.choices[0].message.content or ""
    else:
        final_answer = assistant_msg.content or ""
        print("  No tool calls — direct response")

    waxell.score("function_calling_quality", 0.85, comment="Tool-augmented response quality")

    return {"answer": final_answer, "tools_used": len(tool_results), "tool_results": tool_results}


# ---------------------------------------------------------------------------
# Parent orchestrator
# ---------------------------------------------------------------------------

@waxell.observe(agent_name="groq-orchestrator", workflow_name="groq-function-calling", capture_io=True)
async def run_agent(query: str, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Orchestrate Groq analysis + OpenAI function calling pipeline."""
    waxell.tag("demo", "groq")
    waxell.tag("multi_provider", "groq+openai")
    waxell.metadata("providers", ["groq", "openai"])
    waxell.metadata("models", ["llama-3.3-70b-versatile", "gpt-4o-mini"])

    groq_client = get_groq_client(dry_run=dry_run)
    openai_client = get_openai_client(dry_run=dry_run)

    # Preprocess
    query_info = preprocess_query(query)
    print(f"[Groq Demo] Preprocessed: {query_info['word_count']} words")

    # Strategy decision
    strategy = choose_provider_strategy(query_info)
    print(f"[Groq Demo] Strategy: {strategy['chosen']}")

    # Phase 1: Groq analysis
    print()
    print("=" * 60)
    print("PHASE 1: GROQ ANALYSIS")
    print("=" * 60)
    groq_result = await run_groq_analyzer(query, groq_client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print(f"  Answer: {groq_result['answer'][:120]}...")

    # Phase 2: OpenAI function calling
    print()
    print("=" * 60)
    print("PHASE 2: OPENAI FUNCTION CALLING")
    print("=" * 60)
    fc_result = await run_function_caller(query, openai_client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print(f"  Answer: {fc_result['answer'][:120]}...")

    waxell.score("overall_quality", 0.88, comment="Combined pipeline quality")

    print()
    print(f"[Groq Demo] Complete. (4 LLM calls, 2 child agents, {fc_result['tools_used']} tool calls)")

    return {
        "groq_answer": groq_result["answer"][:200],
        "fc_answer": fc_result["answer"][:200],
        "tools_used": fc_result["tools_used"],
        "strategy": strategy["chosen"],
    }


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

    print("[Groq Agent] Starting Groq & function calling showcase...")
    print(f"[Groq Agent] Session: {session}")
    print(f"[Groq Agent] End user: {user_id} ({user_group})")
    print(f"[Groq Agent] Query: {query[:80]}...")
    print()

    await run_agent(query, dry_run=args.dry_run)

    waxell_observe.shutdown()


if __name__ == "__main__":
    import waxell_observe
    asyncio.run(main())
