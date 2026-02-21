#!/usr/bin/env python3
"""Prompt management & background collector demo.

Demonstrates get_prompt()/PromptInfo.compile(), the background collector
for LLM calls made outside any WaxellContext, get_context(), and
capture_content mode.

Multi-agent architecture:

  prompt-mgmt-orchestrator (parent)
  ├── @step: prepare_prompt_templates
  ├── @decision: choose_prompt_strategy (OpenAI)
  ├── waxell.decide(): content capture routing
  ├── prompt-runner (child)
  │   ├── @tool: compile_text_prompt, compile_chat_prompt, fetch_remote_prompt
  │   └── LLM calls with compiled prompts
  └── prompt-evaluator (child)
      ├── @reasoning: assess_prompt_quality
      ├── @tool: run_background_collector, test_get_context
      ├── waxell.score(): prompt effectiveness
      └── LLM call with capture_content

Usage::

    # Dry-run (no API key needed)
    python examples/14_prompt_management_agent/prompt_management_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/14_prompt_management_agent/prompt_management_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
# Enable capture_content to include prompt/response text in traces
from _common import setup_observe

setup_observe(capture_content=True)

import waxell_observe as waxell
from waxell_observe import generate_session_id
from waxell_observe.types import PromptInfo

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Explain the key principles of responsible AI governance"


# ---------------------------------------------------------------------------
# @step decorator — prepare prompt templates
# ---------------------------------------------------------------------------


@waxell.step_dec(name="prepare_prompt_templates")
async def prepare_prompt_templates(query: str) -> dict:
    """Set up prompt templates for the demo."""
    return {
        "text_template": "Hello {{name}}! Let's discuss {{topic}}.",
        "chat_template_messages": 2,
        "query_length": len(query),
        "templates_ready": True,
    }


# ---------------------------------------------------------------------------
# @decision decorator — choose prompt strategy
# ---------------------------------------------------------------------------


@waxell.decision(name="choose_prompt_strategy", options=["text", "chat", "hybrid"])
async def choose_prompt_strategy(query: str, openai_client) -> dict:
    """Decide which prompt compilation strategy to use."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the best prompt strategy as exactly one of: text, chat, hybrid. "
                    'Respond with JSON: {"chosen": "...", "reasoning": "..."}'
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    content = response.choices[0].message.content
    try:
        import json
        return json.loads(content)
    except Exception:
        return {"chosen": "hybrid", "reasoning": content[:200]}


# ---------------------------------------------------------------------------
# @tool decorators — prompt operations
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="prompt_management")
def compile_text_prompt(template_content: str, variables: dict) -> dict:
    """Compile a text-type PromptInfo with variables."""
    prompt = PromptInfo(
        name="demo-greeting", version=1, prompt_type="text",
        content=template_content,
        config={"temperature": 0.7, "max_tokens": 500},
        labels=["demo", "local"],
    )
    compiled = prompt.compile(**variables)
    return {"compiled": str(compiled), "length": len(str(compiled)), "prompt_name": prompt.name}


@waxell.tool(tool_type="prompt_management")
def compile_chat_prompt(role: str, topic: str, question: str) -> dict:
    """Compile a chat-type PromptInfo with variables."""
    chat_prompt = PromptInfo(
        name="demo-chat", version=1, prompt_type="chat",
        content=[
            {"role": "system", "content": "You are {{role}}. Help the user with {{topic}}."},
            {"role": "user", "content": "{{question}}"},
        ],
        config={"temperature": 0.5},
        labels=["demo", "chat"],
    )
    compiled = chat_prompt.compile(role=role, topic=topic, question=question)
    return {"messages": compiled, "message_count": len(compiled), "prompt_name": chat_prompt.name}


@waxell.tool(tool_type="prompt_management")
async def fetch_remote_prompt(observe_client) -> dict:
    """Attempt to fetch a prompt from the controlplane."""
    try:
        prompt = await observe_client.get_prompt("demo-greeting", label="production")
        if prompt and prompt.content:
            return {"found": True, "name": prompt.name, "version": prompt.version}
    except Exception as e:
        return {"found": False, "error": str(e)[:100]}
    return {"found": False, "error": "Empty content"}


@waxell.tool(tool_type="testing")
def test_get_context() -> dict:
    """Test waxell.get_context() inside and outside contexts."""
    ctx = waxell.get_context()
    return {"context_found": ctx is not None, "context_type": type(ctx).__name__ if ctx else "None"}


@waxell.tool(tool_type="collector")
def run_background_collector() -> dict:
    """Simulate background collector buffering and flushing."""
    from waxell_observe.instrumentors._collector import _collector

    calls = [
        {"model": "gpt-4o-mini", "tokens_in": 120, "tokens_out": 65, "cost": 0.00012, "task": "bg_call_1"},
        {"model": "gpt-4o-mini", "tokens_in": 200, "tokens_out": 110, "cost": 0.00022, "task": "bg_call_2"},
        {"model": "gpt-4o", "tokens_in": 350, "tokens_out": 180, "cost": 0.0045, "task": "bg_call_3"},
    ]
    for call in calls:
        _collector.record_call(call)
    _collector.flush()
    return {"calls_buffered": len(calls), "flushed": True}


# ---------------------------------------------------------------------------
# @retrieval decorator — gather prompt compilation results
# ---------------------------------------------------------------------------


@waxell.retrieval(source="prompt_registry")
def gather_prompt_results(text_result: dict, chat_result: dict) -> list[dict]:
    """Gather and rank prompt compilation outcomes."""
    return [
        {"type": "text", "name": text_result["prompt_name"], "length": text_result["length"]},
        {"type": "chat", "name": chat_result["prompt_name"], "messages": chat_result["message_count"]},
    ]


# ---------------------------------------------------------------------------
# @reasoning decorator — assess prompt quality
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="prompt_quality_assessment")
async def assess_prompt_quality(text_result: dict, chat_result: dict) -> dict:
    """Assess quality of compiled prompts."""
    return {
        "thought": f"Text prompt compiled to {text_result['length']} chars. "
                   f"Chat prompt has {chat_result['message_count']} messages. "
                   "Both templates rendered successfully with variable substitution.",
        "evidence": [f"Text: {text_result['prompt_name']}", f"Chat: {chat_result['prompt_name']}"],
        "conclusion": "Prompt compilation working correctly for both text and chat types.",
    }


# ---------------------------------------------------------------------------
# Child Agent 1: Prompt Runner — compiles and uses prompts
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="prompt-runner", workflow_name="prompt-compilation")
async def run_prompt_compilation(
    query: str,
    openai_client,
    observe_client,
    dry_run: bool = False,
    waxell_ctx=None,
) -> dict:
    """Compile prompts and use them in LLM calls."""
    waxell.tag("agent_role", "prompt_runner")
    waxell.tag("prompt_types", "text,chat")

    # Try fetching remote prompt
    print("  [Prompt Runner] Fetching remote prompt...")
    remote_result = await fetch_remote_prompt(observe_client=observe_client)
    print(f"    Remote prompt found: {remote_result.get('found', False)}")

    # Compile text prompt
    print("  [Prompt Runner] Compiling text prompt...")
    text_result = compile_text_prompt(
        template_content="Hello {{name}}! Let's discuss {{topic}}. Please provide a thorough analysis of the key considerations.",
        variables={"name": "User", "topic": query[:50]},
    )
    print(f"    Compiled text prompt: {text_result['length']} chars")

    # Use compiled prompt in LLM call
    print("  [Prompt Runner] Using compiled prompt in LLM call...")
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a knowledgeable AI governance advisor."},
            {"role": "user", "content": text_result["compiled"]},
        ],
    )
    text_answer = response.choices[0].message.content

    # Compile chat prompt
    print("  [Prompt Runner] Compiling chat prompt...")
    chat_result = compile_chat_prompt(
        role="an AI governance expert",
        topic="responsible deployment",
        question=query,
    )
    print(f"    Chat template: {chat_result['message_count']} messages")
    for msg in chat_result["messages"]:
        print(f"      [{msg['role']}] {msg['content'][:80]}...")

    waxell.metadata("prompts_compiled", 2)
    waxell.metadata("remote_prompt_found", remote_result.get("found", False))

    return {
        "text_result": text_result,
        "chat_result": chat_result,
        "text_answer": text_answer[:200],
        "remote_found": remote_result.get("found", False),
    }


# ---------------------------------------------------------------------------
# Child Agent 2: Prompt Evaluator — tests context, collector, capture
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="prompt-evaluator", workflow_name="prompt-evaluation")
async def run_prompt_evaluation(
    query: str,
    compilation_results: dict,
    openai_client,
    dry_run: bool = False,
    waxell_ctx=None,
) -> dict:
    """Evaluate prompt compilation, test get_context, background collector, capture_content."""
    waxell.tag("agent_role", "prompt_evaluator")
    waxell.tag("provider", "openai")

    text_result = compilation_results["text_result"]
    chat_result = compilation_results["chat_result"]

    # @retrieval — gather prompt results
    prompt_results = gather_prompt_results(text_result=text_result, chat_result=chat_result)
    print(f"  [Evaluator] Gathered {len(prompt_results)} prompt results")

    # @reasoning — assess quality
    print("  [Evaluator] Assessing prompt quality...")
    quality = await assess_prompt_quality(text_result=text_result, chat_result=chat_result)
    print(f"    Conclusion: {quality.get('conclusion', 'N/A')}")

    # Test get_context()
    print("  [Evaluator] Testing get_context()...")
    ctx_result = test_get_context()
    print(f"    Context found: {ctx_result['context_found']} ({ctx_result['context_type']})")

    # Use top-level convenience functions
    waxell.tag("test", "get_context")
    waxell.metadata("phase", "evaluation")
    waxell.score("context_test_passed", True, data_type="boolean")

    # Background collector
    print("  [Evaluator] Running background collector test...")
    bg_result = run_background_collector()
    print(f"    Buffered {bg_result['calls_buffered']} calls, flushed: {bg_result['flushed']}")

    # capture_content mode LLM call
    print("  [Evaluator] Testing capture_content mode...")
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a brief, helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    answer = response.choices[0].message.content
    print(f"    Capture content response: {answer[:120]}...")

    # Scores
    waxell.score("prompt_compilation_quality", 0.92, comment="Both text and chat prompts compiled successfully")
    waxell.score("context_api_working", True, data_type="boolean", comment="get_context() returns active context")
    waxell.score("background_collector_working", True, data_type="boolean", comment="Collector buffers and flushes")

    return {
        "answer": answer,
        "quality_assessment": quality,
        "context_test": ctx_result,
        "background_collector": bg_result,
        "capture_content": True,
    }


# ---------------------------------------------------------------------------
# Orchestrator — coordinates prompt-runner + prompt-evaluator
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="prompt-mgmt-orchestrator", workflow_name="prompt-management-pipeline")
async def run_agent(
    query: str,
    dry_run: bool = False,
    waxell_ctx=None,
    **kwargs,
) -> dict:
    """Orchestrate prompt management demo across child agents."""
    waxell.tag("demo", "prompt_management")
    waxell.tag("category", "prompt_management")
    waxell.metadata("architecture", "orchestrator → prompt-runner + prompt-evaluator")
    waxell.metadata("features", ["get_prompt", "compile", "get_context", "background_collector", "capture_content"])
    waxell.metadata("mode", "dry-run" if dry_run else "live")
    waxell.metadata("capture_content", True)

    openai_client = get_openai_client(dry_run=dry_run)
    observe_client = get_observe_client()

    # Phase 1: @step — prepare prompt templates
    print("[Orchestrator] Phase 1: Preparing prompt templates (@step)...")
    prep = await prepare_prompt_templates(query)
    print(f"  Templates ready: {prep['templates_ready']}")

    # Phase 2: @decision — choose prompt strategy
    print("[Orchestrator] Phase 2: Choosing prompt strategy (@decision)...")
    strategy = await choose_prompt_strategy(query=query, openai_client=openai_client)
    chosen = strategy.get("chosen", "hybrid") if isinstance(strategy, dict) else str(strategy)
    print(f"  Strategy: {chosen}")

    # Phase 3: waxell.decide() — content capture routing
    print("[Orchestrator] Phase 3: Content capture routing (waxell.decide())...")
    waxell.decide(
        "capture_mode",
        chosen="full_capture",
        options=["full_capture", "metadata_only", "disabled"],
        reasoning=f"Strategy '{chosen}' with capture_content=True — using full content capture",
        confidence=0.95,
    )

    # Phase 4: prompt-runner child agent
    print("[Orchestrator] Phase 4: Compiling prompts (prompt-runner)...")
    compilation_results = await run_prompt_compilation(
        query=query,
        openai_client=openai_client,
        observe_client=observe_client,
        dry_run=dry_run,
    )
    print(f"  Compiled {2} prompts, text answer: {compilation_results['text_answer'][:80]}...")

    # Phase 5: prompt-evaluator child agent
    print("[Orchestrator] Phase 5: Evaluating prompts (prompt-evaluator)...")
    evaluation = await run_prompt_evaluation(
        query=query,
        compilation_results=compilation_results,
        openai_client=openai_client,
        dry_run=dry_run,
    )

    return {
        "answer": evaluation["answer"],
        "quality_assessment": evaluation["quality_assessment"],
        "prompts_compiled": 2,
        "context_test": evaluation["context_test"],
        "background_collector": evaluation["background_collector"],
        "capture_content": evaluation["capture_content"],
        "pipeline": "prompt-mgmt-orchestrator → prompt-runner + prompt-evaluator",
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
    observe_active = not is_observe_disabled()

    print("[Prompt Management Agent] Starting prompt management & bg collector showcase...")
    print(f"[Prompt Management Agent] Session: {session}")
    print(f"[Prompt Management Agent] End user: {user_id} ({user_group})")
    print(f"[Prompt Management Agent] Query: {query[:80]}...")
    print(f"[Prompt Management Agent] capture_content=True (full content in traces)")
    print()

    result = await run_agent(
        query=query,
        dry_run=args.dry_run,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        client=get_observe_client(),
    )

    print()
    print("=" * 60)
    print("[Result]")
    print(f"  Answer:      {result['answer'][:200]}...")
    print(f"  Prompts:     {result['prompts_compiled']} compiled")
    print(f"  Context:     {result['context_test']}")
    print(f"  Collector:   {result['background_collector']}")
    print(f"  Pipeline:    {result['pipeline']}")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
