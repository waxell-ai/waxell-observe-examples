#!/usr/bin/env python3
"""Prompt management & background collector demo.

Demonstrates get_prompt()/PromptInfo.compile(), the background collector
for LLM calls made outside any WaxellContext, get_context(), and
capture_content mode.

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

import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError
from waxell_observe.types import PromptInfo

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Explain the key principles of responsible AI governance"


async def run_prompt_management_demo(
    query: str,
    client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
) -> None:
    """Execute the prompt management & background collector showcase."""

    observe_client = get_observe_client()

    # ==================================================================
    # PHASE 1: PROMPT MANAGEMENT (get_prompt / PromptInfo.compile)
    # ==================================================================
    print("=" * 70)
    print("PHASE 1: PROMPT MANAGEMENT")
    print("  Demonstrates get_prompt() to fetch prompts from the controlplane")
    print("  and PromptInfo.compile() to render template variables.")
    print("=" * 70)
    print()

    # Try to fetch a prompt from the controlplane
    prompt = None
    print("[Phase 1] Attempting to fetch prompt 'demo-greeting' from controlplane...")
    try:
        prompt = await observe_client.get_prompt("demo-greeting", label="production")
        if prompt and prompt.content:
            print(f"  Fetched prompt: name={prompt.name}, version={prompt.version}")
            print(f"  Type: {prompt.prompt_type}")
            print(f"  Labels: {prompt.labels}")
            print(f"  Content preview: {str(prompt.content)[:100]}...")
        else:
            print("  Prompt returned but content is empty")
            prompt = None
    except Exception as e:
        print(f"  Could not fetch from server (expected if no prompts configured): {e}")
        prompt = None

    # Fall back to local PromptInfo to demonstrate compile()
    if not prompt:
        print()
        print("[Phase 1] Creating local PromptInfo to demonstrate compile()...")
        prompt = PromptInfo(
            name="demo-greeting",
            version=1,
            prompt_type="text",
            content="Hello {{name}}! Let's discuss {{topic}}. Please provide a thorough analysis of the key considerations.",
            config={"temperature": 0.7, "max_tokens": 500},
            labels=["demo", "local"],
        )
        print(f"  Template: \"{prompt.content}\"")

    # Compile the prompt with variables
    compiled = prompt.compile(name="User", topic=query[:50])
    print(f"  Compiled: \"{compiled}\"")
    print()

    # Use compiled prompt in an LLM call
    print("[Phase 1] Using compiled prompt as LLM input...")
    async with WaxellContext(
        agent_name="prompt-management-demo",
        workflow_name="prompt-compile",
        inputs={"query": query, "prompt_name": prompt.name, "prompt_version": prompt.version},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable AI governance advisor."},
                {"role": "user", "content": str(compiled)},
            ],
        )
        answer = response.choices[0].message.content
        ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="compiled_prompt_call",
            prompt_preview=str(compiled)[:200],
            response_preview=answer[:200],
        )
        ctx.record_step("prompt_compile", output={
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            "compiled_length": len(str(compiled)),
        })
        ctx.set_result({"answer": answer[:200], "prompt_used": prompt.name})
        print(f"  Response: {answer[:120]}...")

    print()

    # Also demonstrate chat-type prompt compilation
    print("[Phase 1] Demonstrating chat-type PromptInfo.compile()...")
    chat_prompt = PromptInfo(
        name="demo-chat",
        version=1,
        prompt_type="chat",
        content=[
            {"role": "system", "content": "You are {{role}}. Help the user with {{topic}}."},
            {"role": "user", "content": "{{question}}"},
        ],
        config={"temperature": 0.5},
        labels=["demo", "chat"],
    )
    compiled_chat = chat_prompt.compile(
        role="an AI governance expert",
        topic="responsible deployment",
        question=query,
    )
    print(f"  Chat template rendered ({len(compiled_chat)} messages):")
    for msg in compiled_chat:
        print(f"    [{msg['role']}] {msg['content'][:80]}...")
    print()

    # ==================================================================
    # PHASE 2: get_context() DEMONSTRATION
    # ==================================================================
    print("=" * 70)
    print("PHASE 2: get_context()")
    print("  Shows waxell.get_context() returns the active WaxellContext")
    print("  inside a context, and None outside.")
    print("=" * 70)
    print()

    # Outside context
    outside_ctx = waxell_observe.get_context()
    print(f"[Phase 2] Outside WaxellContext: get_context() = {outside_ctx}")

    # Inside context
    async with WaxellContext(
        agent_name="prompt-management-demo",
        workflow_name="get-context-test",
        inputs={"test": "get_context"},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        inside_ctx = waxell_observe.get_context()
        print(f"[Phase 2] Inside WaxellContext:  get_context() = {type(inside_ctx).__name__}")
        print(f"[Phase 2] Same object as ctx?    {inside_ctx is ctx}")

        # Use top-level convenience functions (they look up the context automatically)
        waxell_observe.tag("test", "get_context")
        waxell_observe.metadata("phase", 2)
        waxell_observe.score("test_passed", True, data_type="boolean")
        print("[Phase 2] Used waxell.tag(), waxell.metadata(), waxell.score() via get_context()")

    # Back outside
    after_ctx = waxell_observe.get_context()
    print(f"[Phase 2] After WaxellContext:  get_context() = {after_ctx}")
    print()

    # ==================================================================
    # PHASE 3: BACKGROUND COLLECTOR
    # ==================================================================
    print("=" * 70)
    print("PHASE 3: BACKGROUND COLLECTOR")
    print("  Demonstrates how LLM calls outside any WaxellContext are")
    print("  buffered and auto-collected into system-generated runs.")
    print("=" * 70)
    print()

    from waxell_observe.instrumentors._collector import _collector

    print("[Phase 3] Simulating 3 LLM calls outside any WaxellContext...")
    print("  (In production, these fire automatically via the instrumentor when")
    print("   _current_context.get() returns None)")
    print()

    # Simulate buffered calls (in production, the instrumentor wrapper does this)
    call_1 = {
        "model": "gpt-4o-mini",
        "tokens_in": 120,
        "tokens_out": 65,
        "cost": 0.00012,
        "task": "background_call_1",
    }
    call_2 = {
        "model": "gpt-4o-mini",
        "tokens_in": 200,
        "tokens_out": 110,
        "cost": 0.00022,
        "task": "background_call_2",
    }
    call_3 = {
        "model": "gpt-4o",
        "tokens_in": 350,
        "tokens_out": 180,
        "cost": 0.0045,
        "task": "background_call_3",
    }

    _collector.record_call(call_1)
    print(f"  Buffered: {call_1['model']} ({call_1['tokens_in']}+{call_1['tokens_out']} tokens)")
    _collector.record_call(call_2)
    print(f"  Buffered: {call_2['model']} ({call_2['tokens_in']}+{call_2['tokens_out']} tokens)")
    _collector.record_call(call_3)
    print(f"  Buffered: {call_3['model']} ({call_3['tokens_in']}+{call_3['tokens_out']} tokens)")
    print()

    print("[Phase 3] Flushing collector (creates auto-run: 'auto:gpt-4o-mini')...")
    _collector.flush()
    print("  Flush complete. The auto-run should appear in agent-executions as")
    print("  'auto:gpt-4o-mini' with 3 LLM calls and auto-generated workflow.")
    print()

    print("[Phase 3] How it works in production:")
    print("  1. You call openai.chat.completions.create() OUTSIDE any WaxellContext")
    print("  2. The instrumentor wrapper sees _current_context.get() == None")
    print("  3. Call data is pushed to the BackgroundCollector buffer")
    print("  4. A daemon thread flushes every 5 seconds, creating auto-runs")
    print("  5. On shutdown(), any remaining calls are flushed")
    print()

    # ==================================================================
    # PHASE 4: capture_content MODE
    # ==================================================================
    print("=" * 70)
    print("PHASE 4: capture_content MODE")
    print("  This demo was initialized with capture_content=True.")
    print("  Prompt and response text are included in OTel span attributes:")
    print("  gen_ai.input.messages and gen_ai.output.messages")
    print("=" * 70)
    print()

    async with WaxellContext(
        agent_name="prompt-management-demo",
        workflow_name="capture-content",
        inputs={"query": query, "capture_content": True},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
    ) as ctx:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a brief, helpful assistant."},
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content
        ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="capture_content_demo",
            prompt_preview=query[:200],
            response_preview=answer[:200],
        )
        ctx.record_step("capture_content_test", output={"content_captured": True})
        print(f"  LLM call made with capture_content=True")
        print(f"  Response: {answer[:120]}...")
        print(f"  Prompt and response text are now available in trace span attributes")

    print()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 70)
    print("PROMPT MANAGEMENT DEMO SUMMARY")
    print("=" * 70)
    print("  Phase 1: get_prompt() + PromptInfo.compile() (text & chat types)")
    print("  Phase 2: get_context() inside/outside WaxellContext")
    print("  Phase 3: Background collector (3 buffered calls → auto-run)")
    print("  Phase 4: capture_content=True mode for full trace content")
    print()
    print("  SDK features exercised:")
    print("    client.get_prompt()          — fetch prompts from controlplane")
    print("    PromptInfo.compile()         — template variable rendering")
    print("    waxell.get_context()         — current context lookup")
    print("    BackgroundCollector          — context-free LLM call collection")
    print("    init(capture_content=True)   — full content in traces")
    print()
    print("[Prompt Management Demo] Complete. (2 LLM calls, 5 steps, 3 bg calls)")


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Prompt Management Agent] Starting prompt management & bg collector showcase...")
    print(f"[Prompt Management Agent] Session: {session}")
    print(f"[Prompt Management Agent] End user: {user_id} ({user_group})")
    print(f"[Prompt Management Agent] Query: {query[:80]}...")
    print(f"[Prompt Management Agent] capture_content=True (full content in traces)")
    print()

    await run_prompt_management_demo(
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
