#!/usr/bin/env python3
"""Minimal waxell-observe quickstart.

The simplest possible example: initialize, make an LLM call, done.
All tracing happens automatically.

Usage::

    # Dry-run (no API keys needed)
    python examples/16_basic_quickstart/quickstart.py --dry-run

    # With real API calls
    WAXELL_API_KEY=wax_sk_... OPENAI_API_KEY=sk-... python examples/16_basic_quickstart/quickstart.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import setup_observe

# Step 1: Initialize waxell-observe BEFORE importing LLM SDKs
setup_observe()

import asyncio
import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from _common import get_openai_client, get_observe_client, is_observe_disabled, parse_args, pick_demo_user


async def main() -> None:
    args = parse_args()
    query = args.query or "What is observability for AI agents?"

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    demo_user = pick_demo_user()

    print(f"[Quickstart] Query: {query}")
    print(f"[Quickstart] Session: {session}")
    print()

    # Step 2: Wrap your agent logic in a WaxellContext
    async with WaxellContext(
        agent_name="quickstart",
        workflow_name="basic-qa",
        inputs={"query": query},
        session_id=session,
        user_id=demo_user["user_id"],
        user_group=demo_user["user_group"],
        client=get_observe_client(),
    ) as ctx:
        # Step 3: Make LLM calls as usual — they're auto-traced
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
        )
        answer = response.choices[0].message.content

        # Step 4: Record steps and results for the trace
        ctx.record_step("answer", output={"length": len(answer)})
        ctx.record_llm_call(
            model=response.model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            task="answer",
        )
        ctx.set_result({"answer": answer})

        print(f"[Quickstart] Answer: {answer[:200]}...")

    # Step 5: Flush telemetry
    waxell_observe.shutdown()
    print("[Quickstart] Done. Check your Waxell dashboard for the trace.")


if __name__ == "__main__":
    asyncio.run(main())
