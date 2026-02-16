#!/usr/bin/env python3
"""LangChain integration demo.

Demonstrates waxell-observe's LangChain callback handler with automatic
chain/LLM span nesting and telemetry capture.

Usage::

    # Dry-run (no API key needed, uses FakeListChatModel)
    python examples/15_langchain_agent/langchain_agent.py --dry-run

    # With real OpenAI calls via LangChain
    OPENAI_API_KEY=sk-... python examples/15_langchain_agent/langchain_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import generate_session_id
from waxell_observe.integrations.langchain import WaxellLangChainHandler

from _common import (
    get_observe_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "What are the ethical implications of AI in healthcare?"

# Pre-defined responses for dry-run mode
_FAKE_RESPONSES = [
    (
        "Analysis: The ethical implications of AI in healthcare span several "
        "critical dimensions including patient privacy, algorithmic bias, "
        "informed consent, and the balance between automation and human "
        "oversight. Key concerns include the potential for AI systems to "
        "perpetuate existing health disparities, the challenge of explaining "
        "AI-driven diagnoses to patients, and the need for robust validation "
        "frameworks before clinical deployment."
    ),
    (
        "Summary: AI in healthcare raises important ethical questions around "
        "privacy, bias, and accountability. Organizations must implement "
        "strong governance frameworks, ensure algorithmic transparency, and "
        "maintain human-in-the-loop oversight for critical medical decisions. "
        "The path forward requires collaboration between technologists, "
        "ethicists, clinicians, and policymakers."
    ),
]


def _get_llm(dry_run: bool):
    """Return a LangChain LLM -- fake for dry-run, real ChatOpenAI otherwise."""
    if dry_run or not os.environ.get("OPENAI_API_KEY"):
        mode = "dry-run" if dry_run else "no OPENAI_API_KEY"
        print(f"[LangChain Demo] Using FakeListChatModel ({mode})")
        try:
            from langchain_community.chat_models import FakeListChatModel
        except ImportError:
            from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(responses=_FAKE_RESPONSES)

    print("[LangChain Demo] Using ChatOpenAI (gpt-4o-mini)")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    session = generate_session_id()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[LangChain Demo] Starting LangChain pipeline...")
    print(f"[LangChain Demo] Session: {session}")
    print(f"[LangChain Demo] End user: {user_id} ({user_group})")
    print(f"[LangChain Demo] Query: {query}")
    print()

    # Create the Waxell callback handler
    handler = WaxellLangChainHandler(
        agent_name="langchain-demo",
        workflow_name="langchain-pipeline",
        client=get_observe_client(),
        session_id=session,
        user_id=user_id,
        user_group=user_group,
    )

    llm = _get_llm(dry_run=args.dry_run)

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("[LangChain Demo] ERROR: langchain-core not installed.")
        print("  Install with: pip install langchain-core langchain-openai")
        sys.exit(1)

    # Chain 1: Analyze the topic
    print("[LangChain Demo] Step 1/2: Analyzing topic...")

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Provide a detailed analysis of the given topic."),
        ("user", "Analyze this topic in depth: {topic}"),
    ])
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    analysis = analysis_chain.invoke(
        {"topic": query},
        config={"callbacks": [handler]},
    )
    print(f"  Analysis: {analysis[:150]}...")
    print()

    # Chain 2: Summarize the analysis
    print("[LangChain Demo] Step 2/2: Summarizing analysis...")

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise summarizer. Create a brief summary of the analysis."),
        ("user", "Summarize this analysis:\n\n{analysis}"),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke(
        {"analysis": analysis},
        config={"callbacks": [handler]},
    )
    print(f"  Summary: {summary[:150]}...")
    print()

    # Flush telemetry
    handler.flush_sync(result={"analysis": analysis[:500], "summary": summary[:500]})

    print(f"[LangChain Demo] Complete. (2 LLM calls via LangChain, auto-captured steps)")

    waxell_observe.shutdown()


if __name__ == "__main__":
    main()
