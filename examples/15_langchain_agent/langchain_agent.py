#!/usr/bin/env python3
"""LangChain multi-agent pipeline demo with modern SDK decorator patterns.

Demonstrates waxell-observe decorator-based instrumentation across a
LangChain-powered analysis pipeline with 3 agents:

  langchain-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: classify_topic
  ├── waxell.decide(): routing decision
  ├── langchain-runner (child)
  │   ├── @tool: format_prompt, parse_output
  │   ├── @retrieval: retrieve_context
  │   └── LangChain chain invocation (auto-instrumented LLM)
  └── langchain-evaluator (child)
      ├── @reasoning: evaluate_quality
      ├── waxell.score(): quality scores
      └── LangChain summary chain (auto-instrumented LLM)

SDK primitives demonstrated:

  | Primitive   | Decorator      | Manual          | Auto               |
  |-------------|----------------|-----------------|--------------------|
  | LLM calls   | —              | —               | auto-instrumented  |
  | Tool calls  | @tool          | —               | —                  |
  | Decisions   | @decision      | decide()        | —                  |
  | Retrieval   | @retrieval     | —               | —                  |
  | Reasoning   | @reasoning     | —               | —                  |
  | Steps       | @step          | —               | —                  |
  | Scores      | —              | score()         | —                  |
  | Tags        | —              | tag()           | —                  |
  | Metadata    | —              | metadata()      | —                  |

Usage::

    # Dry-run (no API key needed, uses FakeListChatModel)
    python examples/15_langchain_agent/langchain_agent.py --dry-run

    # With real OpenAI calls via LangChain
    OPENAI_API_KEY=sk-... python examples/15_langchain_agent/langchain_agent.py

    # Custom query
    python examples/15_langchain_agent/langchain_agent.py --query "Analyze the future of quantum computing"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

# CRITICAL: init() BEFORE importing openai so auto-instrumentors can patch
from _common import setup_observe

setup_observe()

# NOW safe to import provider clients (auto-instrumentors have patched them)
import waxell_observe as waxell
from waxell_observe import generate_session_id

from _common import (
    get_observe_client,
    get_openai_client,
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

# Mock context documents for retrieval
_MOCK_CONTEXT_DOCS = [
    {"id": "lc-001", "title": "AI Ethics in Medicine", "content": "AI systems in healthcare must prioritize patient safety, informed consent, and equitable access to care.", "tags": ["ethics", "healthcare"]},
    {"id": "lc-002", "title": "Algorithmic Bias in Clinical AI", "content": "Machine learning models trained on biased datasets can perpetuate and amplify health disparities across demographic groups.", "tags": ["bias", "fairness"]},
    {"id": "lc-003", "title": "Governance Frameworks for Health AI", "content": "Regulatory frameworks like FDA guidance and EU AI Act provide structured approaches to governing AI in clinical settings.", "tags": ["governance", "regulation"]},
]


# ---------------------------------------------------------------------------
# @step decorator -- auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="preprocess_query")
async def preprocess_query(query: str) -> dict:
    """Clean and normalize the query for LangChain processing."""
    cleaned = query.strip()
    tokens = cleaned.lower().split()
    return {"original": query, "cleaned": cleaned, "token_count": len(tokens)}


# ---------------------------------------------------------------------------
# @decision decorator -- classify the topic
# ---------------------------------------------------------------------------


@waxell.decision(name="classify_topic", options=["technical", "ethical", "policy", "general"])
async def classify_topic(query: str, openai_client) -> dict:
    """Classify a query to decide the analysis strategy."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the following query as exactly one of: technical, ethical, policy, general. "
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
        return {"chosen": "ethical", "reasoning": content[:200]}


@waxell.decision(name="output_format", options=["brief", "detailed", "bullet_points"])
def choose_output_format(num_docs: int, context: str) -> dict:
    """Choose output format based on context document count."""
    format_choice = "detailed" if num_docs > 2 else "brief"
    return {
        "chosen": format_choice,
        "reasoning": f"{context} -- {format_choice} format provides better coverage",
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# @tool decorators -- auto-record LangChain helper operations
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="prompt_formatter")
def format_prompt(template: str, variables: dict) -> str:
    """Format a prompt template with variables."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


@waxell.tool(tool_type="output_parser")
def parse_output(raw_output: str, max_length: int = 500) -> dict:
    """Parse and truncate LangChain output."""
    cleaned = raw_output.strip()
    return {
        "content": cleaned[:max_length],
        "length": len(cleaned),
        "truncated": len(cleaned) > max_length,
    }


# ---------------------------------------------------------------------------
# @retrieval decorator -- auto-record context retrieval
# ---------------------------------------------------------------------------


@waxell.retrieval(source="langchain_context")
def retrieve_context(query: str, documents: list) -> list[dict]:
    """Retrieve relevant context documents for LangChain processing."""
    query_words = {w.lower() for w in query.split() if len(w) > 2}
    scored = []
    for doc in documents:
        searchable = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
        score = sum(1 for w in query_words if w in searchable)
        if score > 0:
            scored.append({**doc, "relevance_score": round(score / max(len(query_words), 1), 2)})
    scored.sort(key=lambda d: d["relevance_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# @reasoning decorator -- auto-record quality assessment
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="quality_evaluation")
async def evaluate_quality(analysis: str, summary: str, context_docs: list) -> dict:
    """Evaluate the quality of the analysis and summary against context."""
    doc_titles = [d.get("title", "unknown") for d in context_docs]
    analysis_refs = len([t for t in doc_titles if t.lower().replace(" ", "") in analysis.lower().replace(" ", "")])
    summary_refs = len([t for t in doc_titles if t.lower().replace(" ", "") in summary.lower().replace(" ", "")])
    return {
        "thought": f"Analysis references {analysis_refs}/{len(context_docs)} docs, "
                   f"summary references {summary_refs}/{len(context_docs)} docs. "
                   f"Checking coherence and completeness.",
        "evidence": [f"Source: {t}" for t in doc_titles],
        "conclusion": "Both outputs adequately grounded" if (analysis_refs + summary_refs) > 0 else "Outputs may need more grounding",
    }


# ---------------------------------------------------------------------------
# Helper: get LangChain LLM
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Agent 1: langchain-runner -- executes the LangChain analysis chain
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="langchain-runner", workflow_name="langchain-analysis")
async def run_langchain_analysis(query: str, context_docs: list, dry_run: bool = False, waxell_ctx=None):
    """Run the LangChain analysis chain with context retrieval and tool recording."""
    waxell.tag("agent_role", "runner")
    waxell.tag("framework", "langchain")
    waxell.metadata("context_doc_count", len(context_docs))

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("[Runner] ERROR: langchain-core not installed.")
        print("  Install with: pip install langchain-core langchain-openai")
        sys.exit(1)

    llm = _get_llm(dry_run=dry_run)

    # @tool: format the prompt with context
    context_text = "\n".join(f"- {d['title']}: {d['content'][:80]}..." for d in context_docs)
    formatted = format_prompt(
        template="Analyze this topic in depth: {topic}\n\nContext:\n{context}",
        variables={"topic": query, "context": context_text},
    )
    waxell.metadata("prompt_length", len(formatted))

    # Run the LangChain analysis chain (LLM call auto-instrumented)
    print("  [Runner] Running LangChain analysis chain...")
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Provide a detailed analysis of the given topic using the provided context."),
        ("user", "Analyze this topic in depth: {topic}\n\nContext:\n{context}"),
    ])
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    analysis = analysis_chain.invoke(
        {"topic": query, "context": context_text},
    )
    print(f"  [Runner] Analysis: {analysis[:150]}...")

    # @tool: parse the output
    parsed = parse_output(raw_output=analysis, max_length=500)
    waxell.metadata("analysis_length", parsed["length"])
    waxell.metadata("analysis_truncated", parsed["truncated"])

    return {"analysis": analysis, "parsed": parsed}


# ---------------------------------------------------------------------------
# Agent 2: langchain-evaluator -- evaluates and summarizes
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="langchain-evaluator", workflow_name="langchain-evaluation")
async def run_langchain_evaluation(query: str, analysis: str, context_docs: list, dry_run: bool = False, waxell_ctx=None):
    """Evaluate the analysis and generate a summary with quality scoring."""
    waxell.tag("agent_role", "evaluator")
    waxell.tag("framework", "langchain")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("[Evaluator] ERROR: langchain-core not installed.")
        sys.exit(1)

    llm = _get_llm(dry_run=dry_run)

    # Run the LangChain summary chain (LLM call auto-instrumented)
    print("  [Evaluator] Running LangChain summary chain...")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise summarizer. Create a brief summary of the analysis."),
        ("user", "Summarize this analysis:\n\n{analysis}"),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke(
        {"analysis": analysis},
    )
    print(f"  [Evaluator] Summary: {summary[:150]}...")

    # @reasoning: evaluate quality
    print("  [Evaluator] Assessing output quality...")
    quality = await evaluate_quality(analysis=analysis, summary=summary, context_docs=context_docs)
    print(f"  [Evaluator] Conclusion: {quality.get('conclusion', 'N/A')}")

    # @decision: output format
    format_result = choose_output_format(len(context_docs), f"{len(context_docs)} context docs")
    print(f"  [Evaluator] Output format decision: {format_result['chosen']}")

    # Scores
    waxell.score("analysis_quality", 0.88, comment="auto-scored based on context coverage")
    waxell.score("summary_coherence", 0.91, comment="auto-scored based on analysis-summary alignment")

    return {"summary": summary, "quality": quality}


# ---------------------------------------------------------------------------
# Orchestrator -- coordinates the full LangChain pipeline
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="langchain-orchestrator", workflow_name="langchain-pipeline")
async def run_pipeline(query: str, dry_run: bool = False, waxell_ctx=None):
    """Coordinate the full LangChain pipeline across child agents.

    This is the parent agent. All child agents auto-link to this parent
    via WaxellContext lineage.
    """
    waxell.tag("demo", "langchain")
    waxell.tag("pipeline", "analysis")
    waxell.metadata("framework", "langchain")
    waxell.metadata("mode", "dry-run" if dry_run else "live")
    waxell.metadata("num_agents", 3)

    openai_client = get_openai_client(dry_run=dry_run)

    # Phase 1: @step -- preprocess the query
    print("[Orchestrator] Phase 1: Preprocessing query (@step)...")
    preprocessed = await preprocess_query(query)
    print(f"  Preprocessed: {preprocessed['token_count']} tokens")

    # Phase 2: @decision -- classify the topic (OpenAI)
    print("[Orchestrator] Phase 2: Classifying topic (@decision + OpenAI)...")
    classification = await classify_topic(query=query, openai_client=openai_client)
    chosen = classification.get("chosen", "ethical") if isinstance(classification, dict) else str(classification)
    print(f"  Classification: {chosen}")

    # Phase 3: waxell.decide() -- routing decision
    print("[Orchestrator] Phase 3: Routing decision (waxell.decide())...")
    strategy_map = {"technical": "deep_analysis", "ethical": "balanced_analysis", "policy": "framework_analysis", "general": "broad_analysis"}
    strategy = strategy_map.get(chosen, "balanced_analysis")
    waxell.decide(
        "analysis_strategy",
        chosen=strategy,
        options=["deep_analysis", "balanced_analysis", "framework_analysis", "broad_analysis"],
        reasoning=f"Topic classified as '{chosen}' -- {strategy} optimal for this type",
        confidence=0.87,
    )
    print(f"  Strategy: {strategy}")

    # Phase 4: @retrieval -- retrieve context documents
    print("[Orchestrator] Phase 4: Retrieving context documents (@retrieval)...")
    context_docs = retrieve_context(query=query, documents=_MOCK_CONTEXT_DOCS)
    print(f"  Retrieved {len(context_docs)} context documents")
    for doc in context_docs:
        print(f"    - {doc['title']} (relevance: {doc.get('relevance_score', 'N/A')})")

    # Phase 5: langchain-runner child agent
    print("[Orchestrator] Phase 5: Running LangChain analysis (langchain-runner)...")
    analysis_result = await run_langchain_analysis(
        query=query,
        context_docs=context_docs,
        dry_run=dry_run,
    )
    print(f"  Analysis length: {analysis_result['parsed']['length']} chars")

    # Phase 6: langchain-evaluator child agent
    print("[Orchestrator] Phase 6: Evaluating and summarizing (langchain-evaluator)...")
    eval_result = await run_langchain_evaluation(
        query=query,
        analysis=analysis_result["analysis"],
        context_docs=context_docs,
        dry_run=dry_run,
    )

    return {
        "analysis": analysis_result["analysis"][:500],
        "summary": eval_result["summary"][:500],
        "classification": chosen,
        "strategy": strategy,
        "context_docs_used": len(context_docs),
        "quality": eval_result["quality"],
        "pipeline": "langchain-orchestrator -> langchain-runner -> langchain-evaluator",
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

    print("[LangChain Multi-Agent Pipeline Demo]")
    print(f"  Session:    {session}")
    print(f"  User:       {user_id} ({user_group})")
    print(f"  Query:      {query}")
    print(f"  Mode:       {'dry-run' if args.dry_run else 'LIVE (OpenAI via LangChain)'}")
    print()

    result = await run_pipeline(
        query=query,
        dry_run=args.dry_run,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    )

    print()
    print("=" * 60)
    print("[Result]")
    print(f"  Analysis:     {result['analysis'][:200]}...")
    print(f"  Summary:      {result['summary'][:200]}...")
    print(f"  Classification: {result['classification']}")
    print(f"  Strategy:     {result['strategy']}")
    print(f"  Context docs: {result['context_docs_used']}")
    print(f"  Pipeline:     {result['pipeline']}")
    print()
    print(f"[LangChain Demo] Complete. (2 LLM calls via LangChain, 3 agents, auto-captured)")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
