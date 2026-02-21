#!/usr/bin/env python3
"""Anthropic content analysis pipeline demo.

Demonstrates waxell-observe SDK decorators with Anthropic's Claude models
in a multi-agent content analysis pipeline:

  content-analyzer (parent)
  ├── @step: preprocess_query
  ├── @decision: choose_analysis_depth (shallow vs deep vs comprehensive)
  ├── classifier (child agent)
  │   └── Auto-instrumented Anthropic LLM call
  ├── entity-extractor (child agent)
  │   └── Auto-instrumented Anthropic LLM call
  ├── summarizer (child agent)
  │   ├── @reasoning: assess_content_complexity
  │   └── Auto-instrumented Anthropic LLM call
  └── scores, tags, metadata

Usage::

    # Dry-run (no Anthropic API key needed)
    python examples/01_anthropic_agent/anthropic_agent.py --dry-run

    # With real Anthropic calls
    python examples/01_anthropic_agent/anthropic_agent.py

    # Custom query
    python examples/01_anthropic_agent/anthropic_agent.py --query "Explain quantum computing"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

# CRITICAL: init() BEFORE importing anthropic so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import provider clients (auto-instrumentors have patched them)
import waxell_observe as waxell
from waxell_observe import generate_session_id

from _common import (
    get_anthropic_client,
    get_observe_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Analyze the impact of artificial intelligence on modern healthcare systems"


# ---------------------------------------------------------------------------
# Decorated helper functions
# ---------------------------------------------------------------------------

@waxell.step_dec(name="preprocess_query")
def preprocess_query(query: str) -> dict:
    """Normalize and prepare the query for analysis."""
    cleaned = query.strip()
    word_count = len(cleaned.split())
    return {"cleaned_query": cleaned, "word_count": word_count, "needs_context": word_count < 5}


@waxell.decision(name="choose_analysis_depth", options=["shallow", "deep", "comprehensive"])
def choose_analysis_depth(query_info: dict) -> dict:
    """Decide how deep the analysis should go based on query complexity."""
    word_count = query_info.get("word_count", 0)
    if word_count < 10:
        chosen = "shallow"
        reasoning = "Short query — quick classification and summary sufficient"
    elif word_count < 30:
        chosen = "deep"
        reasoning = "Medium query — full entity extraction and detailed summary"
    else:
        chosen = "comprehensive"
        reasoning = "Long query — comprehensive multi-step analysis needed"
    return {"chosen": chosen, "reasoning": reasoning, "confidence": 0.85}


@waxell.reasoning_dec(step="assess_content_complexity")
def assess_content_complexity(classification: str, entities: str) -> dict:
    """Assess the complexity of the content before summarization."""
    entity_count = entities.count(",") + 1
    is_complex = entity_count > 3 or len(classification) > 100
    return {
        "entity_count": entity_count,
        "is_complex": is_complex,
        "recommended_summary_length": "long" if is_complex else "concise",
        "reasoning": f"Found {entity_count} entities; content is {'complex' if is_complex else 'straightforward'}",
    }


# ---------------------------------------------------------------------------
# Child agents
# ---------------------------------------------------------------------------

@waxell.observe(agent_name="classifier", workflow_name="content-classification", capture_io=True)
async def run_classifier(query: str, client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Classify content into categories using Claude."""
    waxell.tag("task", "classification")

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Classify the following text into one of these categories: "
                f"technology, healthcare, finance, education, science, politics. "
                f"Also rate the complexity from 1-5.\n\nText: {query}"
            ),
        }],
    )
    classification = response.content[0].text
    waxell.score("classification_confidence", 0.88, comment="Category assignment confidence")
    return {"classification": classification, "model": response.model}


@waxell.observe(agent_name="entity-extractor", workflow_name="entity-extraction", capture_io=True)
async def run_entity_extractor(query: str, client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Extract key entities from the text using Claude."""
    waxell.tag("task", "entity_extraction")

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Extract the key entities (people, organizations, technologies, "
                f"concepts) from this text. List each entity with its type.\n\n"
                f"Text: {query}"
            ),
        }],
    )
    entities = response.content[0].text
    waxell.score("extraction_completeness", 0.82, comment="Entity extraction coverage")
    return {"entities": entities, "model": response.model}


@waxell.observe(agent_name="summarizer", workflow_name="content-summarization", capture_io=True)
async def run_summarizer(
    query: str, classification: str, entities: str, client,
    *, dry_run: bool = False, waxell_ctx=None,
) -> dict:
    """Generate a summary incorporating classification and entity context."""
    waxell.tag("task", "summarization")

    # Assess complexity before summarizing
    complexity = assess_content_complexity(classification, entities)

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Provide a concise 2-3 sentence summary of the following text, "
                f"highlighting the main argument and key implications.\n\n"
                f"Text: {query}\n\n"
                f"Classification: {classification}\n"
                f"Key entities: {entities}"
            ),
        }],
    )
    summary = response.content[0].text
    waxell.score("summary_quality", 0.90, comment="Summary coherence and completeness")
    return {"summary": summary, "complexity": complexity, "model": response.model}


# ---------------------------------------------------------------------------
# Parent orchestrator
# ---------------------------------------------------------------------------

@waxell.observe(agent_name="content-analyzer", workflow_name="content-analysis", capture_io=True)
async def run_agent(query: str, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Orchestrate the multi-agent content analysis pipeline."""
    waxell.tag("demo", "anthropic")
    waxell.tag("provider", "anthropic")
    waxell.metadata("sdk", "anthropic-python")
    waxell.metadata("pipeline", "classify → extract → summarize")

    client = get_anthropic_client(dry_run=dry_run)

    # Preprocess
    query_info = preprocess_query(query)
    print(f"[Anthropic Demo] Preprocessed: {query_info['word_count']} words")

    # Decide analysis depth
    depth = choose_analysis_depth(query_info)
    print(f"[Anthropic Demo] Analysis depth: {depth['chosen']}")

    # Step 1: Classify
    print("[Anthropic Demo] Step 1/3: Classifying content...")
    class_result = await run_classifier(query, client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print(f"  Classification: {class_result['classification'][:120]}...")

    # Step 2: Extract entities
    print("[Anthropic Demo] Step 2/3: Extracting entities...")
    entity_result = await run_entity_extractor(query, client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print(f"  Entities: {entity_result['entities'][:120]}...")

    # Step 3: Summarize
    print("[Anthropic Demo] Step 3/3: Generating summary...")
    summary_result = await run_summarizer(
        query, class_result["classification"], entity_result["entities"],
        client, dry_run=dry_run, waxell_ctx=waxell_ctx,
    )
    print(f"  Summary: {summary_result['summary'][:120]}...")

    # Overall scores
    waxell.score("pipeline_quality", 0.87, comment="Overall analysis pipeline quality")

    print()
    print(f"[Anthropic Demo] Complete. (3 LLM calls, 3 child agents, provider: Anthropic)")

    return {
        "classification": class_result["classification"],
        "entities": entity_result["entities"],
        "summary": summary_result["summary"],
        "analysis_depth": depth["chosen"],
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

    print("[Anthropic Demo] Starting content analysis pipeline...")
    print(f"[Anthropic Demo] Session: {session}")
    print(f"[Anthropic Demo] End user: {user_id} ({user_group})")
    print(f"[Anthropic Demo] Input: {query[:80]}...")
    print()

    await run_agent(
        query,
        dry_run=args.dry_run,
        waxell_ctx=None,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    import waxell_observe
    asyncio.run(main())
