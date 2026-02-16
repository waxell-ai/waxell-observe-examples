# Waxell Observe Examples

Production-ready examples for [waxell-observe](https://github.com/waxell-ai/waxell-observe) — lightweight observability & governance for AI agents.

Every example works out of the box with `--dry-run` (no API keys needed).

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/waxell-ai/waxell-observe-examples.git
cd waxell-observe-examples
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Try it (no API keys needed)
python examples/16_basic_quickstart/quickstart.py --dry-run
```

You'll see mock LLM responses and the full agent pipeline running locally. No keys, no costs.

---

## Connect to Waxell (see your traces in the dashboard)

The examples work offline with `--dry-run`, but to see traces, token usage, and policy checks in the Waxell dashboard you need an API key.

### Step 1: Create an account

Go to [app.waxell.dev](https://app.waxell.dev) and sign up. Free tier is available.

### Step 2: Get your API key

1. Log in to [app.waxell.dev](https://app.waxell.dev)
2. Go to **Settings > API Keys** ([direct link](https://app.waxell.dev/settings/api-keys))
3. Click **Create API Key**
4. Copy the key — it starts with `wax_sk_`

### Step 3: Configure your environment

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```bash
WAXELL_API_KEY=wax_sk_your_actual_key_here
WAXELL_API_URL=https://api.waxell.dev
```

### Step 4: Run an example (for real this time)

```bash
# With a real LLM provider key
OPENAI_API_KEY=sk-... python examples/16_basic_quickstart/quickstart.py

# Or still use mock clients but send traces to Waxell
python examples/01_anthropic_agent/anthropic_agent.py --dry-run
```

Even with `--dry-run`, if `WAXELL_API_KEY` is set the traces are sent to the platform. You can see them at [app.waxell.dev](https://app.waxell.dev).

### Step 5: Check your dashboard

Go to [app.waxell.dev](https://app.waxell.dev). You should see:

- **Traces** — every LLM call, step, and decision in the pipeline
- **Token usage** — input/output tokens per call
- **Latency** — wall-clock time per step and total
- **Policy checks** — any governance decisions that were made

---

## Examples

### Start Here

| # | Example | What It Shows |
|---|---------|--------------|
| 16 | [**Basic Quickstart**](examples/16_basic_quickstart/) | The 5-step pattern every example follows. Start here. |

### Core Patterns

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 01 | [Anthropic Agent](examples/01_anthropic_agent/) | 3-step content analysis (classify, extract, summarize) | Anthropic |
| 02 | [RAG Agent](examples/02_rag_agent/) | Document retrieval, relevance filtering, answer synthesis | OpenAI |
| 03 | [Multi-Agent](examples/03_multi_agent/) | Coordinator + 3 sub-agents with `@waxell_agent` | OpenAI |

### Integrations

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 05 | [Streaming](examples/05_streaming_agent/) | OpenAI + Anthropic streaming side-by-side | OpenAI, Anthropic |
| 11 | [LiteLLM](examples/11_litellm_agent/) | 3 providers through LiteLLM's unified API | LiteLLM |
| 12 | [All Providers](examples/12_all_providers_agent/) | OpenAI + Anthropic + Groq in one trace | All |
| 13 | [Groq & Function Calling](examples/13_groq_agent/) | Groq Llama + OpenAI tool calls | Groq, OpenAI |
| 15 | [LangChain](examples/15_langchain_agent/) | `WaxellLangChainHandler` callback | LangChain |

### Governance & Security

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 04 | [Governance](examples/04_governance_agent/) | Policy checks, `check_policy()`, retry feedback, events | OpenAI |
| 06 | [Prompt Guard](examples/06_prompt_guard_agent/) | PII/credential/injection detection (block/warn/redact) | OpenAI |
| 14 | [Prompt Management](examples/14_prompt_management_agent/) | Prompt versioning, background collector, `get_context()` | OpenAI |

### Advanced Features

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 10 | [Enrichment](examples/10_enrichment_agent/) | `@observe`, scores, tags, metadata | OpenAI |

### Real-World Pipelines

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 07 | [Customer Support](examples/07_support_agent/) | Intent classify, tool calls, decision tree, retry | OpenAI |
| 08 | [Research](examples/08_research_agent/) | Retrieval, web search, reasoning chain | OpenAI |
| 09 | [Code Review](examples/09_code_review_agent/) | Diff parsing, static analysis, review generation | Anthropic |

---

## How It Works

Every example follows the same pattern:

```python
from _common import setup_observe

# 1. Initialize BEFORE importing LLM SDKs (enables auto-instrumentation)
setup_observe()

from waxell_observe import WaxellContext

# 2. Wrap your agent logic
async with WaxellContext(agent_name="my-agent", ...) as ctx:

    # 3. Make LLM calls — they're automatically traced
    response = await client.chat.completions.create(...)

    # 4. Record steps and results
    ctx.record_step("classify", output={"category": "tech"})
    ctx.record_llm_call(model="gpt-4o-mini", tokens_in=50, tokens_out=120, task="classify")
    ctx.set_result({"answer": "..."})

# 5. Flush
waxell_observe.shutdown()
```

`setup_observe()` patches the OpenAI/Anthropic/Groq SDKs so every LLM call is captured automatically. `WaxellContext` groups everything into a single trace.

---

## Dry-Run Mode

Every example supports `--dry-run`:

```bash
python examples/02_rag_agent/rag_agent.py --dry-run
python examples/04_governance_agent/governance_agent.py --dry-run --policy-triggers
python examples/07_support_agent/support_agent.py --dry-run --query "Where is my order?"
```

Mock clients return realistic responses. No API keys, no costs, no network calls.

---

## Environment Variables

Copy `.env.example` to `.env`. Everything is optional if you're just running `--dry-run`.

**To send traces to Waxell:**

| Variable | What to set | Where to get it |
|----------|------------|-----------------|
| `WAXELL_API_KEY` | `wax_sk_...` | [Settings > API Keys](https://app.waxell.dev/settings/api-keys) |
| `WAXELL_API_URL` | `https://api.waxell.dev` | Already set in `.env.example` |

**To use real LLM providers (instead of mock clients):**

| Variable | Provider | Where to get it |
|----------|---------|-----------------|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `GROQ_API_KEY` | Groq | [console.groq.com/keys](https://console.groq.com/keys) |

**Tuning:**

| Variable | Default | What it does |
|----------|---------|-------------|
| `WAXELL_OBSERVE` | `true` | Set `false` to disable all telemetry |
| `WAXELL_DEBUG` | `false` | Print spans to console (useful for debugging) |
| `WAXELL_CAPTURE_CONTENT` | `false` | Include prompt/response text in traces (off by default for privacy) |

---

## Installation

```bash
# Recommended: core + OpenTelemetry
pip install "waxell-observe[otel] @ git+https://github.com/waxell-ai/waxell-observe.git"

# Everything (all provider instrumentors)
pip install "waxell-observe[all] @ git+https://github.com/waxell-ai/waxell-observe.git"
```

---

## Documentation

| Resource | Link |
|----------|------|
| Quickstart | [waxell.ai/docs/observe/quickstart](https://waxell.ai/docs/observe/quickstart) |
| Installation | [waxell.ai/docs/observe/installation](https://waxell.ai/docs/observe/installation) |
| Auto-instrumentation | [waxell.ai/docs/observe/integrations/auto-instrumentation](https://waxell.ai/docs/observe/integrations/auto-instrumentation) |
| Context Manager | [waxell.ai/docs/observe/integrations/context-manager](https://waxell.ai/docs/observe/integrations/context-manager) |
| Governance | [waxell.ai/docs/observe/features/governance](https://waxell.ai/docs/observe/features/governance) |
| API Reference | [waxell.ai/docs/observe/api/python-sdk](https://waxell.ai/docs/observe/api/python-sdk) |
| **For coding agents** | [CLAUDE.md](CLAUDE.md) |
