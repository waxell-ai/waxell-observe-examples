# Groq & Function Calling

Groq auto-instrumentor with Llama 3.3 models plus OpenAI function calling with tools parameter and auto-captured tool_calls.

## Quick Start

```bash
# From the repo root:

# Dry-run (no API keys needed — uses mock clients)
python examples/13_groq_agent/groq_agent.py --dry-run

# With real API calls
python examples/13_groq_agent/groq_agent.py

# Custom query
python examples/13_groq_agent/groq_agent.py --dry-run --query "Your question here"

# Trigger policy enforcement (intentionally exceed limits)
python examples/13_groq_agent/groq_agent.py --policy-triggers
```

## What to Look For

After running, check your [Waxell dashboard](https://app.waxell.dev) for:

- **Trace view** — see every LLM call, step, and decision in the pipeline
- **Token usage** — input/output tokens per call
- **Latency** — wall-clock time per step and total
- **Policy checks** — any governance decisions that were made

## Setup

1. Follow the [Quick Start](../../README.md#quick-start) to install dependencies
2. To see traces in the dashboard, [connect to Waxell](../../README.md#connect-to-waxell-see-your-traces-in-the-dashboard) (get your API key at [app.waxell.dev/settings/api-keys](https://app.waxell.dev/settings/api-keys))
3. For real LLM calls (instead of mock), set your provider key: `GROQ_API_KEY`, `OPENAI_API_KEY`

## Learn More

- [waxell-observe Quickstart](https://waxell.ai/docs/observe/quickstart)
- [API Reference](https://waxell.ai/docs/observe/api/python-sdk)
- [Back to all examples](../../README.md)
