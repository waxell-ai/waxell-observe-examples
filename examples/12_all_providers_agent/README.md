# All Providers in One Trace

Single WaxellContext trace spanning OpenAI, Anthropic, and LiteLLM/Groq calls for multi-provider comparison.

## Quick Start

```bash
# From the repo root:

# Dry-run (no API keys needed — uses mock clients)
python examples/12_all_providers_agent/all_providers_agent.py --dry-run

# With real API calls
python examples/12_all_providers_agent/all_providers_agent.py

# Custom query
python examples/12_all_providers_agent/all_providers_agent.py --dry-run --query "Your question here"

# Trigger policy enforcement (intentionally exceed limits)
python examples/12_all_providers_agent/all_providers_agent.py --policy-triggers
```

## What to Look For

After running, check your [Waxell dashboard](https://app.waxell.dev) for:

- **Trace view** — see every LLM call, step, and decision in the pipeline
- **Token usage** — input/output tokens per call
- **Latency** — wall-clock time per step and total
- **Policy checks** — any governance decisions that were made

## Environment Variables

Copy `/.env.example` to `/.env`. This example needs:

| Variable | Required | Description |
|----------|----------|-------------|
| `WAXELL_API_KEY` | For observability | Platform API key (`wax_sk_...`) |
| Provider key(s) | For real calls | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LITELLM_API_KEY` |

## Learn More

- [waxell-observe Quickstart](https://waxell.ai/docs/observe/quickstart)
- [API Reference](https://waxell.ai/docs/observe/api/python-sdk)
- [Back to all examples](../../README.md)
