"""Shared utilities for waxell-observe demo agents.

Provides mock OpenAI/Anthropic clients, streaming mocks, document corpus,
retrieval helpers, and observe initialization for demo scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Simulated end-user pool (opaque IDs, no PII)
# ---------------------------------------------------------------------------

DEMO_USERS = [
    {"user_id": "usr_a1b2c3", "user_group": "enterprise"},
    {"user_id": "usr_d4e5f6", "user_group": "enterprise"},
    {"user_id": "usr_g7h8i9", "user_group": "pro"},
    {"user_id": "usr_j0k1l2", "user_group": "pro"},
    {"user_id": "usr_m3n4o5", "user_group": "free"},
    {"user_id": "usr_p6q7r8", "user_group": "free"},
    {"user_id": "usr_s9t0u1", "user_group": "internal"},
]


def pick_demo_user() -> dict:
    """Return a random demo user identity dict with user_id and user_group."""
    return random.choice(DEMO_USERS)


# ---------------------------------------------------------------------------
# Observe initialization
# ---------------------------------------------------------------------------


def is_observe_disabled() -> bool:
    """Check if WAXELL_OBSERVE is explicitly disabled via environment."""
    return os.environ.get("WAXELL_OBSERVE", "").lower() in ("false", "0", "no")


def get_observe_client():
    """Return a WaxellObserveClient appropriate for the current mode.

    When WAXELL_OBSERVE is disabled, returns a client with forcibly empty
    config so that all HTTP calls are no-ops (the client logs a warning and
    returns ``{}`` when ``is_configured`` is False).
    """
    from waxell_observe.client import WaxellObserveClient
    from waxell_observe.config import ObserveConfig

    if is_observe_disabled():
        # Force-unconfigured client that bypasses ~/.waxell/config and env vars
        client = WaxellObserveClient.__new__(WaxellObserveClient)
        client.config = ObserveConfig(api_url="", api_key="")
        client._http = None
        client._http_warned = False
        return client

    # Normal path: resolve from env / CLI config / global configure()
    return WaxellObserveClient()


def setup_observe(
    debug: bool = True,
    prompt_guard: bool = False,
    prompt_guard_action: str = "block",
    prompt_guard_server: bool = False,
    capture_content: bool = False,
) -> None:
    """Initialize waxell-observe. Must be called BEFORE importing openai.

    Loads .env via dotenv and maps WAX_API_KEY/WAX_API_URL to WAXELL_*
    env vars so the observe SDK picks them up.

    Args:
        debug: Enable console span export for demo visibility.
        prompt_guard: Enable client-side prompt guard (PII/credential/injection).
        prompt_guard_action: Action on violations: "block", "warn", or "redact".
        prompt_guard_server: Also check server-side guard (ML-powered).
        capture_content: Include prompt/response content in traces.
    """
    import logging

    # Ensure waxell_observe warnings are visible in subprocess output
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    # Load .env and map WAX_* → WAXELL_* for the observe SDK
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("WAXELL_API_KEY") or os.environ.get("WAX_API_KEY", "")
    api_url = os.environ.get("WAXELL_API_URL") or os.environ.get("WAX_API_URL", "http://localhost:8001")

    import waxell_observe

    waxell_observe.init(
        api_key=api_key,
        api_url=api_url,
        debug=debug,
        prompt_guard=prompt_guard,
        prompt_guard_action=prompt_guard_action,
        prompt_guard_server=prompt_guard_server,
        capture_content=capture_content,
    )


# ---------------------------------------------------------------------------
# Policy trigger helpers
# ---------------------------------------------------------------------------

# A large prompt (~2000 words) that intentionally exceeds the demo budget
# policy's per_workflow_token_limit of 500 tokens when sent to an LLM.
_SAFETY_PARAGRAPH = (
    "Artificial intelligence safety is a critical field of research and practice "
    "that addresses the challenge of ensuring AI systems behave in alignment with "
    "human values and intentions. As AI capabilities continue to advance at an "
    "unprecedented pace, the importance of developing robust safety frameworks "
    "cannot be overstated. Organizations deploying AI agents in production "
    "environments must consider several key dimensions of safety including "
    "technical robustness, ethical alignment, transparency, accountability, and "
    "operational reliability. Technical robustness encompasses the ability of AI "
    "systems to perform correctly under a wide range of conditions, including "
    "adversarial inputs, distribution shifts, and edge cases that were not "
    "anticipated during training or development. This requires comprehensive "
    "testing strategies that go beyond standard unit and integration tests to "
    "include adversarial red-teaming exercises, stress testing under extreme "
    "conditions, and continuous monitoring of system behavior in production. "
    "Ethical alignment refers to the ongoing challenge of ensuring that AI "
    "systems make decisions that are consistent with human moral values and "
    "societal norms. This is particularly important in high-stakes domains such "
    "as healthcare, criminal justice, financial services, and autonomous systems "
    "where AI decisions can have profound impacts on human lives and well-being. "
    "Transparency in AI systems involves making the decision-making process "
    "understandable and interpretable to relevant stakeholders including end "
    "users, developers, regulators, and affected communities. This includes "
    "providing explanations for individual decisions, documenting model "
    "limitations and known failure modes, and maintaining clear audit trails "
    "that enable post-hoc analysis of system behavior. Accountability frameworks "
    "establish clear lines of responsibility for AI system outcomes, ensuring "
    "that there are designated individuals or teams responsible for monitoring, "
    "maintaining, and intervening in AI system operations when necessary. "
    "Operational reliability encompasses the infrastructure and processes needed "
    "to ensure AI systems operate consistently and predictably in production "
    "environments, including redundancy, failover mechanisms, graceful "
    "degradation under load, and rapid incident response capabilities. "
    "The governance of AI systems requires a multi-layered approach that "
    "combines technical controls such as rate limiting, budget caps, and safety "
    "filters with organizational processes such as review boards, approval "
    "workflows, and regular audits. Policy-based governance enables organizations "
    "to define and enforce rules about how AI agents operate, what resources "
    "they can consume, and what actions they are permitted to take. These "
    "policies should be dynamic and adaptable, allowing organizations to "
    "respond quickly to emerging risks and changing requirements without "
    "requiring code changes or system redeployments. Monitoring and "
    "observability are essential components of any AI safety strategy, "
    "providing real-time visibility into system behavior and enabling "
    "rapid detection of anomalies, policy violations, and potential safety "
    "incidents. Distributed tracing with semantic conventions specific to "
    "AI workloads enables end-to-end visibility across complex multi-agent "
    "execution pipelines, making it possible to understand how individual "
    "decisions contribute to overall system behavior. Cost management is "
    "another critical aspect of AI operations that intersects with safety "
    "concerns. Uncontrolled token consumption can lead to runaway costs "
    "that impact organizational budgets, while also potentially indicating "
    "problematic system behavior such as infinite loops, excessive retries, "
    "or adversarial prompt injection attacks. Budget policies with automatic "
    "enforcement provide a safety net that prevents both financial harm and "
    "the underlying technical issues that cause excessive resource consumption."
)

LARGE_PROMPT: str = " ".join([_SAFETY_PARAGRAPH] * 4)
"""A ~2000-word prompt designed to exceed the demo budget policy's
per_workflow_token_limit of 500 tokens."""


async def simulate_slow_operation(seconds: float = 4.0) -> None:
    """Simulate a slow operation to trigger the operations timeout policy.

    The demo operations policy has ``timeout_seconds: 3``, so sleeping for
    *seconds* (default 4.0) reliably triggers a latency warning.
    """
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Mock OpenAI response objects
# ---------------------------------------------------------------------------


@dataclass
class _MockUsage:
    prompt_tokens: int = 150
    completion_tokens: int = 80
    total_tokens: int = 230


@dataclass
class _MockFunctionCall:
    name: str = ""
    arguments: str = "{}"


@dataclass
class _MockToolCall:
    id: str = "call_mock_001"
    type: str = "function"
    function: _MockFunctionCall = field(default_factory=_MockFunctionCall)


@dataclass
class _MockMessage:
    role: str = "assistant"
    content: str | None = ""
    tool_calls: list[_MockToolCall] | None = None


@dataclass
class _MockToolMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_MockToolCall] | None = None


@dataclass
class _MockChoice:
    index: int = 0
    message: _MockMessage = field(default_factory=_MockMessage)
    finish_reason: str = "stop"


# Tool response patterns: keyword -> (tool_name, arguments_json)
_TOOL_RESPONSE_PATTERNS: dict[str, tuple[str, str]] = {
    "search": ("web_search", '{"query": "AI safety best practices", "results": ["doc1", "doc2"]}'),
    "calculate": ("calculator", '{"operation": "average", "values": [85, 92, 78]}'),
    "look up": ("order_lookup", '{"order_id": "ORD-12345", "status": "shipped"}'),
    "inventory": ("inventory_check", '{"product_id": "PROD-789", "in_stock": true}'),
    "lint": ("linter", '{"warnings": 2, "errors": 0}'),
    "test": ("test_runner", '{"passed": 47, "failed": 1}'),
    "scan": ("security_scanner", '{"vulnerabilities": [{"severity": "medium", "type": "sql_injection"}]}'),
    "parse": ("diff_parser", '{"files_changed": 3, "lines_added": 42}'),
}


class MockChatCompletion:
    """Mimics the OpenAI ChatCompletion response structure."""

    def __init__(
        self,
        content: str | None,
        model: str = "gpt-4o-mini",
        prompt_tokens: int = 150,
        completion_tokens: int = 80,
        tool_calls: list[_MockToolCall] | None = None,
    ):
        self.id = "chatcmpl-mock-demo-00001"
        self.object = "chat.completion"
        self.model = model
        finish_reason = "tool_calls" if tool_calls else "stop"
        self.choices = [
            _MockChoice(
                message=_MockMessage(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
        self.usage = _MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# ---------------------------------------------------------------------------
# Contextual response logic
# ---------------------------------------------------------------------------

_RESPONSE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"retrieve|search", re.IGNORECASE),
        (
            "Found 3 relevant documents covering AI safety guidelines, "
            "deployment best practices, and cost optimization strategies."
        ),
    ),
    (
        re.compile(r"synthesize|answer", re.IGNORECASE),
        (
            "Based on the provided documents, the key recommendations are: "
            "1) Implement safety guardrails at every stage of the pipeline. "
            "2) Use staged deployments with canary releases and rollback plans. "
            "3) Monitor token usage and set budget alerts to control costs."
        ),
    ),
    (
        re.compile(r"plan|break down", re.IGNORECASE),
        (
            "1. Research AI safety frameworks\n"
            "2. Analyze deployment patterns\n"
            "3. Evaluate cost optimization strategies"
        ),
    ),
    (
        re.compile(r"relevance|evaluate|filter", re.IGNORECASE),
        (
            "Documents 1 and 3 are most relevant. Document 1 directly "
            "addresses safety guidelines. Document 3 covers monitoring "
            "which is essential for production deployment."
        ),
    ),
]

_DEFAULT_RESPONSE = (
    "The analysis indicates several important considerations for AI system "
    "design including robustness, scalability, and continuous monitoring."
)


def _contextual_response(messages: list[dict[str, Any]]) -> str:
    """Return a mock response based on the last user message content."""
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = str(msg.get("content", ""))
            break

    for pattern, response in _RESPONSE_PATTERNS:
        if pattern.search(last_user):
            return response

    return _DEFAULT_RESPONSE


# ---------------------------------------------------------------------------
# Mock async OpenAI client
# ---------------------------------------------------------------------------


class _MockCompletions:
    """Mock for client.chat.completions.

    When ``tools`` is present in kwargs and no ``role="tool"`` messages
    have been sent yet, returns a tool_calls response.  On the follow-up
    call (with tool results), returns a normal text response.
    """

    async def create(self, **kwargs: Any) -> MockChatCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gpt-4o-mini")
        tools_param = kwargs.get("tools")

        # If tools are defined, decide whether to invoke one
        if tools_param:
            has_tool_result = any(
                m.get("role") == "tool" for m in messages
            )
            if not has_tool_result:
                # First call with tools -- return a tool_calls response
                last_user = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user = str(msg.get("content", "")).lower()
                        break

                tool_name = "web_search"
                tool_args = '{"query": "general search"}'
                for keyword, (tname, targs) in _TOOL_RESPONSE_PATTERNS.items():
                    if keyword in last_user:
                        tool_name, tool_args = tname, targs
                        break

                call_id = f"call_mock_{random.randint(100, 999)}"
                tc = _MockToolCall(
                    id=call_id,
                    function=_MockFunctionCall(name=tool_name, arguments=tool_args),
                )
                return MockChatCompletion(
                    content=None,
                    model=model,
                    prompt_tokens=180,
                    completion_tokens=25,
                    tool_calls=[tc],
                )

        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model)


class _MockChat:
    """Mock for client.chat."""

    def __init__(self) -> None:
        self.completions = _MockCompletions()


class _MockEmbeddingData:
    """Mock embedding data point."""

    def __init__(self, embedding: list[float], index: int = 0) -> None:
        self.embedding = embedding
        self.index = index
        self.object = "embedding"


class _MockEmbeddingUsage:
    """Mock embedding usage."""

    def __init__(self, total_tokens: int = 0) -> None:
        self.total_tokens = total_tokens
        self.prompt_tokens = total_tokens


class _MockEmbeddingResponse:
    """Mock embedding response."""

    def __init__(self, data: list, usage: _MockEmbeddingUsage) -> None:
        self.data = data
        self.usage = usage
        self.model = "text-embedding-3-small"
        self.object = "list"


class _MockAsyncEmbeddings:
    """Mock for client.embeddings with async create()."""

    async def create(self, **kwargs: Any) -> _MockEmbeddingResponse:
        model = kwargs.get("model", "text-embedding-3-small")
        input_data = kwargs.get("input", "")
        if isinstance(input_data, str):
            inputs = [input_data]
        elif isinstance(input_data, list):
            inputs = input_data
        else:
            inputs = [str(input_data)]

        dim = 1536
        data = [
            _MockEmbeddingData(
                embedding=[0.01 * (i + 1)] * dim, index=i,
            )
            for i in range(len(inputs))
        ]
        tokens = sum(len(t.split()) * 2 for t in inputs)
        response = _MockEmbeddingResponse(data=data, usage=_MockEmbeddingUsage(total_tokens=tokens))
        response.model = model
        return response


class MockAsyncOpenAI:
    """A mock async OpenAI client that returns contextual responses.

    Usage::

        client = MockAsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "retrieve docs about safety"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockChat()
        self.embeddings = _MockAsyncEmbeddings()


# ---------------------------------------------------------------------------
# Mock sync OpenAI client (for sync context manager demos)
# ---------------------------------------------------------------------------


class _MockSyncCompletions:
    """Mock for client.chat.completions with synchronous create()."""

    def create(self, **kwargs: Any) -> MockChatCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gpt-4o-mini")
        tools_param = kwargs.get("tools")

        if tools_param:
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if not has_tool_result:
                last_user = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user = str(msg.get("content", "")).lower()
                        break

                tool_name = "web_search"
                tool_args = '{"query": "general search"}'
                for keyword, (tname, targs) in _TOOL_RESPONSE_PATTERNS.items():
                    if keyword in last_user:
                        tool_name, tool_args = tname, targs
                        break

                call_id = f"call_mock_{random.randint(100, 999)}"
                tc = _MockToolCall(
                    id=call_id,
                    function=_MockFunctionCall(name=tool_name, arguments=tool_args),
                )
                return MockChatCompletion(
                    content=None,
                    model=model,
                    prompt_tokens=180,
                    completion_tokens=25,
                    tool_calls=[tc],
                )

        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model)


class _MockSyncChat:
    """Mock for client.chat (sync)."""

    def __init__(self) -> None:
        self.completions = _MockSyncCompletions()


class MockSyncOpenAI:
    """A mock synchronous OpenAI client that returns contextual responses.

    Usage::

        client = MockSyncOpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "classify this ticket"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockSyncChat()


# ---------------------------------------------------------------------------
# Mock failing clients (for retry / fallback demos)
# ---------------------------------------------------------------------------


class _MockFailingCompletions:
    """Completions mock that fails on the first N calls."""

    def __init__(self, fail_count: int = 1, error_class: type = Exception,
                 error_message: str = "429 Too Many Requests"):
        self._fail_count = fail_count
        self._call_count = 0
        self._error_class = error_class
        self._error_message = error_message

    async def create(self, **kwargs: Any) -> MockChatCompletion:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._error_class(self._error_message)
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gpt-4o-mini")
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model)


class _MockFailingChat:
    def __init__(self, fail_count: int = 1, error_class: type = Exception,
                 error_message: str = "429 Too Many Requests") -> None:
        self.completions = _MockFailingCompletions(
            fail_count=fail_count,
            error_class=error_class,
            error_message=error_message,
        )


class MockFailingOpenAI:
    """A mock async OpenAI client that raises on the first N calls, then succeeds.

    Useful for demonstrating retry and fallback patterns.

    Usage::

        client = MockFailingOpenAI(fail_count=1)
        # First call raises Exception("429 Too Many Requests")
        # Second call returns a normal MockChatCompletion
    """

    def __init__(self, fail_count: int = 1, error_class: type = Exception,
                 error_message: str = "429 Too Many Requests", **kwargs: Any) -> None:
        self.chat = _MockFailingChat(
            fail_count=fail_count,
            error_class=error_class,
            error_message=error_message,
        )


class _MockFailingAnthropicMessages:
    """Anthropic messages mock that fails on the first N calls."""

    def __init__(self, fail_count: int = 1, error_class: type = Exception,
                 error_message: str = "Request timed out"):
        self._fail_count = fail_count
        self._call_count = 0
        self._error_class = error_class
        self._error_message = error_message

    async def create(self, **kwargs: Any) -> "MockAnthropicMessage":
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._error_class(self._error_message)
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "claude-sonnet-4-5-20250929")
        content = _contextual_response(messages)
        return MockAnthropicMessage(content=content, model=model)


class MockFailingAnthropic:
    """A mock async Anthropic client that raises on the first N calls, then succeeds.

    Useful for demonstrating retry and fallback patterns.
    """

    def __init__(self, fail_count: int = 1, error_class: type = Exception,
                 error_message: str = "Request timed out", **kwargs: Any) -> None:
        self.messages = _MockFailingAnthropicMessages(
            fail_count=fail_count,
            error_class=error_class,
            error_message=error_message,
        )


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mock Anthropic response objects
# ---------------------------------------------------------------------------


@dataclass
class _MockContentBlock:
    text: str = ""
    type: str = "text"


@dataclass
class _MockAnthropicToolUseBlock:
    """Mimics an Anthropic tool_use content block."""
    type: str = "tool_use"
    id: str = "toolu_mock_001"
    name: str = "verify_answer"
    input: dict = field(default_factory=dict)


@dataclass
class _MockAnthropicUsage:
    input_tokens: int = 150
    output_tokens: int = 80


class MockAnthropicMessage:
    """Mimics the Anthropic Message response structure."""

    def __init__(
        self,
        content: str,
        model: str = "claude-sonnet-4-5-20250929",
        input_tokens: int = 150,
        output_tokens: int = 80,
    ):
        self.id = "msg-mock-demo-00001"
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = [_MockContentBlock(text=content)]
        self.usage = _MockAnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.stop_reason = "end_turn"
        self.stop_sequence = None


class _MockAnthropicMessages:
    """Mock for client.messages.

    When ``tools`` is present in kwargs and no tool_result messages have
    been sent yet, returns a tool_use response.  On the follow-up call
    (with tool results), returns a normal text response.
    """

    async def create(self, **kwargs: Any) -> MockAnthropicMessage:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "claude-sonnet-4-5-20250929")
        tools_param = kwargs.get("tools")

        if tools_param:
            # Check if we already have a tool result in the messages
            has_tool_result = any(
                m.get("role") == "tool" for m in messages if isinstance(m, dict)
            )
            if not has_tool_result:
                # First call with tools — return a tool_use response
                tool_name = "verify_answer"
                if tools_param and isinstance(tools_param[0], dict):
                    tool_name = tools_param[0].get("name", "verify_answer")

                block = _MockAnthropicToolUseBlock(
                    id=f"toolu_mock_{random.randint(100, 999)}",
                    name=tool_name,
                    input={"query": "verification check", "confidence": 0.95},
                )

                msg = MockAnthropicMessage.__new__(MockAnthropicMessage)
                msg.id = "msg-mock-tool-use"
                msg.type = "message"
                msg.role = "assistant"
                msg.model = model
                msg.content = [block]
                msg.usage = _MockAnthropicUsage(input_tokens=200, output_tokens=50)
                msg.stop_reason = "tool_use"
                msg.stop_sequence = None
                return msg

        content = _contextual_response(messages)
        return MockAnthropicMessage(content=content, model=model)


class MockAsyncAnthropic:
    """A mock async Anthropic client that returns contextual responses.

    Usage::

        client = MockAsyncAnthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.content[0].text)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.messages = _MockAnthropicMessages()


# ---------------------------------------------------------------------------
# Mock streaming objects
# ---------------------------------------------------------------------------


@dataclass
class _MockStreamDelta:
    content: str | None = None
    role: str | None = None


@dataclass
class _MockStreamChoice:
    index: int = 0
    delta: _MockStreamDelta = field(default_factory=_MockStreamDelta)
    finish_reason: str | None = None


class _MockStreamChunk:
    """Mimics an OpenAI ChatCompletionChunk."""

    def __init__(self, content: str | None = None, finish_reason: str | None = None,
                 usage: _MockUsage | None = None):
        self.choices = [_MockStreamChoice(
            delta=_MockStreamDelta(content=content),
            finish_reason=finish_reason,
        )]
        self.usage = usage


class MockOpenAIStream:
    """Async iterator that mimics an OpenAI streaming response."""

    def __init__(self, content: str, model: str = "gpt-4o-mini"):
        self._words = content.split()
        self._model = model
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index < len(self._words):
            word = self._words[self._index]
            self._index += 1
            prefix = "" if self._index == 1 else " "
            await asyncio.sleep(0.02)
            return _MockStreamChunk(content=prefix + word)
        elif self._index == len(self._words):
            self._index += 1
            return _MockStreamChunk(
                finish_reason="stop",
                usage=_MockUsage(prompt_tokens=150, completion_tokens=80, total_tokens=230),
            )
        raise StopAsyncIteration


@dataclass
class _MockAnthropicEvent:
    type: str
    message: Any = None
    index: int = 0
    content_block: Any = None
    delta: Any = None
    usage: Any = None


@dataclass
class _MockAnthropicDelta:
    type: str = "text_delta"
    text: str = ""
    stop_reason: str | None = None


@dataclass
class _MockAnthropicMessageStart:
    usage: _MockAnthropicUsage = field(default_factory=lambda: _MockAnthropicUsage(input_tokens=150, output_tokens=0))


class MockAnthropicStream:
    """Async iterator that mimics an Anthropic streaming response."""

    def __init__(self, content: str, model: str = "claude-sonnet-4-5-20250929"):
        self._words = content.split()
        self._model = model
        self._phase = "start"
        self._word_index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._phase == "start":
            self._phase = "content"
            return _MockAnthropicEvent(
                type="message_start",
                message=_MockAnthropicMessageStart(),
            )
        elif self._phase == "content":
            if self._word_index < len(self._words):
                word = self._words[self._word_index]
                self._word_index += 1
                prefix = "" if self._word_index == 1 else " "
                await asyncio.sleep(0.02)
                return _MockAnthropicEvent(
                    type="content_block_delta",
                    delta=_MockAnthropicDelta(text=prefix + word),
                )
            self._phase = "delta"
            return _MockAnthropicEvent(
                type="message_delta",
                delta=_MockAnthropicDelta(stop_reason="end_turn"),
                usage=_MockAnthropicUsage(input_tokens=0, output_tokens=80),
            )
        elif self._phase == "delta":
            self._phase = "done"
            return _MockAnthropicEvent(type="message_stop")
        raise StopAsyncIteration


class _MockStreamingCompletions:
    """Mock for client.chat.completions that supports stream=True."""

    async def create(self, **kwargs: Any):
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gpt-4o-mini")
        content = _contextual_response(messages)
        if kwargs.get("stream"):
            return MockOpenAIStream(content=content, model=model)
        return MockChatCompletion(content=content, model=model)


class _MockStreamingChat:
    def __init__(self) -> None:
        self.completions = _MockStreamingCompletions()


class MockStreamingOpenAI:
    """Mock OpenAI client that supports both streaming and non-streaming."""

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockStreamingChat()


class _MockStreamingAnthropicMessages:
    """Mock for client.messages that supports stream=True."""

    async def create(self, **kwargs: Any):
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "claude-sonnet-4-5-20250929")
        content = _contextual_response(messages)
        if kwargs.get("stream"):
            return MockAnthropicStream(content=content, model=model)
        return MockAnthropicMessage(content=content, model=model)


class MockStreamingAnthropic:
    """Mock Anthropic client that supports both streaming and non-streaming."""

    def __init__(self, **kwargs: Any) -> None:
        self.messages = _MockStreamingAnthropicMessages()


# ---------------------------------------------------------------------------
# Mock Groq response objects (Groq uses OpenAI-compatible response format)
# ---------------------------------------------------------------------------


class _MockGroqCompletions:
    """Mock for groq_client.chat.completions."""

    async def create(self, **kwargs: Any) -> MockChatCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "llama-3.3-70b-versatile")
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model)


class _MockGroqChat:
    def __init__(self) -> None:
        self.completions = _MockGroqCompletions()


class MockAsyncGroq:
    """A mock async Groq client that returns contextual responses.

    Groq uses the same response structure as OpenAI, so we reuse
    MockChatCompletion.

    Usage::

        client = MockAsyncGroq()
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockGroqChat()


# ---------------------------------------------------------------------------
# Mock Mistral response objects
# ---------------------------------------------------------------------------


@dataclass
class _MockMistralUsage:
    prompt_tokens: int = 150
    completion_tokens: int = 80
    total_tokens: int = 230


@dataclass
class _MockMistralChoice:
    index: int = 0
    message: _MockMessage = field(default_factory=_MockMessage)
    finish_reason: str = "stop"


class MockMistralCompletion:
    """Mimics the Mistral ChatCompletionResponse."""

    def __init__(self, content: str, model: str = "mistral-small-latest",
                 prompt_tokens: int = 150, completion_tokens: int = 80):
        self.id = "chatcmpl-mock-mistral-00001"
        self.object = "chat.completion"
        self.model = model
        self.choices = [_MockMistralChoice(
            message=_MockMessage(content=content),
        )]
        self.usage = _MockMistralUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class _MockMistralChat:
    async def complete_async(self, **kwargs: Any) -> MockMistralCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "mistral-small-latest")
        content = _contextual_response(messages)
        return MockMistralCompletion(content=content, model=model)

    def complete(self, **kwargs: Any) -> MockMistralCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "mistral-small-latest")
        content = _contextual_response(messages)
        return MockMistralCompletion(content=content, model=model)


class MockAsyncMistral:
    """Mock Mistral client."""

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockMistralChat()


# ---------------------------------------------------------------------------
# Mock Cohere response objects
# ---------------------------------------------------------------------------


@dataclass
class _MockCohereTokens:
    input_tokens: int = 150
    output_tokens: int = 80


@dataclass
class _MockCohereUsage:
    tokens: _MockCohereTokens = field(default_factory=_MockCohereTokens)
    billed_units: _MockCohereTokens = field(default_factory=_MockCohereTokens)


@dataclass
class _MockCohereContentItem:
    type: str = "text"
    text: str = ""


class MockCohereResponse:
    """Mimics the Cohere V2 chat response."""

    def __init__(self, content: str, model: str = "command-r"):
        self.id = "chatcmpl-mock-cohere-00001"
        self.message = type("Msg", (), {
            "role": "assistant",
            "content": [_MockCohereContentItem(text=content)],
        })()
        self.model = model
        self.usage = _MockCohereUsage(
            tokens=_MockCohereTokens(input_tokens=150, output_tokens=80),
        )
        self.finish_reason = "COMPLETE"


class _MockCohereV2Chat:
    async def chat(self, **kwargs: Any) -> MockCohereResponse:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "command-r")
        content = _contextual_response(messages)
        return MockCohereResponse(content=content, model=model)


class MockAsyncCohere:
    """Mock Cohere V2 client."""

    def __init__(self, **kwargs: Any) -> None:
        self.v2 = _MockCohereV2Chat()


# ---------------------------------------------------------------------------
# Mock Gemini response objects
# ---------------------------------------------------------------------------


@dataclass
class _MockGeminiUsageMetadata:
    prompt_token_count: int = 150
    candidates_token_count: int = 80
    total_token_count: int = 230


class MockGeminiResponse:
    """Mimics the Google GenerativeAI response."""

    def __init__(self, content: str):
        self.text = content
        self.usage_metadata = _MockGeminiUsageMetadata()
        self.candidates = []


class MockGeminiModel:
    """Mock for google.generativeai.GenerativeModel."""

    def __init__(self, model_name: str = "gemini-2.0-flash", **kwargs: Any):
        self.model_name = model_name

    async def generate_content_async(self, contents, **kwargs: Any) -> MockGeminiResponse:
        text = str(contents) if isinstance(contents, str) else str(contents)
        content = _contextual_response([{"role": "user", "content": text}])
        return MockGeminiResponse(content=content)

    def generate_content(self, contents, **kwargs: Any) -> MockGeminiResponse:
        text = str(contents) if isinstance(contents, str) else str(contents)
        content = _contextual_response([{"role": "user", "content": text}])
        return MockGeminiResponse(content=content)


# ---------------------------------------------------------------------------
# Mock Bedrock response objects
# ---------------------------------------------------------------------------


class MockBedrockResponse:
    """Mimics the Bedrock Converse response."""

    def __init__(self, content: str, model: str = "amazon.nova-lite-v1:0"):
        self._data = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": content}],
                }
            },
            "usage": {
                "inputTokens": 150,
                "outputTokens": 80,
                "totalTokens": 230,
            },
            "stopReason": "end_turn",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockBedrockClient:
    """Mock for boto3 bedrock-runtime client."""

    def __init__(self, **kwargs: Any):
        pass

    def converse(self, **kwargs: Any) -> dict:
        messages = kwargs.get("messages", [])
        model = kwargs.get("modelId", "amazon.nova-lite-v1:0")
        content = _contextual_response(messages)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": content}],
                }
            },
            "usage": {
                "inputTokens": 150,
                "outputTokens": 80,
                "totalTokens": 230,
            },
            "stopReason": "end_turn",
        }


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def get_openai_client(dry_run: bool = False) -> Any:
    """Return a mock or real OpenAI async client.

    Args:
        dry_run: Force mock client regardless of API key.

    Returns mock client if *dry_run* is True or ``OPENAI_API_KEY`` is not set.
    Otherwise returns ``AsyncOpenAI()``.
    """
    if dry_run or not os.environ.get("OPENAI_API_KEY"):
        mode = "dry-run" if dry_run else "no OPENAI_API_KEY"
        print(f"[Common] Using mock OpenAI client ({mode})")
        return MockAsyncOpenAI()

    # Lazy import -- openai may not be installed in all envs
    from openai import AsyncOpenAI

    print("[Common] Using real OpenAI client")
    return AsyncOpenAI()


def get_sync_openai_client(dry_run: bool = False) -> Any:
    """Return a mock or real synchronous OpenAI client.

    Args:
        dry_run: Force mock client regardless of API key.

    Returns mock client if *dry_run* is True or ``OPENAI_API_KEY`` is not set.
    Otherwise returns ``OpenAI()`` (sync client).
    """
    if dry_run or not os.environ.get("OPENAI_API_KEY"):
        mode = "dry-run" if dry_run else "no OPENAI_API_KEY"
        print(f"[Common] Using mock sync OpenAI client ({mode})")
        return MockSyncOpenAI()

    # Lazy import -- openai may not be installed in all envs
    from openai import OpenAI

    print("[Common] Using real sync OpenAI client")
    return OpenAI()


def get_anthropic_client(dry_run: bool = False) -> Any:
    """Return a mock or real Anthropic async client.

    Returns mock client if *dry_run* is True or ``ANTHROPIC_API_KEY`` is not set.
    """
    if dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        mode = "dry-run" if dry_run else "no ANTHROPIC_API_KEY"
        print(f"[Common] Using mock Anthropic client ({mode})")
        return MockAsyncAnthropic()

    from anthropic import AsyncAnthropic

    print("[Common] Using real Anthropic client")
    return AsyncAnthropic()


def get_streaming_openai_client(dry_run: bool = False) -> Any:
    """Return a mock or real OpenAI client with streaming support."""
    if dry_run or not os.environ.get("OPENAI_API_KEY"):
        mode = "dry-run" if dry_run else "no OPENAI_API_KEY"
        print(f"[Common] Using mock streaming OpenAI client ({mode})")
        return MockStreamingOpenAI()

    from openai import AsyncOpenAI

    print("[Common] Using real OpenAI client (streaming)")
    return AsyncOpenAI()


def get_streaming_anthropic_client(dry_run: bool = False) -> Any:
    """Return a mock or real Anthropic client with streaming support."""
    if dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        mode = "dry-run" if dry_run else "no ANTHROPIC_API_KEY"
        print(f"[Common] Using mock streaming Anthropic client ({mode})")
        return MockStreamingAnthropic()

    from anthropic import AsyncAnthropic

    print("[Common] Using real Anthropic client (streaming)")
    return AsyncAnthropic()


def get_groq_client(dry_run: bool = False) -> Any:
    """Return a mock or real Groq async client.

    Returns mock client if *dry_run* is True or ``GROQ_API_KEY`` is not set.
    """
    if dry_run or not os.environ.get("GROQ_API_KEY"):
        mode = "dry-run" if dry_run else "no GROQ_API_KEY"
        print(f"[Common] Using mock Groq client ({mode})")
        return MockAsyncGroq()

    from groq import AsyncGroq

    print("[Common] Using real Groq client")
    return AsyncGroq()


def get_mistral_client(dry_run: bool = False) -> Any:
    """Return a mock or real Mistral client."""
    if dry_run or not os.environ.get("MISTRAL_API_KEY"):
        mode = "dry-run" if dry_run else "no MISTRAL_API_KEY"
        print(f"[Common] Using mock Mistral client ({mode})")
        return MockAsyncMistral()

    from mistralai import Mistral

    print("[Common] Using real Mistral client")
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def get_cohere_client(dry_run: bool = False) -> Any:
    """Return a mock or real Cohere client."""
    if dry_run or not os.environ.get("COHERE_API_KEY"):
        mode = "dry-run" if dry_run else "no COHERE_API_KEY"
        print(f"[Common] Using mock Cohere client ({mode})")
        return MockAsyncCohere()

    import cohere

    print("[Common] Using real Cohere client")
    return cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])


def get_gemini_model(dry_run: bool = False, model_name: str = "gemini-2.0-flash") -> Any:
    """Return a mock or real Gemini GenerativeModel."""
    if dry_run or not os.environ.get("GOOGLE_API_KEY"):
        mode = "dry-run" if dry_run else "no GOOGLE_API_KEY"
        print(f"[Common] Using mock Gemini model ({mode})")
        return MockGeminiModel(model_name=model_name)

    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print(f"[Common] Using real Gemini model ({model_name})")
    return genai.GenerativeModel(model_name)


def get_bedrock_client(dry_run: bool = False) -> Any:
    """Return a mock or real Bedrock runtime client."""
    if dry_run or not os.environ.get("AWS_ACCESS_KEY_ID"):
        mode = "dry-run" if dry_run else "no AWS credentials"
        print(f"[Common] Using mock Bedrock client ({mode})")
        return MockBedrockClient()

    import boto3

    print("[Common] Using real Bedrock client")
    return boto3.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# Mock Ollama response objects
# ---------------------------------------------------------------------------


class MockOllamaResponse:
    """Mimics the Ollama chat/generate response dict-like object."""

    def __init__(self, content: str, model: str = "llama3.2"):
        self._data = {
            "model": model,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 1500000000,
            "prompt_eval_count": 150,
            "eval_count": 80,
        }

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data


class MockOllamaClient:
    """Mock for ollama.Client (sync)."""

    def __init__(self, host: str = "http://localhost:11434", **kwargs: Any):
        self.host = host

    def chat(self, model: str = "llama3.2", messages: list | None = None, **kwargs: Any):
        content = _contextual_response(messages or [])
        return MockOllamaResponse(content=content, model=model)

    def generate(self, model: str = "llama3.2", prompt: str = "", **kwargs: Any):
        content = _contextual_response([{"role": "user", "content": prompt}])
        return {
            "model": model,
            "response": content,
            "done": True,
            "total_duration": 1500000000,
            "prompt_eval_count": 150,
            "eval_count": 80,
        }


class MockAsyncOllamaClient:
    """Mock for ollama.AsyncClient."""

    def __init__(self, host: str = "http://localhost:11434", **kwargs: Any):
        self.host = host

    async def chat(self, model: str = "llama3.2", messages: list | None = None, **kwargs: Any):
        content = _contextual_response(messages or [])
        return MockOllamaResponse(content=content, model=model)

    async def generate(self, model: str = "llama3.2", prompt: str = "", **kwargs: Any):
        content = _contextual_response([{"role": "user", "content": prompt}])
        return {
            "model": model,
            "response": content,
            "done": True,
            "total_duration": 1500000000,
            "prompt_eval_count": 150,
            "eval_count": 80,
        }


def get_ollama_client(dry_run: bool = False, sync: bool = False) -> Any:
    """Return a mock or real Ollama client."""
    if dry_run or not os.environ.get("OLLAMA_HOST", ""):
        mode = "dry-run" if dry_run else "mock (no OLLAMA_HOST)"
        print(f"[Common] Using mock Ollama client ({mode})")
        return MockOllamaClient() if sync else MockAsyncOllamaClient()

    import ollama
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"[Common] Using real Ollama client ({host})")
    return ollama.Client(host=host) if sync else ollama.AsyncClient(host=host)


# ---------------------------------------------------------------------------
# Mock Together response objects (OpenAI-compatible)
# ---------------------------------------------------------------------------


class _MockTogetherCompletions:
    """Mock for together_client.chat.completions."""

    async def create(self, **kwargs: Any) -> MockChatCompletion:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model)


class _MockTogetherChat:
    def __init__(self) -> None:
        self.completions = _MockTogetherCompletions()


class MockAsyncTogether:
    """Mock for together.AsyncTogether (OpenAI-compatible)."""

    def __init__(self, **kwargs: Any) -> None:
        self.chat = _MockTogetherChat()


def get_together_client(dry_run: bool = False) -> Any:
    """Return a mock or real Together AI async client."""
    if dry_run or not os.environ.get("TOGETHER_API_KEY"):
        mode = "dry-run" if dry_run else "no TOGETHER_API_KEY"
        print(f"[Common] Using mock Together client ({mode})")
        return MockAsyncTogether()

    from together import AsyncTogether
    print("[Common] Using real Together client")
    return AsyncTogether()


# ---------------------------------------------------------------------------
# Mock HuggingFace response objects
# ---------------------------------------------------------------------------


class MockHuggingFaceClient:
    """Mock for huggingface_hub.InferenceClient."""

    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs: Any):
        self.model = model

    def text_generation(self, prompt: str, **kwargs: Any) -> str:
        content = _contextual_response([{"role": "user", "content": prompt}])
        return content

    def chat_completion(self, messages: list | None = None, **kwargs: Any):
        content = _contextual_response(messages or [])
        return MockChatCompletion(content=content, model=self.model)


def get_huggingface_client(dry_run: bool = False, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> Any:
    """Return a mock or real HuggingFace InferenceClient."""
    if dry_run or not os.environ.get("HF_TOKEN"):
        mode = "dry-run" if dry_run else "no HF_TOKEN"
        print(f"[Common] Using mock HuggingFace client ({mode})")
        return MockHuggingFaceClient(model=model)

    from huggingface_hub import InferenceClient
    print(f"[Common] Using real HuggingFace client ({model})")
    return InferenceClient(model=model, token=os.environ["HF_TOKEN"])


# ---------------------------------------------------------------------------
# Mock Milvus objects
# ---------------------------------------------------------------------------


class MockMilvusSearchResult:
    """Single search hit."""

    def __init__(self, id: int, distance: float, entity: dict):
        self.id = id
        self.distance = distance
        self.entity = entity

    def __getitem__(self, key):
        if key == "id":
            return self.id
        if key == "distance":
            return self.distance
        return self.entity.get(key)


class MockMilvusCollection:
    """Mock for pymilvus.Collection."""

    def __init__(self, name: str = "demo_collection", **kwargs: Any):
        self.name = name
        self._data: list[dict] = []

    def insert(self, data: list, **kwargs: Any) -> dict:
        """Mock insert that stores data."""
        count = len(data[0]) if data and isinstance(data[0], list) else len(data)
        return {"insert_count": count, "ids": list(range(count))}

    def search(self, data: list, anns_field: str = "embedding", param: dict | None = None,
               limit: int = 5, output_fields: list | None = None, **kwargs: Any) -> list[list]:
        """Mock search returning fake results."""
        results = [
            MockMilvusSearchResult(i, 0.95 - i * 0.1, {"text": f"Document {i} about AI safety", "source": f"doc-{i}"})
            for i in range(min(limit, 3))
        ]
        return [results]

    def query(self, expr: str = "", output_fields: list | None = None, **kwargs: Any) -> list[dict]:
        """Mock query."""
        return [
            {"id": 0, "text": "AI safety best practices document", "source": "doc-0"},
            {"id": 1, "text": "Model deployment guidelines", "source": "doc-1"},
        ]

    def delete(self, expr: str = "", **kwargs: Any) -> dict:
        return {"delete_count": 1}

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


def get_milvus_collection(dry_run: bool = False, name: str = "demo_collection") -> Any:
    """Return a mock or real Milvus Collection."""
    if dry_run or not os.environ.get("MILVUS_URI"):
        mode = "dry-run" if dry_run else "mock (no MILVUS_URI)"
        print(f"[Common] Using mock Milvus collection ({mode})")
        return MockMilvusCollection(name=name)

    from pymilvus import Collection
    print(f"[Common] Using real Milvus collection ({name})")
    return Collection(name=name)


# ---------------------------------------------------------------------------
# Mock MongoDB objects (for vector search)
# ---------------------------------------------------------------------------


class MockMongoCollection:
    """Mock for pymongo.collection.Collection with $vectorSearch aggregate."""

    def __init__(self, name: str = "documents", **kwargs: Any):
        self.name = name

    def aggregate(self, pipeline: list, **kwargs: Any) -> list[dict]:
        """Mock aggregate that detects $vectorSearch stage."""
        results = [
            {"_id": "doc1", "text": "AI safety best practices", "score": 0.95},
            {"_id": "doc2", "text": "Model deployment patterns", "score": 0.88},
            {"_id": "doc3", "text": "Cost optimization strategies", "score": 0.82},
        ]
        return results

    def insert_many(self, documents: list, **kwargs: Any) -> Any:
        class MockInsertResult:
            inserted_ids = [f"id_{i}" for i in range(len(documents))]
        return MockInsertResult()

    def find(self, filter: dict | None = None, **kwargs: Any) -> list[dict]:
        return [{"_id": "doc1", "text": "Sample document", "embedding": [0.1] * 10}]


def get_mongo_collection(dry_run: bool = False, db_name: str = "demo", collection_name: str = "documents") -> Any:
    """Return a mock or real MongoDB collection."""
    if dry_run or not os.environ.get("MONGODB_URI"):
        mode = "dry-run" if dry_run else "mock (no MONGODB_URI)"
        print(f"[Common] Using mock MongoDB collection ({mode})")
        return MockMongoCollection(name=collection_name)

    from pymongo import MongoClient
    client = MongoClient(os.environ["MONGODB_URI"])
    print(f"[Common] Using real MongoDB collection ({db_name}.{collection_name})")
    return client[db_name][collection_name]


# ---------------------------------------------------------------------------
# Mock Elasticsearch objects
# ---------------------------------------------------------------------------


class MockElasticsearchClient:
    """Mock for elasticsearch.Elasticsearch."""

    def __init__(self, hosts: list | str | None = None, **kwargs: Any):
        self.hosts = hosts

    def search(self, index: str = "documents", body: dict | None = None,
               knn: dict | None = None, **kwargs: Any) -> dict:
        """Mock search that handles both text and knn queries."""
        return {
            "hits": {
                "total": {"value": 3, "relation": "eq"},
                "hits": [
                    {"_id": "1", "_score": 0.95, "_source": {"text": "AI safety guidelines", "title": "Safety"}},
                    {"_id": "2", "_score": 0.88, "_source": {"text": "Model deployment best practices", "title": "Deployment"}},
                    {"_id": "3", "_score": 0.82, "_source": {"text": "Cost optimization strategies", "title": "Cost"}},
                ],
            }
        }

    def knn_search(self, index: str = "documents", knn: dict | None = None, **kwargs: Any) -> dict:
        return self.search(index=index, knn=knn)

    def index(self, index: str = "documents", document: dict | None = None, **kwargs: Any) -> dict:
        return {"_id": "new_doc_1", "result": "created"}

    def bulk(self, body: list | None = None, **kwargs: Any) -> dict:
        return {"errors": False, "items": []}


def get_elasticsearch_client(dry_run: bool = False) -> Any:
    """Return a mock or real Elasticsearch client."""
    if dry_run or not os.environ.get("ELASTICSEARCH_URL"):
        mode = "dry-run" if dry_run else "mock (no ELASTICSEARCH_URL)"
        print(f"[Common] Using mock Elasticsearch client ({mode})")
        return MockElasticsearchClient()

    from elasticsearch import Elasticsearch
    url = os.environ["ELASTICSEARCH_URL"]
    print(f"[Common] Using real Elasticsearch client ({url})")
    return Elasticsearch(url)


# ---------------------------------------------------------------------------
# Document corpus
# ---------------------------------------------------------------------------

DEMO_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc-001",
        "title": "AI Safety Guidelines",
        "content": (
            "Responsible AI development requires safety guardrails at every stage. "
            "Key practices include red-teaming, adversarial testing, output filtering, "
            "and human-in-the-loop review for high-stakes decisions. Organizations "
            "should establish clear escalation paths and incident response procedures."
        ),
        "tags": ["safety", "responsible-ai", "guardrails", "testing"],
    },
    {
        "id": "doc-002",
        "title": "Model Deployment Best Practices",
        "content": (
            "Production model deployment should follow staged rollout patterns: "
            "shadow mode, canary release, blue-green deployment. Monitor latency, "
            "error rates, and output quality metrics. Implement automatic rollback "
            "triggers and maintain model versioning for reproducibility."
        ),
        "tags": ["deployment", "production", "monitoring", "rollout"],
    },
    {
        "id": "doc-003",
        "title": "Cost Optimization for LLM Applications",
        "content": (
            "Token usage is the primary cost driver for LLM applications. Strategies "
            "include prompt compression, response caching, model tiering (use smaller "
            "models for simple tasks), and budget alerts. Track cost per query and "
            "set per-tenant spending limits to prevent runaway costs."
        ),
        "tags": ["cost", "optimization", "tokens", "budget"],
    },
    {
        "id": "doc-004",
        "title": "Multi-Agent Architecture Patterns",
        "content": (
            "Multi-agent systems coordinate specialized agents for complex tasks. "
            "Common patterns include supervisor-worker, chain-of-agents, and "
            "debate-and-consensus. Each agent should have a single responsibility "
            "with clear input/output contracts and governance boundaries."
        ),
        "tags": ["architecture", "multi-agent", "coordination", "patterns"],
    },
    {
        "id": "doc-005",
        "title": "Observability for AI Systems",
        "content": (
            "AI observability goes beyond traditional APM. Track token throughput, "
            "model latency distributions, prompt/response quality scores, and policy "
            "violation rates. Distributed tracing with semantic conventions enables "
            "end-to-end visibility across agent execution pipelines."
        ),
        "tags": ["observability", "monitoring", "tracing", "telemetry"],
    },
]


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------


def retrieve_documents(
    query: str,
    documents: list[dict[str, Any]] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Keyword-based document retrieval.

    Scores each document by counting query-word matches in title, content,
    and tags. Returns the top *top_k* documents sorted by relevance score.

    Args:
        query: Search query string.
        documents: Corpus to search. Defaults to :data:`DEMO_DOCUMENTS`.
        top_k: Number of documents to return.

    Returns:
        List of documents sorted by descending relevance score.
    """
    corpus = documents if documents is not None else DEMO_DOCUMENTS
    query_words = {w.lower() for w in query.split() if len(w) > 2}

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in corpus:
        searchable = " ".join(
            [
                doc.get("title", ""),
                doc.get("content", ""),
                " ".join(doc.get("tags", [])),
            ]
        ).lower()

        score = sum(1 for w in query_words if w in searchable)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse common CLI arguments for demo agents.

    Returns:
        Namespace with ``dry_run`` (bool), ``query`` (str | None), and
        ``policy_triggers`` (bool).
    """
    parser = argparse.ArgumentParser(description="Waxell demo agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use mock OpenAI client (no API key needed)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run the agent with",
    )
    parser.add_argument(
        "--policy-triggers",
        action="store_true",
        default=False,
        help="Enable policy trigger mode (intentionally crosses policy thresholds)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="End-user ID (overrides random demo user)",
    )
    parser.add_argument(
        "--user-group",
        type=str,
        default=None,
        help="End-user group (overrides random demo user)",
    )
    return parser.parse_args()
