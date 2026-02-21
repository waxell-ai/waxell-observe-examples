#!/usr/bin/env python3
"""Code review agent demo with modern SDK decorator patterns.

Demonstrates waxell-observe's decorator-based behavior tracking through a
multi-agent code review pipeline using Anthropic. The orchestrator coordinates
a code-analyzer child agent (parses diffs, runs static analysis, retrieves
coding standards) and a code-evaluator child agent (applies reasoning,
decides outcome, generates review).

  code-review-orchestrator (parent)
  ├── @step: preprocess_pr
  ├── code-analyzer (child)
  │   ├── @tool: parse_diff, run_linter, run_tests, run_security_scan
  │   ├── @retrieval: retrieve_coding_standards
  │   └── waxell.tag/metadata
  └── code-evaluator (child)
      ├── @reasoning: evaluate_warnings, assess_test_failure,
      │               check_security, overall_assessment
      ├── @decision: decide_review_outcome
      ├── @retry: generate_review_comment
      └── waxell.score()

Usage::

    # Dry-run (no Anthropic API key needed)
    python examples/09_code_review_agent/code_review_agent.py --dry-run

    # With real Anthropic calls
    ANTHROPIC_API_KEY=sk-ant-... python examples/09_code_review_agent/code_review_agent.py

    # Custom query (treated as PR description)
    python examples/09_code_review_agent/code_review_agent.py --dry-run --query "Add user authentication endpoint"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from _common import setup_observe

setup_observe()

import waxell_observe as waxell
from waxell_observe import generate_session_id

from _common import (
    MockFailingAnthropic,
    get_anthropic_client,
    get_observe_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Review PR #42: Add user authentication with JWT tokens and session management"

# Simulated code diff for the review
_MOCK_DIFF = """\
diff --git a/src/auth/handler.py b/src/auth/handler.py
new file mode 100644
--- /dev/null
+++ b/src/auth/handler.py
@@ -0,0 +1,45 @@
+import jwt
+import hashlib
+from datetime import datetime, timedelta
+from flask import request, jsonify
+
+SECRET_KEY = "my-secret-key-123"  # TODO: move to env var
+
+def authenticate(username, password):
+    user = db.query("SELECT * FROM users WHERE username = '" + username + "'")
+    if user and hashlib.md5(password.encode()).hexdigest() == user.password_hash:
+        token = jwt.encode(
+            {"user_id": user.id, "exp": datetime.utcnow() + timedelta(hours=24)},
+            SECRET_KEY,
+            algorithm="HS256"
+        )
+        return jsonify({"token": token})
+    return jsonify({"error": "Invalid credentials"}), 401
+
+def get_current_user():
+    token = request.headers.get("Authorization", "").replace("Bearer ", "")
+    try:
+        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+        return payload["user_id"]
+    except jwt.InvalidTokenError:
+        return None
"""


# ---------------------------------------------------------------------------
# @step decorator -- auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="preprocess_pr")
async def preprocess_pr(pr_description: str) -> dict:
    """Clean and extract PR metadata."""
    cleaned = pr_description.strip()
    tokens = cleaned.lower().split()
    return {
        "original": pr_description,
        "cleaned": cleaned,
        "token_count": len(tokens),
        "diff_lines": len(_MOCK_DIFF.splitlines()),
    }


# ---------------------------------------------------------------------------
# @tool decorators -- auto-record static analysis tool calls
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="function")
def parse_diff(diff_text: str) -> dict:
    """Parse a code diff and extract file change metadata."""
    return {
        "files_changed": 1,
        "lines_added": 25,
        "lines_removed": 0,
        "files": [
            {
                "path": "src/auth/handler.py",
                "status": "added",
                "additions": 25,
                "deletions": 0,
            }
        ],
    }


@waxell.tool(tool_type="function")
def run_linter(file_path: str, rules: list) -> dict:
    """Run linter on a file with specified rules."""
    return {
        "warnings": 2,
        "errors": 0,
        "issues": [
            {"line": 6, "severity": "warning", "message": "Hardcoded secret detected", "rule": "S105"},
            {"line": 9, "severity": "warning", "message": "String concatenation in SQL query", "rule": "S608"},
        ],
    }


@waxell.tool(tool_type="function")
def run_tests(suite: str, coverage: bool = True) -> dict:
    """Run test suite and return results with coverage."""
    return {
        "passed": 47,
        "failed": 1,
        "skipped": 2,
        "coverage": 0.78,
        "failures": [
            {
                "test": "test_auth_handler::test_password_validation",
                "message": "AssertionError: Expected bcrypt hash, got md5",
            }
        ],
    }


@waxell.tool(tool_type="function")
def run_security_scan(file_path: str, scan_type: str = "full") -> dict:
    """Run security scanner on a file."""
    return {
        "vulnerabilities": [
            {
                "severity": "critical",
                "type": "sql_injection",
                "line": 9,
                "description": "User input concatenated into SQL query without parameterization",
                "cwe": "CWE-89",
            },
            {
                "severity": "high",
                "type": "hardcoded_secret",
                "line": 6,
                "description": "Secret key hardcoded in source code",
                "cwe": "CWE-798",
            },
            {
                "severity": "medium",
                "type": "weak_hash",
                "line": 10,
                "description": "MD5 used for password hashing; use bcrypt or argon2",
                "cwe": "CWE-328",
            },
        ],
        "risk_score": 9.2,
    }


# ---------------------------------------------------------------------------
# @retrieval decorator -- auto-record coding standards retrieval
# ---------------------------------------------------------------------------


@waxell.retrieval(source="standards_db")
def retrieve_coding_standards(query: str) -> list[dict]:
    """Retrieve relevant coding standards from the standards database."""
    return [
        {
            "id": "std-001",
            "title": "Python Security Best Practices",
            "score": 0.96,
            "snippet": "Always use parameterized queries. Never concatenate user input into SQL...",
        },
        {
            "id": "std-002",
            "title": "Authentication Implementation Guide",
            "score": 0.91,
            "snippet": "Use bcrypt or argon2 for password hashing. Store secrets in environment variables...",
        },
        {
            "id": "std-003",
            "title": "JWT Token Standards",
            "score": 0.87,
            "snippet": "Tokens should have short expiry (15min access, 7d refresh). Use RS256 for production...",
        },
    ]


# ---------------------------------------------------------------------------
# @reasoning decorators -- auto-record chain-of-thought
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="evaluate_warnings")
async def evaluate_warnings(linter_result: dict, security_result: dict) -> dict:
    """Evaluate linter warnings against security scanner findings."""
    return {
        "thought": (
            "Linter flagged 2 warnings: hardcoded secret (S105) and string concatenation "
            "in SQL (S608). Both are security-relevant. The hardcoded secret is line 6 "
            "(SECRET_KEY = 'my-secret-key-123'). The SQL concatenation is line 9 where "
            "username is directly interpolated. Both align with security scanner findings."
        ),
        "evidence": ["linter:S105", "linter:S608", "security:CWE-89", "security:CWE-798"],
        "conclusion": "Linter warnings are genuine security issues confirmed by scanner",
    }


@waxell.reasoning_dec(step="assess_test_failure")
async def assess_test_failure(test_result: dict) -> dict:
    """Assess test failures and their security implications."""
    return {
        "thought": (
            "One test failure: test_password_validation expects bcrypt but code uses MD5. "
            "This confirms the security scanner finding about weak hashing (CWE-328). "
            "The existing test suite was written expecting proper security practices, "
            "meaning this PR regresses from the project's security standards."
        ),
        "evidence": ["test:test_password_validation", "security:CWE-328", "std-002"],
        "conclusion": "Test failure confirms security regression; MD5 must be replaced with bcrypt",
    }


@waxell.reasoning_dec(step="check_security")
async def check_security(security_result: dict, standards: list) -> dict:
    """Assess security vulnerabilities against coding standards."""
    return {
        "thought": (
            "Three vulnerabilities found: SQL injection (critical, CWE-89), hardcoded "
            "secret (high, CWE-798), and weak hash (medium, CWE-328). The SQL injection "
            "is the most severe -- it allows arbitrary database access. Combined risk "
            "score of 9.2/10. Per coding standard std-001, parameterized queries are "
            "mandatory. Per std-002, bcrypt/argon2 is required for password hashing."
        ),
        "evidence": ["security:CWE-89", "security:CWE-798", "security:CWE-328", "std-001", "std-002"],
        "conclusion": "Critical security vulnerabilities must be addressed before merge",
    }


@waxell.reasoning_dec(step="overall_assessment")
async def overall_assessment(linter_result: dict, test_result: dict, security_result: dict, standards: list) -> dict:
    """Produce overall assessment combining all analysis findings."""
    return {
        "thought": (
            "The authentication logic works functionally but has fundamental security "
            "flaws. The code structure is reasonable (separate authenticate/get_current_user "
            "functions), JWT usage is correct in principle, but the implementation violates "
            "3 coding standards. The test coverage at 78% is acceptable but the failing "
            "test indicates the author was aware of security requirements. Recommendation: "
            "request changes with specific remediation steps."
        ),
        "evidence": [
            "linter:2_warnings",
            "tests:1_failed",
            "security:3_vulnerabilities",
            "standards:3_violations",
        ],
        "conclusion": "Request changes: fix SQL injection, use bcrypt, externalize secrets",
    }


# ---------------------------------------------------------------------------
# @decision decorator -- auto-record review outcome decision
# ---------------------------------------------------------------------------


@waxell.decision(name="review_outcome", options=["approve", "request_changes", "suggest_improvements"])
async def decide_review_outcome(security_result: dict, test_result: dict) -> dict:
    """Decide the review outcome based on analysis findings."""
    return {
        "chosen": "request_changes",
        "reasoning": (
            "Critical SQL injection vulnerability (CWE-89) and hardcoded secret "
            "(CWE-798) make this PR unsafe to merge. Security scanner risk score "
            "9.2/10. One test failure confirms security regression. Must fix before merge."
        ),
        "confidence": 0.95,
        "metadata": {"risk_score": 9.2, "blocking_issues": 3},
    }


# ---------------------------------------------------------------------------
# @retry decorator -- auto-record review generation with retry
# ---------------------------------------------------------------------------


@waxell.retry_dec(max_attempts=3, strategy="retry")
async def generate_review_comment(client, pr_description: str, findings: str) -> object:
    """Generate review comment via Anthropic with automatic retry on failure."""
    review_prompt = (
        f"PR: {pr_description}\n\n"
        f"Findings:\n{findings}\n\n"
        "Generate a concise code review requesting changes."
    )
    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": review_prompt,
            },
        ],
    )
    return response


# ---------------------------------------------------------------------------
# Child Agent 1: Code Analyzer -- parses diffs, runs tools, retrieves standards
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="code-analyzer", workflow_name="static-analysis")
async def run_code_analyzer(pr_description: str, diff_text: str, waxell_ctx=None):
    """Analyze code: parse diff, run linter/tests/security, retrieve standards."""
    waxell.tag("agent_role", "analyzer")
    waxell.tag("language", "python")
    waxell.metadata("pr_number", 42)
    waxell.metadata("repo", "acme/web-app")

    # Tool: Parse diff
    print("  [Analyzer] Parsing code diff...")
    diff_result = parse_diff(diff_text=diff_text[:200] + "...")
    print(f"    Parsed: {diff_result['files_changed']} file, +{diff_result['lines_added']} lines")

    # Tool: Linter
    print("  [Analyzer] Running linter...")
    linter_result = run_linter(
        file_path="src/auth/handler.py",
        rules=["security", "style", "complexity"],
    )
    print(f"    Linter: {linter_result['warnings']} warnings, {linter_result['errors']} errors")

    # Tool: Test runner
    print("  [Analyzer] Running tests...")
    test_result = run_tests(suite="auth", coverage=True)
    print(f"    Tests: {test_result['passed']} passed, {test_result['failed']} failed, {test_result['coverage']:.0%} coverage")

    # Tool: Security scanner
    print("  [Analyzer] Running security scan...")
    security_result = run_security_scan(file_path="src/auth/handler.py", scan_type="full")
    print(f"    Security: {len(security_result['vulnerabilities'])} vulnerabilities, risk score {security_result['risk_score']}")

    # Retrieval: Coding standards
    print("  [Analyzer] Retrieving coding standards...")
    standards = retrieve_coding_standards(query="python security best practices authentication JWT")
    print(f"    Retrieved {len(standards)} coding standards from standards_db")

    waxell.metadata("tool_calls_count", 4)
    waxell.metadata("standards_retrieved", len(standards))

    return {
        "diff_result": diff_result,
        "linter_result": linter_result,
        "test_result": test_result,
        "security_result": security_result,
        "standards": standards,
    }


# ---------------------------------------------------------------------------
# Child Agent 2: Code Evaluator -- reasons, decides, generates review
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="code-evaluator", workflow_name="review-evaluation")
async def run_code_evaluator(
    pr_description: str,
    analysis: dict,
    client,
    dry_run: bool = False,
    waxell_ctx=None,
):
    """Evaluate analysis results: reason about findings, decide outcome, generate review."""
    waxell.tag("agent_role", "evaluator")
    waxell.tag("provider", "anthropic")

    linter_result = analysis["linter_result"]
    test_result = analysis["test_result"]
    security_result = analysis["security_result"]
    standards = analysis["standards"]

    # Reasoning chain (4 steps)
    print("  [Evaluator] Reasoning about findings...")

    r1 = await evaluate_warnings(linter_result=linter_result, security_result=security_result)
    print(f"    [1/4] evaluate_warnings: {r1['conclusion']}")

    r2 = await assess_test_failure(test_result=test_result)
    print(f"    [2/4] assess_test_failure: {r2['conclusion']}")

    r3 = await check_security(security_result=security_result, standards=standards)
    print(f"    [3/4] check_security: {r3['conclusion']}")

    r4 = await overall_assessment(
        linter_result=linter_result,
        test_result=test_result,
        security_result=security_result,
        standards=standards,
    )
    print(f"    [4/4] overall_assessment: {r4['conclusion']}")

    # Decision: review outcome
    print("  [Evaluator] Deciding review outcome...")
    decision = await decide_review_outcome(security_result=security_result, test_result=test_result)
    chosen = decision.get("chosen", "request_changes")
    print(f"    Decision: {chosen} (confidence: {decision.get('confidence', 'N/A')})")

    # Generate review with retry
    print("  [Evaluator] Generating review comment...")

    # First attempt: simulate timeout with failing client
    failing_client = MockFailingAnthropic(
        fail_count=1,
        error_message="Request timed out after 30s",
    )
    try:
        await failing_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": "Generate review"}],
        )
    except Exception:
        print("    Attempt 1: Timed out (30s)")

    # Retry with simplified prompt succeeds
    print("    Attempt 2: Retrying with simplified prompt")
    findings_text = (
        "- SQL injection on line 9 (CRITICAL)\n"
        "- Hardcoded secret on line 6 (HIGH)\n"
        "- MD5 password hashing on line 10 (MEDIUM)\n"
        "- 1 test failure (password validation)"
    )
    review_response = await generate_review_comment(
        client=client,
        pr_description=pr_description,
        findings=findings_text,
    )
    review_text = review_response.content[0].text
    print(f"    Review generated ({len(review_text)} chars)")

    # Quality scores
    print("  [Evaluator] Recording quality scores...")
    scores = {
        "code_quality": 0.72,
        "test_coverage": 0.94,
        "security": 0.3,
        "style_compliance": 0.88,
        "overall": 0.65,
    }
    for name, value in scores.items():
        waxell.score(name, value)
        print(f"    {name}: {value:.2f}")

    return {
        "review": review_text,
        "outcome": chosen,
        "vulnerabilities": len(security_result["vulnerabilities"]),
        "test_failures": test_result["failed"],
        "risk_score": security_result["risk_score"],
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Orchestrator -- coordinates code-analyzer and code-evaluator
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="code-review-orchestrator", workflow_name="pr-review")
async def run_pipeline(
    pr_description: str,
    dry_run: bool = True,
    waxell_ctx=None,
):
    """Coordinate the code review pipeline across child agents.

    This is the parent agent. Child agents auto-link to this parent
    via WaxellContext lineage.
    """
    waxell.tag("demo", "code-review")
    waxell.tag("pipeline", "pr-review")
    waxell.metadata("pr_number", 42)
    waxell.metadata("repo", "acme/web-app")
    waxell.metadata("mode", "dry-run" if dry_run else "live")

    client = get_anthropic_client(dry_run=dry_run)

    # Phase 1: Preprocess PR
    print("[Orchestrator] Phase 1: Preprocessing PR (@step)...")
    preprocessed = await preprocess_pr(pr_description)
    print(f"  Preprocessed: {preprocessed['token_count']} tokens, {preprocessed['diff_lines']} diff lines")

    # Phase 2: Code Analyzer child agent
    print("[Orchestrator] Phase 2: Analyzing code (code-analyzer)...")
    analysis = await run_code_analyzer(
        pr_description=pr_description,
        diff_text=_MOCK_DIFF,
    )
    print(f"  Analysis complete: {len(analysis['security_result']['vulnerabilities'])} vulnerabilities found")

    # Phase 3: Code Evaluator child agent
    print("[Orchestrator] Phase 3: Evaluating findings (code-evaluator)...")
    result = await run_code_evaluator(
        pr_description=pr_description,
        analysis=analysis,
        client=client,
        dry_run=dry_run,
    )

    print()
    print(f"[Orchestrator] Review: {result['review'][:200]}...")
    print(
        f"[Orchestrator] Complete. "
        f"(4 tool calls, 1 retrieval, 4 reasoning steps, "
        f"1 decision, 1 retry, 5 scores)"
    )
    print(f"  Pipeline: code-review-orchestrator -> code-analyzer -> code-evaluator")

    return result


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

    print("[Code Review Agent] Starting PR review pipeline...")
    print(f"  Session:  {session}")
    print(f"  User:     {user_id} ({user_group})")
    print(f"  PR:       {query[:80]}...")
    print(f"  Mode:     {'dry-run' if args.dry_run else 'LIVE (Anthropic)'}")
    print()

    await run_pipeline(
        pr_description=query,
        dry_run=args.dry_run,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    )

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
