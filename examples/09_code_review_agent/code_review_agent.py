#!/usr/bin/env python3
"""Code review agent demo with agentic behavior tracking.

Demonstrates waxell-observe's behavior tracking methods through a code review
pipeline using Anthropic. The agent parses diffs, runs parallel static
analysis tools, retrieves coding standards, applies multi-step reasoning,
and generates a review with retry/fallback handling.

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
import os

from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

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


async def run_code_review_agent(
    pr_description: str,
    client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
    dry_run: bool = True,
) -> dict:
    """Execute the code review agent pipeline.

    Args:
        pr_description: The PR title/description to review.
        client: Anthropic-compatible async client (real or mock).
        observe_active: Whether observe/governance is active.
        session: Session ID for trace correlation.
        user_id: End-user identifier.
        user_group: End-user group/tier.
        dry_run: Whether to use mock clients.

    Returns:
        Dict with review results and scores.
    """
    async with WaxellContext(
        agent_name="code-review-agent",
        workflow_name="pr-review",
        inputs={"pr_description": pr_description, "diff_lines": len(_MOCK_DIFF.splitlines())},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "code-review")
        ctx.set_tag("pipeline", "pr-review")
        ctx.set_tag("language", "python")
        ctx.set_metadata("pr_number", 42)
        ctx.set_metadata("repo", "acme/web-app")

        try:
            # ------------------------------------------------------------------
            # Step 1: Parse Diff (tool call)
            # ------------------------------------------------------------------
            print("[CodeReview] Step 1: Parsing code diff...")

            diff_result = {
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

            ctx.record_tool_call(
                name="diff_parser",
                input={"diff": _MOCK_DIFF[:200] + "..."},
                output=diff_result,
                duration_ms=12,
                status="ok",
                tool_type="function",
            )
            print(f"           Parsed: {diff_result['files_changed']} file, +{diff_result['lines_added']} lines")

            # ------------------------------------------------------------------
            # Step 2: Parallel Static Analysis Tools
            # ------------------------------------------------------------------
            print("[CodeReview] Step 2: Running static analysis (parallel tools)...")

            # Tool 2a: Linter
            linter_result = {
                "warnings": 2,
                "errors": 0,
                "issues": [
                    {"line": 6, "severity": "warning", "message": "Hardcoded secret detected", "rule": "S105"},
                    {"line": 9, "severity": "warning", "message": "String concatenation in SQL query", "rule": "S608"},
                ],
            }
            ctx.record_tool_call(
                name="linter",
                input={"file": "src/auth/handler.py", "rules": ["security", "style", "complexity"]},
                output=linter_result,
                duration_ms=340,
                status="ok",
                tool_type="function",
            )
            print(f"           Linter: {linter_result['warnings']} warnings, {linter_result['errors']} errors (340ms)")

            # Tool 2b: Test Runner
            test_result = {
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
            ctx.record_tool_call(
                name="test_runner",
                input={"suite": "auth", "coverage": True},
                output=test_result,
                duration_ms=2150,
                status="ok",
                tool_type="function",
            )
            print(f"           Tests: {test_result['passed']} passed, {test_result['failed']} failed, {test_result['coverage']:.0%} coverage (2150ms)")

            # Tool 2c: Security Scanner
            security_result = {
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
            ctx.record_tool_call(
                name="security_scanner",
                input={"file": "src/auth/handler.py", "scan_type": "full"},
                output=security_result,
                duration_ms=890,
                status="ok",
                tool_type="function",
            )
            print(f"           Security: {len(security_result['vulnerabilities'])} vulnerabilities, risk score {security_result['risk_score']} (890ms)")

            # ------------------------------------------------------------------
            # Step 3: Retrieve Coding Standards
            # ------------------------------------------------------------------
            print("[CodeReview] Step 3: Retrieving coding standards...")

            standards_docs = [
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

            ctx.record_retrieval(
                query="python security best practices authentication JWT",
                documents=standards_docs,
                source="standards_db",
                duration_ms=32,
                top_k=5,
            )
            print(f"           Retrieved {len(standards_docs)} coding standards from standards_db")

            # ------------------------------------------------------------------
            # Step 4: Reasoning Chain (4 steps)
            # ------------------------------------------------------------------
            print("[CodeReview] Step 4: Analyzing findings...")

            # Reasoning 1: Evaluate warnings
            ctx.record_reasoning(
                step="evaluate_warnings",
                thought=(
                    "Linter flagged 2 warnings: hardcoded secret (S105) and string concatenation "
                    "in SQL (S608). Both are security-relevant. The hardcoded secret is line 6 "
                    "(SECRET_KEY = 'my-secret-key-123'). The SQL concatenation is line 9 where "
                    "username is directly interpolated. Both align with security scanner findings."
                ),
                evidence=["linter:S105", "linter:S608", "security:CWE-89", "security:CWE-798"],
                conclusion="Linter warnings are genuine security issues confirmed by scanner",
            )
            print("           [1/4] evaluate_warnings: Confirmed genuine security issues")

            # Reasoning 2: Assess test failure
            ctx.record_reasoning(
                step="assess_test_failure",
                thought=(
                    "One test failure: test_password_validation expects bcrypt but code uses MD5. "
                    "This confirms the security scanner finding about weak hashing (CWE-328). "
                    "The existing test suite was written expecting proper security practices, "
                    "meaning this PR regresses from the project's security standards."
                ),
                evidence=["test:test_password_validation", "security:CWE-328", "std-002"],
                conclusion="Test failure confirms security regression; MD5 must be replaced with bcrypt",
            )
            print("           [2/4] assess_test_failure: Test failure confirms security regression")

            # Reasoning 3: Check security
            ctx.record_reasoning(
                step="check_security",
                thought=(
                    "Three vulnerabilities found: SQL injection (critical, CWE-89), hardcoded "
                    "secret (high, CWE-798), and weak hash (medium, CWE-328). The SQL injection "
                    "is the most severe -- it allows arbitrary database access. Combined risk "
                    "score of 9.2/10. Per coding standard std-001, parameterized queries are "
                    "mandatory. Per std-002, bcrypt/argon2 is required for password hashing."
                ),
                evidence=["security:CWE-89", "security:CWE-798", "security:CWE-328", "std-001", "std-002"],
                conclusion="Critical security vulnerabilities must be addressed before merge",
            )
            print("           [3/4] check_security: Critical vulnerabilities found")

            # Reasoning 4: Overall assessment
            ctx.record_reasoning(
                step="overall_assessment",
                thought=(
                    "The authentication logic works functionally but has fundamental security "
                    "flaws. The code structure is reasonable (separate authenticate/get_current_user "
                    "functions), JWT usage is correct in principle, but the implementation violates "
                    "3 coding standards. The test coverage at 78% is acceptable but the failing "
                    "test indicates the author was aware of security requirements. Recommendation: "
                    "request changes with specific remediation steps."
                ),
                evidence=[
                    "linter:2_warnings",
                    "tests:1_failed",
                    "security:3_vulnerabilities",
                    "standards:3_violations",
                ],
                conclusion="Request changes: fix SQL injection, use bcrypt, externalize secrets",
            )
            print("           [4/4] overall_assessment: Request changes recommended")

            # ------------------------------------------------------------------
            # Step 5: Decide Review Outcome
            # ------------------------------------------------------------------
            print("[CodeReview] Step 5: Deciding review outcome...")

            ctx.record_decision(
                name="review_outcome",
                options=["approve", "request_changes", "suggest_improvements"],
                chosen="request_changes",
                reasoning=(
                    "Critical SQL injection vulnerability (CWE-89) and hardcoded secret "
                    "(CWE-798) make this PR unsafe to merge. Security scanner risk score "
                    "9.2/10. One test failure confirms security regression. Must fix before merge."
                ),
                confidence=0.95,
                metadata={"risk_score": 9.2, "blocking_issues": 3},
            )
            print("           Decision: request_changes (confidence: 0.95)")

            # ------------------------------------------------------------------
            # Step 6: Generate Review with Retry
            # ------------------------------------------------------------------
            print("[CodeReview] Step 6: Generating review comment...")

            # First attempt times out
            ctx.record_retry(
                attempt=1,
                reason="Request timed out generating detailed review",
                strategy="retry",
                original_error="Request timed out after 30s",
                max_attempts=3,
            )
            print("           Attempt 1: Timed out (30s)")

            # Simulate timeout with failing client
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
                pass  # Expected failure

            # Retry with simplified prompt
            ctx.record_retry(
                attempt=2,
                reason="Retrying with simplified prompt to reduce latency",
                strategy="retry",
                max_attempts=3,
            )
            print("           Attempt 2: Retrying with simplified prompt")

            review_prompt = (
                f"PR: {pr_description}\n\n"
                f"Findings:\n"
                f"- SQL injection on line 9 (CRITICAL)\n"
                f"- Hardcoded secret on line 6 (HIGH)\n"
                f"- MD5 password hashing on line 10 (MEDIUM)\n"
                f"- 1 test failure (password validation)\n\n"
                "Generate a concise code review requesting changes."
            )

            review_response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": review_prompt,
                    },
                ],
            )
            review_text = review_response.content[0].text

            ctx.record_step("generate_review", output={"review_length": len(review_text)})
            ctx.record_llm_call(
                model=review_response.model,
                tokens_in=review_response.usage.input_tokens,
                tokens_out=review_response.usage.output_tokens,
                task="generate_review",
                prompt_preview=review_prompt[:200],
                response_preview=review_text[:200],
            )
            print(f"           Review generated ({len(review_text)} chars)")

            # ------------------------------------------------------------------
            # Step 7: Quality Scores
            # ------------------------------------------------------------------
            print("[CodeReview] Step 7: Recording quality scores...")

            scores = {
                "code_quality": 0.72,
                "test_coverage": 0.94,
                "security": 0.3,
                "style_compliance": 0.88,
                "overall": 0.65,
            }
            for name, value in scores.items():
                ctx.record_score(name, value)
                print(f"           {name}: {value:.2f}")

            # ------------------------------------------------------------------
            # Set final result
            # ------------------------------------------------------------------
            result = {
                "review": review_text,
                "outcome": "request_changes",
                "vulnerabilities": len(security_result["vulnerabilities"]),
                "test_failures": test_result["failed"],
                "risk_score": security_result["risk_score"],
                "scores": scores,
            }
            ctx.set_result(result)

            print()
            print(f"[CodeReview] Review: {review_text[:200]}...")
            print(
                f"[CodeReview] Complete. "
                f"(1 LLM call, 4 tool calls, 1 retrieval, 4 reasoning steps, "
                f"1 decision, 2 retries, 5 scores)"
            )
            return result

        except PolicyViolationError as e:
            print(f"\n[CodeReview] POLICY VIOLATION: {e}")
            print("[CodeReview] Agent halted by governance policy.")
            return {"error": str(e)}


async def main() -> None:
    args = parse_args()
    pr_description = args.query or DEFAULT_QUERY

    client = get_anthropic_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Code Review Agent] Starting PR review pipeline...")
    print(f"[Code Review Agent] Session: {session}")
    print(f"[Code Review Agent] End user: {user_id} ({user_group})")
    print(f"[Code Review Agent] PR: {pr_description[:80]}...")
    print()

    await run_code_review_agent(
        pr_description=pr_description,
        client=client,
        observe_active=observe_active,
        session=session,
        user_id=user_id,
        user_group=user_group,
        dry_run=args.dry_run,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
