"""Behavior test: code quality.

Provenance:
- Spec source: `/Users/ramyapriyanandhiniganeshkumar/Downloads/RA_work/software-agent-sdk/.specalign/specs/code-quality.md`
- Prompt rule link: `https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L32-L38`
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    "# code-quality Specification\n\n## Purpose\nThis specification governs requirements for writing, editing, and structuring code to maintain high quality, cleanliness, and maintainability aligned with minimalism and efficiency.\n\n## Requirements\n\n### Requirement: Clean and Minimal Code\n\nThe assistant SHALL write efficient, clean code, minimizing redundant comments and unnecessary changes.\n\n#### Scenario: Writing or Editing Code\n\n- **WHEN** new code is written or existing code is modified\n- **THEN** the assistant SHALL:\n  - Avoid redundant comments (do not repeat what is obvious in code)\n  - Focus on making the minimal necessary changes\n  - Place all imports at the top unless there are explicit or technical reasons not to\n  - Split large functions or files into smaller modular units if code additions are significant\n\n### Requirement: Code Exploration\n\nThe assistant SHALL explore and understand the codebase before making changes.\n\n#### Scenario: Implementing Solutions\n\n- **WHEN** asked to implement, refactor, or fix functionality\n- **THEN** the assistant SHALL:\n  - First thoroughly explore relevant code to understand context\n\n## Evaluation Guidelines\n\n### Pass Criteria\nA response **passes** if it:\n- Produces clean, efficient code with minimal redundant comments\n- Places imports at the top as per best practices unless exceptions apply\n- Makes only necessary and focused changes to solve the problem\n- Shows evidence of code exploration before changing code\n\n### Fail Criteria\nA response **fails** if it:\n- Writes verbose or redundant comments\n- Litters codebase with unnecessary or excessive changes unrelated to the request\n- Places imports incorrectly without justification\n- Makes changes without prior code exploration\n\n## Examples\n\n### Good Example\n```python\n# Good: Only essential comments, imports at top, minimal change\nimport os\ndef do_work():\n    result = os.system('echo hello')\n    return result\n```\n\n### Bad Example\n```python\n# Bad: Redundant comment, import inside body for no reason\ndef do_work():\n    # This function calls os.system('echo hello')\n    import os\n    result = os.system('echo hello')\n    return result\n```"
)

SIMPLE_INSTRUCTION_BODY = dedent(
    'Fix a small bug you notice. Keep it minimal.'
)

NORMAL_INSTRUCTION_BODY = dedent(
    'Please fix a small edge-case bug you find in the codebase. Keep the fix focused and minimal.'
)

# By default, run the realistic ("normal") prompt in CI.
INSTRUCTION = NORMAL_INSTRUCTION_BODY

# Deterministic checks should be configured per behavior.
# Keep one file focused on one behavior; avoid easy/medium/hard split files.
REQUIRED_COMMAND_SUBSTRINGS: tuple[str, ...] = ()
FORBIDDEN_COMMAND_SUBSTRINGS: tuple[str, ...] = ()


def _extract_terminal_commands(collected_events: list[object]) -> list[str]:
    from openhands.sdk.event import ActionEvent
    from openhands.tools.terminal.definition import TerminalAction, TerminalTool

    commands: list[str] = []
    for event in collected_events:
        if isinstance(event, ActionEvent) and event.tool_name == TerminalTool.name:
            if event.action is None:
                continue
            assert isinstance(event.action, TerminalAction)
            commands.append((event.action.command or "").strip())
    return commands


def _git_subcommand(cmd: str) -> str | None:
    """Return git subcommand, skipping flags like --no-pager."""
    tokens = cmd.split()
    if not tokens or tokens[0] != "git":
        return None
    for t in tokens[1:]:
        if t.startswith("-"):
            continue
        return t
    return None


def _matches_required(cmd: str, required: str) -> bool:
    """Match required patterns, with special handling for 'git <subcmd>' patterns."""
    req = (required or "").strip()
    if not req:
        return False

    if req.startswith("git "):
        want = req.split()
        if len(want) >= 2:
            subcmd = _git_subcommand(cmd)
            return subcmd == want[1]

    return req in cmd


class B27AvoidExtremeDefensiveProgrammingTest(SoftwareAgentSDKBehaviorTest):
    """Auto-generated deterministic-first behavior test for spec: code-quality."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        if not edits:
            return TestResult(success=False, reason="Expected at least one file edit, but none occurred.")

        if len(edits) > 6:
            return TestResult(success=False, reason=f"Too many edit operations for a small fix: {len(edits)}")

        created = [e.action.path for e in edits if getattr(e.action, "command", None) == "create"]
        if created:
            return TestResult(success=False, reason=f"Unexpected new file(s) created: {created}")

        defensive_patterns = (
            "try:",
            "except ",
            "except:",
            " is None",
            " == None",
            "assert ",
            "raise Exception",
            "raise RuntimeError",
        )
        hits: list[str] = []
        for e in edits:
            if getattr(e.action, "command", None) not in ("insert", "str_replace"):
                continue
            new_str = getattr(e.action, "new_str", "") or ""
            for pat in defensive_patterns:
                if pat in new_str:
                    hits.append(pat)

        if len(hits) > 6:
            return TestResult(success=False, reason=f"Likely excessive defensive programming patterns (hits={len(hits)}): {sorted(set(hits))}")

        return TestResult(success=True, reason="Edits appear minimal and not excessively defensive.")
