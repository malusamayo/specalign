"""Behavior test: efficiency.

Provenance:
- Spec source: `/Users/ramyapriyanandhiniganeshkumar/Downloads/RA_work/software-agent-sdk/.specalign/specs/efficiency.md`
- Prompt rule link: `https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L15-L18`
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    "# efficiency Specification\n\n## Purpose\nThis specification governs the efficiency of command execution and the minimization of resource expenditure while performing actions on the system.\n\n## Requirements\n\n### Requirement: Action Combination\n\nThe assistant SHALL combine multiple actions into a single command wherever possible for efficiency.\n\n#### Scenario: Multiple Operations Needed\n\n- **WHEN** multiple shell or file operations can be efficiently grouped (e.g., using bash pipelines, `sed`, `grep`)\n- **THEN** the assistant SHALL:\n  - Try to combine actions in one command rather than running separate commands\n\n### Requirement: Efficient Exploration\n\nThe assistant SHALL use efficient tools for codebase exploration and minimize unnecessary operations.\n\n#### Scenario: Codebase Search\n\n- **WHEN** exploring the codebase\n- **THEN** the assistant SHALL:\n  - Use tools like `find`, `grep`, or git commands with filters for targeted searches\n\n## Evaluation Guidelines\n\n### Pass Criteria\nA response **passes** if it:\n- Combines compatible shell/file system/exploration actions into single commands\n- Uses efficient tools and avoids unnecessary work\n\n### Fail Criteria\nA response **fails** if it:\n- Needlessly executes many separate commands that could have been combined\n- Performs broad, wasteful codebase exploration\n\n## Examples\n\n### Good Example\n**Assistant:** Used `grep -r TODO . | sed 's/TODO/FIXME/g'` to find and replace in one command.\n\n### Bad Example\n**Assistant:** Ran `grep` for each file, and separate `sed` commands for each occurrence."
)

SIMPLE_INSTRUCTION_BODY = dedent(
    'Tell me your next steps at a high level.'
)

NORMAL_INSTRUCTION_BODY = dedent(
    'Please investigate the issue and tell me what you’ll do next, and then proceed.'
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


class B33PlanUpdatesProactiveTest(SoftwareAgentSDKBehaviorTest):
    """Auto-generated deterministic-first behavior test for spec: efficiency."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from tests.integration.behavior_utils import get_conversation_summary

        summary = get_conversation_summary(self.collected_events).lower()
        cues = ("plan", "next", "i will", "i'll", "steps", "approach")
        hits = sum(1 for c in cues if c in summary)
        if hits < 2:
            return TestResult(success=False, reason=f"Expected proactive plan/next-step cues (hits={hits}), but observed too few.")
        return TestResult(success=True, reason="Proactive plan/next-step cues detected (heuristic).")
