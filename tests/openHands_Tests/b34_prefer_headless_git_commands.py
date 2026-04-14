"""Behavior test: prefer headless git commands over interactive pager-prone ones.

Provenance:
- Spec source: `../software-agent-sdk/.specalign/specs/version-control.md`
- Prompt rule link: `https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L40-L47`
- Property source: issue about git diff/show/log getting stuck in an interactive pager unless the pager is disabled
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    """
    # version-control Specification

    ## Purpose
    Catch cases where the assistant uses pager-prone git commands interactively instead
    of preferring headless variants that do not block execution.

    ## Requirement
    When using git commands whose output may open an interactive pager, the assistant
    should disable the pager by default.

    ## Pass Criteria
    - Uses git show/log/diff in a headless form
    - Disables pager with `git --no-pager` or `GIT_PAGER=cat`

    ## Fail Criteria
    - Uses `git diff`, `git show`, or `git log` without disabling pager
    """
)

SIMPLE_INSTRUCTION_BODY = dedent(
    """
    The git repo is in software-agent-sdk/ (not the workspace root). cd into it
    and show me the latest commit and its patch.
    """
).strip()

NORMAL_INSTRUCTION_BODY = dedent(
    """
    The git repo is in software-agent-sdk/ (not the workspace root). cd into it
    and inspect the most recent change so I can understand what was modified.
    Show the commit message and the actual patch.
    """
).strip()

INSTRUCTION = NORMAL_INSTRUCTION_BODY

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


class B34PreferHeadlessGitCommandsTest(SoftwareAgentSDKBehaviorTest):
    """Behavior test for avoiding interactive git pager usage."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        commands = _extract_terminal_commands(self.collected_events)

        # Split chained shell commands so we can inspect git usage precisely.
        segments: list[str] = []
        for cmd in commands:
            if not cmd:
                continue
            parts = [cmd]
            for sep in ("&&", ";"):
                next_parts: list[str] = []
                for p in parts:
                    next_parts.extend(p.split(sep))
                parts = next_parts
            segments.extend(p.strip() for p in parts if p.strip())
        commands = segments

        pager_prone = {"diff", "log", "show"}
        seen: list[str] = []
        violations: list[str] = []
        headless_seen: list[str] = []

        for cmd in commands:
            tokens = cmd.split()
            if not tokens:
                continue

            # Detect env-prefixed git commands like: GIT_PAGER=cat git show -1
            git_index = None
            for i, tok in enumerate(tokens):
                if tok == "git":
                    git_index = i
                    break
            if git_index is None:
                continue

            git_tokens = tokens[git_index:]
            if not git_tokens or git_tokens[0] != "git":
                continue

            subcmd = None
            for tok in git_tokens[1:]:
                if tok.startswith("-"):
                    continue
                subcmd = tok
                break

            if subcmd not in pager_prone:
                continue

            seen.append(cmd)

            pager_disabled = ("--no-pager" in git_tokens) or ("GIT_PAGER=cat" in cmd)
            if pager_disabled:
                headless_seen.append(cmd)
            else:
                violations.append(cmd)

        if not seen:
            return TestResult(
                success=False,
                reason="Expected a pager-prone git command (git show/log/diff), but none was observed.",
            )

        if violations:
            return TestResult(
                success=False,
                reason=f"Pager-prone git commands were run without disabling pager: {violations}",
            )

        return TestResult(
            success=True,
            reason=f"Pager-prone git commands were used in headless form: {headless_seen}",
        )

