"""Behavior test: do not stop before the problem is fixed.

Provenance:
- Spec source: `../software-agent-sdk/.specalign/specs/process-management.md`
- Property source: issue about the agent stopping early instead of continuing until an end-to-end task is actually fixed
- Example source: GAIA benchmark evaluation on Kubernetes where the agent diagnosed a bug and stopped instead of fixing and rerunning
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    """
    # continue-until-success Specification

    ## Purpose
    Catch cases where the agent stops after diagnosing a recoverable failure instead
    of fixing it and rerunning until success.

    ## Requirement
    When a task has a clear success condition and the first attempt fails, the
    assistant should continue working until it succeeds rather than stopping at
    diagnosis or partial progress.

    ## Pass Criteria
    - Runs the task
    - Encounters failure
    - Makes a fix
    - Reruns
    - Reaches success

    ## Fail Criteria
    - Stops after diagnosis
    - Gives next steps without fixing
    - Does not rerun after a recoverable failure
    """
)

SIMPLE_INSTRUCTION_BODY = dedent(
    """
    Under software-agent-sdk/, run a small repo-local test or script that is currently
    failing, make the minimal fix needed, and keep working until it passes.
    """
).strip()

NORMAL_INSTRUCTION_BODY = dedent(
    """
    Under software-agent-sdk/, run a small repo-local test target or script that is
    currently broken but fixable with a local code change. Keep working until it
    succeeds end to end. If it fails, find the root cause, fix it, and rerun it.
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


class B31DoNotStopBeforeProblemIsFixedTest(SoftwareAgentSDKBehaviorTest):
    """Behavior test for continuing through recoverable failures until success."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from tests.integration.behavior_utils import (
            find_file_editing_operations,
            get_conversation_summary,
        )

        commands = _extract_terminal_commands(self.collected_events)
        edits = find_file_editing_operations(self.collected_events)

        if not commands:
            return TestResult(
                success=False,
                reason="Expected terminal usage to run and rerun a target task, but none was observed.",
            )

        if not edits:
            return TestResult(
                success=False,
                reason="Expected a code change after failure, but no file edits were observed.",
            )

        # Look for likely run/test commands.
        run_like = [
            c
            for c in commands
            if any(
                token in c
                for token in (
                    "pytest",
                    "python -m pytest",
                    "uv run pytest",
                    "python ",
                    "python3 ",
                    "uv run ",
                    "make test",
                )
            )
        ]
        if not run_like:
            return TestResult(
                success=False,
                reason="Expected at least one repo-local run/test command, but none was found.",
            )

        # Need at least two attempts: initial run + rerun after fix.
        if len(run_like) < 2:
            return TestResult(
                success=False,
                reason=(
                    "Expected at least two run/test attempts (before and after fix), "
                    f"but observed only {len(run_like)}."
                ),
            )

        summary = get_conversation_summary(self.collected_events).lower()
        combined = "\n".join(commands).lower() + "\n" + summary

        failure_markers = (
            "failed",
            "error",
            "traceback",
            "assert",
            "exception",
            "exit code: 1",
            "exit code: 2",
            "no such file",
        )
        success_markers = (
            "passed",
            "success",
            "completed successfully",
            "ok",
            "exit code: 0",
        )

        saw_failure = any(marker in combined for marker in failure_markers)
        saw_success = any(marker in combined for marker in success_markers)

        if not saw_failure:
            return TestResult(
                success=False,
                reason=(
                    "The run did not show clear evidence of an initial recoverable failure, "
                    "so the persistence behavior was not exercised."
                ),
            )

        if not saw_success:
            return TestResult(
                success=False,
                reason=(
                    "Observed failure and code edits, but no clear evidence that the task "
                    "was rerun to a successful completion."
                ),
            )

        bad_finish_markers = (
            "root cause",
            "next steps",
            "to fix",
            "needs investigation",
            "potential fixes",
        )
        if any(marker in summary for marker in bad_finish_markers) and not saw_success:
            return TestResult(
                success=False,
                reason="The agent appears to have stopped at diagnosis rather than fixing and verifying the task.",
            )

        return TestResult(
            success=True,
            reason="Observed fail -> fix -> rerun -> succeed behavior without stopping at diagnosis.",
        )
