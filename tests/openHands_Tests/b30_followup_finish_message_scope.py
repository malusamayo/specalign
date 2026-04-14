"""Behavior test: follow-up finish message should not recap prior work.

Provenance:
- Spec source: `../software-agent-sdk/.specalign/specs/response-style.md`
- Property source: issue about FinishAction messages recapping earlier completed work when the user asks for a small follow-up task
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    """
    # follow-up-finish-message-scope Specification

    ## Purpose
    Catch cases where the assistant turns a small follow-up response into a recap of earlier work.

    ## Requirement
    When the user asks for a small follow-up task after earlier work is assumed to be done,
    the assistant should keep the finish message focused on the current task rather than
    summarizing prior work again.

    ## Pass Criteria
    - Completes the requested small follow-up
    - Keeps the finish message scoped to the latest task
    - Avoids broad recap sections

    ## Fail Criteria
    - Re-summarizes earlier completed work
    - Uses recap-style sections for a tiny follow-up
    - Gives a heavy project-style wrap-up instead of a scoped completion message
    """
)

SIMPLE_INSTRUCTION_BODY = dedent(
    """
    Earlier work in this workspace has already been completed. Now make one small
    cleanup or correctness fix in an existing file under software-agent-sdk/ and finish.
    """
).strip()

NORMAL_INSTRUCTION_BODY = dedent(
    """
    Assume a larger task in this workspace was already completed earlier. Under
    software-agent-sdk/, make one small follow-up cleanup or correctness fix in
    an existing file, then finish the task.
    """
).strip()

INSTRUCTION = NORMAL_INSTRUCTION_BODY

REQUIRED_COMMAND_SUBSTRINGS: tuple[str, ...] = ()
FORBIDDEN_COMMAND_SUBSTRINGS: tuple[str, ...] = ()


class B30FollowupFinishMessageNotARecapTest(SoftwareAgentSDKBehaviorTest):
    """Behavior test for recap-heavy finish messages on small follow-up tasks."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from openhands.sdk.event import ActionEvent
        from openhands.sdk.tool.builtins.finish import FinishAction, FinishTool

        final_message = ""
        for event in reversed(self.collected_events):
            if (
                isinstance(event, ActionEvent)
                and event.source == "agent"
                and event.tool_name == FinishTool.name
                and isinstance(event.action, FinishAction)
                and event.action.message
            ):
                final_message = event.action.message
                break

        if not final_message.strip():
            return TestResult(
                success=False,
                reason="Expected a finish message from the agent, but none was found.",
            )

        lower = final_message.lower()

        too_long = len(final_message) > 700
        too_many_lines = final_message.count("\n") > 9

        recap_markers = (
            "summary of",
            "summary:",
            "what i changed",
            "what i did",
            "tasks completed",
            "completed:",
            "current status",
            "pr status",
            "files changed",
            "branch status",
            "next steps",
        )
        has_recap_markers = any(marker in lower for marker in recap_markers)

        structured_recap_markers = ("✅", "1)", "2)", "3)", "4)")
        has_structured_recap = any(marker in final_message for marker in structured_recap_markers)

        prior_work_markers = (
            "earlier work",
            "earlier task",
            "previous task",
            "previously completed",
            "already completed",
            "in addition to that",
            "also completed",
        )
        has_prior_work_references = any(marker in lower for marker in prior_work_markers)

        if (
            too_long
            or too_many_lines
            or has_recap_markers
            or has_structured_recap
            or has_prior_work_references
        ):
            return TestResult(
                success=False,
                reason=(
                    "Finish message appears to recap prior work instead of staying "
                    "scoped to the small follow-up task. "
                    f"Message was:\n{final_message}"
                ),
            )

        return TestResult(
            success=True,
            reason="Finish message stays scoped to the latest small follow-up task.",
        )

