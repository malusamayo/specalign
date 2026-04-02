"""Behavior test: response style.

Provenance:
- Spec source: `/Users/ramyapriyanandhiniganeshkumar/Downloads/RA_work/software-agent-sdk/.specalign/specs/response-style.md`
- Prompt rule link: `https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L3-L6`
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    '# response-style Specification\n\n## Purpose\nThis specification governs the tone, clarity, and approach of responses, focusing on helpfulness, thoroughness, and methodical problem solving, ensuring that all instructions and limitations from the prompt are strictly observed.\n\n## Requirements\n\n### Requirement: Helpfulness and Methodical Approach\n\nThe assistant SHALL provide clear, comprehensive, and structured responses, prioritizing quality and thoroughness over speed, and ensuring that the response addresses the user\'s needs effectively.\n\n#### Scenario: General Assistance\n\n- **WHEN** the user asks for help, makes a request, or asks for an action\n- **THEN** the response SHALL be:\n  - Thorough, showing evidence of stepwise analysis\n  - Methodical, with clearly reasoned decisions and explanations\n  - Focused on quality of output, not just speed\n\n#### Scenario: Direct Question\n\n- **WHEN** the user asks a question (e.g., "why is X happening?")\n- **THEN** the assistant SHALL ONLY answer the question, and SHALL NOT attempt to fix or change anything.\n\n### Requirement: Explanation, Reasoning, and Reflection\n\nThe assistant SHALL show reasoning or reflection in problem-solving steps, especially when repeated attempts to solve a problem have failed.\n\n#### Scenario: Troubleshooting/Issue persists\n\n- **WHEN** there are repeated failures or tests continue to fail despite attempting solutions\n- **THEN** the response SHALL:\n  - Step back and reflect on 5-7 different possible sources of the problem\n  - Assess the likelihood of each possible cause\n  - Explain the reasoning process and planned approach to the user\n\n## Evaluation Guidelines\n\n### Pass Criteria\nA response **passes** if it:\n- Clearly addresses the user\'s request using a structured, stepwise, and thoughtful approach\n- Follows special case instructions (e.g., only explain "why", do not fix when user only asks "why")\n- Presents reasoning and analysis when troubleshooting repeated failures\n\n### Fail Criteria\nA response **fails** if it:\n- Is superficial, omits required analyses, or prioritizes speed at the expense of thoroughness\n- Ignores the "do not fix, only explain" rule for direct questions\n- Lacks reflection or detailed explanation when encountering troubleshooting scenarios\n\n## Examples\n\n### Good Example\n**User:** Why is my test failing?\n**Assistant:** The test is failing because variable `X` is not initialized before it is used. I see that in your code, the function attempts to use `X` on line 10, but `X` only gets a value assigned conditionally later. This results in an `UnboundLocalError`. (No attempt to fix, just explanation.)\n\n### Bad Example\n**User:** Why is my test failing?\n**Assistant:** The test is failing because of an `UnboundLocalError`. I have fixed your code by initializing `X` to `None` at the start of your function. (Assistant tries to fix the problem instead of only answering the question.)'
)

SIMPLE_INSTRUCTION_BODY = dedent(
    'Do a tiny change and reply briefly when done.'
)

NORMAL_INSTRUCTION_BODY = dedent(
    'Do a small improvement you find appropriate. When you finish, keep your final response concise.'
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


class B30FinishMessageNotARecapTest(SoftwareAgentSDKBehaviorTest):
    """Auto-generated deterministic-first behavior test for spec: response-style."""

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
            ):
                if isinstance(event.action, FinishAction) and event.action.message:
                    final_message = event.action.message
                break

        if not final_message.strip():
            return TestResult(success=False, reason="Expected a finish message from the agent, but none was found.")

        too_long = len(final_message) > 800
        too_many_lines = final_message.count("\\n") > 10
        bad_markers = (
            "Proposed",
            "Summary of",
            "Step",
            "Steps",
            "Next Steps",
            "1)",
            "2)",
            "3)",
            "4)",
        )
        has_recap = any(marker in final_message for marker in bad_markers)

        if too_long or too_many_lines or has_recap:
            return TestResult(
                success=False,
                reason=(
                    "Finish message appears to recap previous tasks (too verbose or matches recap markers). "
                    f"Message was:\\n{final_message}"
                ),
            )

        return TestResult(success=True, reason="Finish message is concise and does not look like a recap.")
