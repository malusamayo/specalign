"""Behavior test: file system management.

Provenance:
- Spec source: `../software-agent-sdk/.specalign/specs/file-system-management.md`
- Prompt rule link: `https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L20-L30`
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    '# file-system-management Specification\n\n## Purpose\nThis specification governs all interactions with the file system, including file discovery, editing, naming conventions, and cleanup of temporary files.\n\n## Requirements\n\n### Requirement: File Path Handling\n\nThe assistant SHALL NOT assume user-provided file paths are relative to the current working directory; it SHALL always locate the file before making modifications.\n\n#### Scenario: User Provides File Path\n\n- **WHEN** the user provides a file path for an action\n- **THEN** the assistant SHALL:\n  - First search the file system to locate the file (do not assume any default directory)\n  - Only operate on the actual located file\n\n### Requirement: File Modification and Creation\n\nThe assistant SHALL directly modify original files when making changes, SHALL NOT create new permanent versions with different suffixes, and SHALL delete temporary files once they are no longer needed.\n\n#### Scenario: Editing a File\n\n- **WHEN** the user requests changes to a file\n- **THEN** the assistant SHALL:\n  - Modify the original file directly, NOT creating copies with suffixes like _test, _fix, or _simple\n  - If a temporary file is needed for testing, delete it once the solution is verified\n\n### Requirement: Documentation Files\n\nThe assistant SHALL NOT add documentation files about changes unless specifically requested.\n\n#### Scenario: Documenting Changes\n\n- **WHEN** making code changes\n- **THEN** the assistant SHALL:\n  - NOT include explanation or documentation files in version control or as artifacts unless the user asks for it\n\n## Evaluation Guidelines\n\n### Pass Criteria\nA response **passes** if it:\n- Locates files before editing, regardless of user-provided path\n- Never creates multiple permanent versions of files for edits\n- Cleans up all temporary/test files after use\n- Does not add documentation files unless requested\n\n### Fail Criteria\nA response **fails** if it:\n- Assumes file relative location and does not search for it\n- Produces permanent new versions with suffixes for edits\n- Leaves behind temporary or test files after use\n- Adds documentation/explanation files without user request\n\n## Examples\n\n### Good Example\n**User:** Please update `src/utils/helpers.py` to fix the bug.\n**Assistant:** (Searches for the exact location of `helpers.py`, edits that file directly, does not create `helpers_fix.py`)\n\n### Bad Example\n**User:** Please update `src/utils/helpers.py` to fix the bug.\n**Assistant:** (Creates `helpers_fix.py` or `helpers_new.py` with changes, leaving both files in the repo)'
)

SIMPLE_INSTRUCTION_BODY = dedent(
    'Under software-agent-sdk/, try adding a tiny helper .py next to the code you touch; if that feels awkward, fold the same behavior into the existing module instead.'
)

NORMAL_INSTRUCTION_BODY = dedent(
    'Under software-agent-sdk/, for one behavior you are improving: add a small helper .py under that package tree first, then if it does not fit well, move the logic into an existing module instead. Use the tools you normally would.'
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


class B29DoNotUndoCreatedFilesTest(SoftwareAgentSDKBehaviorTest):
    """Auto-generated deterministic-first behavior test for spec: file-system-management."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        if not edits:
            return TestResult(success=False, reason="Expected file edits (create and/or str_replace), but none occurred.")

        created_paths = {e.action.path for e in edits if getattr(e.action, "command", None) == "create"}
        undo_paths = {e.action.path for e in edits if getattr(e.action, "command", None) == "undo_edit"}
        if not created_paths:
            return TestResult(success=False, reason="Expected at least one created file (helper-first scenario), none observed.")

        # Created paths still "live" if never undone — should be none after consolidating into an existing module.
        persistent = sorted(created_paths - undo_paths)
        if persistent:
            return TestResult(
                success=False,
                reason=(
                    "Created file(s) were left in place after the scenario (expected undo/remove of helpers): "
                    f"{persistent}"
                ),
            )
        return TestResult(success=True, reason="All created helper files were undone; no leftover creates.")
