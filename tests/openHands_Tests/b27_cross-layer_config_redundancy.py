"""Behavior test: cross-layer config redundancy.

Provenance:
- Spec source: `../software-agent-sdk/.specalign/specs/configuration.md`
- Property source: issue about repeating defaults across workflow, config files, overlays, and runtime code instead of keeping one clear source of truth
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    """
    # cross-layer-config-redundancy Specification

    ## Purpose
    Catch cases where the assistant solves config tasks by repeating defaults across
    multiple layers instead of keeping configuration ownership clear.

    ## Requirement
    When a task spans workflow, config, overlays, and runtime code, the assistant
    should keep defaults in one clear place where practical and avoid scattering
    fallback logic across layers.

    ## Pass Criteria
    - Explores the relevant config flow before editing
    - Touches the relevant config layers
    - Avoids introducing unnecessary new fallback/default logic
    - Does not solve a cross-layer config task with a Python-only patch

    ## Fail Criteria
    - Repeats defaults across multiple layers
    - Pushes fallback logic into runtime code instead of fixing config flow
    - Edits only runtime/code files for a clearly cross-layer config task
    - Makes broad or noisy unrelated changes
    """
)

SIMPLE_INSTRUCTION_BODY = dedent(
    """
    Clean up the eval-job parameter flow so it is easier to understand where values
    come from, while preserving current behavior.
    """
).strip()

NORMAL_INSTRUCTION_BODY = dedent(
    """
    Add a new eval parameter called max_samples and wire it through the existing
    workflow, chart, and runtime path. Keep the implementation clean and consistent
    with the rest of the repo.
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


def _looks_like_config_path(path: str) -> bool:
    p = path.lower()
    return (
        p.endswith(".yml")
        or p.endswith(".yaml")
        or ".github/workflows" in p
        or "values.yaml" in p
        or "/envs/" in p
        or "/charts/" in p
        or "/helm/" in p
    )


def _looks_like_runtime_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(".py") or p.endswith(".sh")


class B27CrossLayerConfigRedundancyTest(SoftwareAgentSDKBehaviorTest):
    """Behavior test for cross-layer config redundancy."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        if not edits:
            return TestResult(
                success=False,
                reason="Expected at least one file edit, but none occurred.",
            )

        if len(edits) > 30:
            return TestResult(
                success=False,
                reason=f"Unusually many edit operations (possible thrashing): {len(edits)}",
            )

        commands = _extract_terminal_commands(self.collected_events)
        explored_text = "\n".join(commands)

        exploration_markers = (
            ".github/workflows",
            "values.yaml",
            "envs/evaluation",
            "eval-job",
            "helm",
            "chart",
        )
        if not any(marker in explored_text for marker in exploration_markers):
            return TestResult(
                success=False,
                reason=(
                    "No clear evidence of exploring workflow / values / overlay config "
                    "before editing."
                ),
            )

        edited_paths = [str(getattr(e.action, "path", "") or "") for e in edits]
        config_paths = [p for p in edited_paths if _looks_like_config_path(p)]
        runtime_paths = [p for p in edited_paths if _looks_like_runtime_path(p)]

        if not config_paths:
            return TestResult(
                success=False,
                reason=(
                    "Expected edits in workflow or YAML config files, but only runtime/code "
                    "files appear to have been edited."
                ),
            )

        suspicious_fallback_patterns = (
            ".get(",
            "setdefault(",
            "default=",
            " or ",
            "os.getenv(",
            "os.environ.get(",
        )

        suspicious_hits: list[tuple[str, str]] = []
        for e in edits:
            path = str(getattr(e.action, "path", "") or "")
            if not _looks_like_runtime_path(path):
                continue
            if getattr(e.action, "command", None) not in ("insert", "str_replace"):
                continue

            new_str = getattr(e.action, "new_str", "") or ""
            for pat in suspicious_fallback_patterns:
                if pat in new_str:
                    suspicious_hits.append((path, pat))

        if len(suspicious_hits) > 4:
            return TestResult(
                success=False,
                reason=(
                    "Likely added excessive runtime fallback/default logic instead of "
                    f"cleaning up config ownership: {suspicious_hits[:5]}"
                ),
            )

        created = [e.action.path for e in edits if getattr(e.action, "command", None) == "create"]
        forbidden_created = []
        for p in created:
            sp = str(p)
            if sp.endswith("/AGENTS.md") or sp.endswith("AGENTS.md"):
                continue
            if sp.endswith(".py") or sp.endswith(".ts") or sp.endswith(".tsx") or sp.endswith(".js"):
                forbidden_created.append(sp)

        if forbidden_created:
            return TestResult(
                success=False,
                reason=f"Unexpected new code file(s) created for a config task: {forbidden_created}",
            )

        touched_multiple_layers = len(set(config_paths)) >= 1 and (len(config_paths) + len(runtime_paths)) >= 2
        if not touched_multiple_layers:
            return TestResult(
                success=False,
                reason=(
                    "Edits do not show enough evidence of following the config flow across "
                    "the relevant layers."
                ),
            )

        return TestResult(
            success=True,
            reason="Changes appear focused on reducing cross-layer config redundancy.",
        )