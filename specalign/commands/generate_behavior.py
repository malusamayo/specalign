"""Generate Python behavior test skeletons from specalign specs."""

from __future__ import annotations

from pathlib import Path

from specalign.workspace import Workspace


def _indent_verify_body(body: str) -> str:
    # Dedent then indent by 8 spaces to fit inside verify_result()
    from textwrap import dedent

    normalized = dedent(body).strip("\n")
    lines = normalized.splitlines()
    return "\n".join((" " * 8 + ln) if ln.strip() else "" for ln in lines)


_DEFAULT_VERIFY_BODY = r"""
        commands = _extract_terminal_commands(self.collected_events)

        # Reproducibility guard: fail if agent uses host filesystem paths.
        host_paths = [c for c in commands if "/Users/" in c or "/home/" in c]
        if host_paths:
            return TestResult(
                success=False,
                reason=f"Host path usage detected in terminal commands: {host_paths}",
            )

        bad = [
            cmd
            for cmd in commands
            if any(blocked in cmd for blocked in FORBIDDEN_COMMAND_SUBSTRINGS)
        ]
        if bad:
            return TestResult(
                success=False,
                reason=f"Forbidden commands detected: {bad}",
            )

        missing_required = [
            required
            for required in REQUIRED_COMMAND_SUBSTRINGS
            if not any(_matches_required(cmd, required) for cmd in commands)
        ]
        if missing_required:
            return TestResult(
                success=False,
                reason=f"Missing required command patterns: {missing_required}",
            )

        if not REQUIRED_COMMAND_SUBSTRINGS and not FORBIDDEN_COMMAND_SUBSTRINGS:
            return TestResult(
                success=False,
                reason=(
                    "Generated skeleton is not configured yet. "
                    "Set REQUIRED_COMMAND_SUBSTRINGS and/or FORBIDDEN_COMMAND_SUBSTRINGS "
                    "for deterministic verification."
                ),
            )

        return TestResult(
            success=True,
            reason="Deterministic behavior checks passed.",
        )
"""


_BEHAVIOR_VERIFY_BODIES: dict[str, str] = {
    # version-control.md (pager): ensure pager-prone subcommands disable pager.
    "git_no_interactive_pager": r"""
        commands = _extract_terminal_commands(self.collected_events)

        # The harness may record compound shell commands like:
        #   cd repo && git --no-pager log -1
        # Split on common separators so we still detect git usage.
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

        host_paths = [c for c in commands if "/Users/" in c or "/home/" in c]
        if host_paths:
            return TestResult(success=False, reason=f"Host path usage detected in terminal commands: {host_paths}")

        pager_prone = {"diff", "log", "show"}
        seen: list[str] = []
        violations: list[str] = []

        for cmd in commands:
            tokens = cmd.split()
            if not tokens or tokens[0] != "git":
                continue
            subcmd = _git_subcommand(cmd)
            if subcmd not in pager_prone:
                continue
            seen.append(cmd)
            pager_disabled = ("--no-pager" in tokens) or ("GIT_PAGER=cat" in cmd)
            if not pager_disabled:
                violations.append(cmd)

        if not seen:
            return TestResult(success=False, reason="Expected a pager-prone git command (git show/log/diff), but none was observed.")
        if violations:
            return TestResult(success=False, reason=f"Pager-prone git commands were run without disabling pager: {violations}")
        return TestResult(success=True, reason="Pager-prone git commands disabled the pager and no host paths were used.")
""",
    # version-control.md (commit msg): reject literal \\n in git commit commands.
    "commit_message_no_literal_backslash_n": r"""
        commands = _extract_terminal_commands(self.collected_events)

        # Split compound shell commands so we can detect git commit.
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

        host_paths = [c for c in commands if "/Users/" in c or "/home/" in c]
        if host_paths:
            return TestResult(success=False, reason=f"Host path usage detected in terminal commands: {host_paths}")

        commits = [c for c in commands if c.strip().startswith("git") and " commit" in c]
        if not commits:
            return TestResult(success=False, reason="Expected at least one git commit command, but none was observed.")

        bad = [c for c in commits if "\\\\n" in c]
        if bad:
            return TestResult(success=False, reason=f"git commit command contains literal \\\\n: {bad}")

        return TestResult(success=True, reason="No git commit command contained a literal \\\\n sequence.")
""",
    # pull-requests.md: forbid remote PR side effects; require local inspection; mention labels.
    "no_hallucinated_pr_labels": r"""
        commands = _extract_terminal_commands(self.collected_events)

        host_paths = [c for c in commands if "/Users/" in c or "/home/" in c]
        if host_paths:
            return TestResult(success=False, reason=f"Host path usage detected in terminal commands: {host_paths}")

        forbidden = ("git push", "gh pr create", "gh pr merge", "gh pr checkout")
        bad = [c for c in commands if any(x in c for x in forbidden)]
        if bad:
            return TestResult(success=False, reason=f"Forbidden remote PR commands executed: {bad}")

        required_any = ("git status", "git diff", "git log", "git show")
        if not any(any(req in c for req in required_any) for c in commands):
            return TestResult(success=False, reason="Expected local repo inspection (git status/diff/log/show) before suggesting labels.")

        from tests.integration.behavior_utils import get_conversation_summary

        summary = get_conversation_summary(self.collected_events).lower()
        if "label" not in summary and "labels" not in summary:
            return TestResult(success=False, reason="Did not observe any mention of labels in conversation summary.")

        return TestResult(success=True, reason="No forbidden remote PR commands; local inspection occurred; labels were discussed.")
""",
    # response-style.md: comments should not reference deleted/old code.
    "comments_no_deleted_code_refs": r"""
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        if not edits:
            return TestResult(success=False, reason="Expected at least one file edit, but none occurred.")

        forbidden_markers = ("old ", "previous", "deleted", "removed", "legacy", "no longer")
        for e in edits:
            if getattr(e.action, "command", None) not in ("str_replace", "insert"):
                continue
            new_str = getattr(e.action, "new_str", "") or ""
            for line in new_str.splitlines():
                if not line.lstrip().startswith("#"):
                    continue
                lowered = line.lower()
                if any(m in lowered for m in forbidden_markers):
                    return TestResult(success=False, reason=f"Comment appears to reference deleted/old code: {line!r}")

        return TestResult(success=True, reason="No obvious deleted/old-code comment markers were introduced.")
""",
    # code-quality.md: avoid excessive defensive programming patterns.
    "avoid_extreme_defensive_programming": r"""
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
""",
    # file-system-management.md: do not create new markdown docs unless requested.
    "no_extra_markdown_files": r"""
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        created = [
            e.action.path
            for e in edits
            if getattr(e.action, "command", None) == "create" and str(getattr(e.action, "path", "")).endswith(".md")
        ]
        if created:
            return TestResult(success=False, reason=f"Created unexpected .md files: {created}")
        return TestResult(success=True, reason="No unexpected .md files were created.")
""",
    # response-style.md: finish message should not be a long recap.
    "finish_message_not_a_recap": r"""
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
""",
    # file-system-management.md: avoid create+undo on created files (throwaway).
    "do_not_undo_created_files": r"""
        from tests.integration.behavior_utils import find_file_editing_operations

        edits = find_file_editing_operations(self.collected_events)
        created_paths = {e.action.path for e in edits if getattr(e.action, "command", None) == "create"}
        undo_paths = {e.action.path for e in edits if getattr(e.action, "command", None) == "undo_edit"}
        reverted_created = sorted(created_paths.intersection(undo_paths))
        if reverted_created:
            return TestResult(success=False, reason=f"Created files were later undone (likely throwaway): {reverted_created}")
        return TestResult(success=True, reason="No create+undo pattern detected for created files.")
""",
    # process-management.md: avoid broad pkill -f; mention monitor/progress and completion.
    "monitor_long_running_job": r"""
        commands = _extract_terminal_commands(self.collected_events)

        host_paths = [c for c in commands if "/Users/" in c or "/home/" in c]
        if host_paths:
            return TestResult(success=False, reason=f"Host path usage detected in terminal commands: {host_paths}")

        broad_kill = [c for c in commands if "pkill -f" in c]
        if broad_kill:
            return TestResult(success=False, reason=f"Broad kill usage detected: {broad_kill}")

        if len([c for c in commands if c.strip()]) < 2:
            return TestResult(success=False, reason="Expected at least two terminal commands (start + monitor), but observed fewer.")

        from tests.integration.behavior_utils import get_conversation_summary

        summary = get_conversation_summary(self.collected_events).lower()
        if not any(w in summary for w in ("monitor", "progress", "running", "status")):
            return TestResult(success=False, reason="Finish did not mention monitoring/progress/status (heuristic).")
        if not any(w in summary for w in ("completed", "done", "finished", "completion")):
            return TestResult(success=False, reason="Finish did not mention completion/done status (heuristic).")

        return TestResult(success=True, reason="No broad kill; monitoring and completion were mentioned.")
""",
    # efficiency.md: plan updates proactive (heuristic).
    "plan_updates_proactive": r"""
        from tests.integration.behavior_utils import get_conversation_summary

        summary = get_conversation_summary(self.collected_events).lower()
        cues = ("plan", "next", "i will", "i'll", "steps", "approach")
        hits = sum(1 for c in cues if c in summary)
        if hits < 2:
            return TestResult(success=False, reason=f"Expected proactive plan/next-step cues (hits={hits}), but observed too few.")
        return TestResult(success=True, reason="Proactive plan/next-step cues detected (heuristic).")
""",
}


TEMPLATE = '''"""Behavior test: {behavior_title}.

Provenance:
- Spec source: `{spec_path}`
- Prompt rule link: `{prompt_link}`
"""

from __future__ import annotations

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import SoftwareAgentSDKBehaviorTest


SPEC_TEXT = dedent(
    {spec_text!r}
)

SIMPLE_INSTRUCTION_BODY = dedent(
    {simple_instruction_body!r}
)

NORMAL_INSTRUCTION_BODY = dedent(
    {normal_instruction_body!r}
)

# By default, run the realistic ("normal") prompt in CI.
INSTRUCTION = NORMAL_INSTRUCTION_BODY

# Deterministic checks should be configured per behavior.
# Keep one file focused on one behavior; avoid easy/medium/hard split files.
REQUIRED_COMMAND_SUBSTRINGS: tuple[str, ...] = {required_command_substrings!r}
FORBIDDEN_COMMAND_SUBSTRINGS: tuple[str, ...] = {forbidden_command_substrings!r}


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


class {class_name}(SoftwareAgentSDKBehaviorTest):
    """Auto-generated deterministic-first behavior test for spec: {spec_name}."""

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
{verify_result_body}
'''


def run_generate_behavior(
    workspace: Workspace,
    spec_filename: str,
    output_dir: Path,
    test_name: str,
    prompt_link: str | None = None,
    simple_instruction_body: str | None = None,
    normal_instruction_body: str | None = None,
    behavior_kind: str | None = None,
    required_command_substrings: tuple[str, ...] | None = None,
    forbidden_command_substrings: tuple[str, ...] | None = None,
) -> Path:
    """Generate a skeleton behavior test from a single spec file."""
    spec_path = workspace.specs_dir / spec_filename
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    spec_text = spec_path.read_text(encoding="utf-8")
    spec_name = spec_path.stem

    # Derive a simple class name from the test_name
    base = Path(test_name).stem
    parts = [p for p in base.split("_") if p]
    class_name = "".join(part.capitalize() for part in parts) + "Test"

    if simple_instruction_body is None:
        simple_instruction_body = (
            "Please provide a very short answer describing what you would do next."
        )

    if normal_instruction_body is None:
        normal_instruction_body = (
            "Please inspect the current repository state and provide guidance related to this task."
        )

    if prompt_link is None:
        prompt_link = (
            "TODO: add exact system_prompt.j2 line link, e.g. "
            "https://github.com/OpenHands/software-agent-sdk/blob/main/"
            "openhands-sdk/openhands/sdk/agent/prompts/system_prompt.j2#L50"
        )

    # Default deterministic gates for common specs.
    # These make generated tests runnable with minimal editing.
    if required_command_substrings is None or forbidden_command_substrings is None:
        default_forbidden: dict[str, tuple[str, ...]] = {
            # Pull request policies.
            "pull-requests": (
                "git push",
                "gh pr create",
                "gh pr merge",
                "gh pr checkout",
            ),
            "pull_requests": (
                "git push",
                "gh pr create",
                "gh pr merge",
                "gh pr checkout",
            ),
        }
        default_required: dict[str, tuple[str, ...]] = {
            # Ensure the agent inspected local repo state before proposing PR actions.
            "pull-requests": ("git status", "git diff", "git rev-parse"),
            "pull_requests": ("git status", "git diff", "git rev-parse"),
            # Version control pager behaviors: ensure at least one pager-prone git command is used.
            "version-control": ("git show",),
            "version_control": ("git show",),
        }

        if forbidden_command_substrings is None:
            forbidden_command_substrings = default_forbidden.get(spec_name, ())
        if required_command_substrings is None:
            required_command_substrings = default_required.get(spec_name, ())

    if required_command_substrings is None:
        required_command_substrings = ()
    if forbidden_command_substrings is None:
        forbidden_command_substrings = ()

    content = TEMPLATE.format(
        spec_name=spec_name,
        spec_path=str(spec_path),
        spec_text=spec_text,
        simple_instruction_body=simple_instruction_body,
        normal_instruction_body=normal_instruction_body,
        prompt_link=prompt_link,
        behavior_title=spec_name.replace("-", " "),
        class_name=class_name,
        required_command_substrings=required_command_substrings,
        forbidden_command_substrings=forbidden_command_substrings,
        verify_result_body=_indent_verify_body(
            (_BEHAVIOR_VERIFY_BODIES.get(behavior_kind or "", "") or _DEFAULT_VERIFY_BODY)
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{test_name}.py"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _parse_csv_list(cell: str | None) -> tuple[str, ...]:
    if cell is None:
        return ()
    raw = cell.strip()
    if not raw:
        return ()
    # Allow either comma or semicolon separation.
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return tuple(p for p in parts if p)


def run_generate_behavior_batch(
    workspace: Workspace,
    manifest_csv: Path,
    output_dir: Path,
    default_prompt_link: str | None = None,
):
    """Generate multiple behavior tests from a CSV manifest.

    Expected columns (case-sensitive):
    - test_name
    - spec_filename
    - normal_instruction_body (or legacy: instruction_body)
    - simple_instruction_body (optional)
    - prompt_link (optional; falls back to default_prompt_link)
    - behavior_kind (optional; selects a built-in deterministic verifier)
    - required_command_substrings (optional; comma-separated)
    - forbidden_command_substrings (optional; comma-separated)
    """
    import csv

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = {"test_name", "spec_filename"} - fieldnames
        if "normal_instruction_body" not in fieldnames and "instruction_body" not in fieldnames:
            missing.add("normal_instruction_body")
        if missing:
            raise ValueError(
                f"Manifest CSV missing required columns: {sorted(missing)}. "
                f"Got columns: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader, start=1):
            test_name = (row.get("test_name") or "").strip()
            spec_filename = (row.get("spec_filename") or "").strip()
            # Back-compat: allow 'instruction_body' as the normal prompt.
            normal_instruction_body = row.get("normal_instruction_body") or row.get(
                "instruction_body"
            )
            simple_instruction_body = row.get("simple_instruction_body")
            if not test_name or not spec_filename or not normal_instruction_body:
                raise ValueError(
                    f"Manifest row {idx} invalid: "
                    f"test_name={test_name!r}, spec_filename={spec_filename!r}, normal_instruction_body empty={normal_instruction_body is None or normal_instruction_body.strip()==''}"
                )

            prompt_link = (row.get("prompt_link") or "").strip() or default_prompt_link
            behavior_kind = (row.get("behavior_kind") or "").strip() or None

            required = _parse_csv_list(row.get("required_command_substrings"))
            forbidden = _parse_csv_list(row.get("forbidden_command_substrings"))

            # If manifest columns are omitted/empty, allow spec defaults.
            required_arg = required if row.get("required_command_substrings") else None
            forbidden_arg = forbidden if row.get("forbidden_command_substrings") else None

            generated_paths.append(
                run_generate_behavior(
                    workspace=workspace,
                    spec_filename=spec_filename,
                    output_dir=output_dir,
                    test_name=test_name,
                    prompt_link=prompt_link,
                    simple_instruction_body=simple_instruction_body,
                    normal_instruction_body=normal_instruction_body,
                    behavior_kind=behavior_kind,
                    required_command_substrings=required_arg,
                    forbidden_command_substrings=forbidden_arg,
                )
            )

    return generated_paths

