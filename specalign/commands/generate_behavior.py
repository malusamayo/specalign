"""Generate Python behavior test skeletons from specalign specs."""

from __future__ import annotations

from pathlib import Path

from specalign.workspace import Workspace


# Very simple template: single-quoted string, no nested triple quotes.
TEMPLATE = """# Auto-generated behavior test from specalign spec: {spec_name}
# Source spec path: {spec_path}

from textwrap import dedent

from tests.integration.base import TestResult
from tests.integration.utils.behavior_helpers import (
    SoftwareAgentSDKBehaviorTest,
    append_environment_tips,
)
from tests.integration.behavior_utils import get_conversation_summary
from tests.integration.utils.llm_judge import judge_agent_behavior


SPEC_TEXT = dedent(
    {spec_text!r}
)

INSTRUCTION_BODY = dedent(
    {instruction_body!r}
)

INSTRUCTION = append_environment_tips(INSTRUCTION_BODY)


class {class_name}(SoftwareAgentSDKBehaviorTest):
    \"\"\"Auto-generated behavior test skeleton for spec: {spec_name}.\"\"\"

    INSTRUCTION: str = INSTRUCTION

    def verify_result(self) -> TestResult:
        \"\"\"Default verify_result that only uses the LLM judge.

        NOTE: You should replace this with concrete deterministic checks
        (filesystem, git, terminal events, etc.) derived from SPEC_TEXT, and
        optionally keep the judge as a qualitative layer.
        \"\"\"
        conversation_summary = get_conversation_summary(self.collected_events)
        criteria = (
            "Did the agent follow the specification '{spec_name}'?\\n"
            "Specification (excerpt):\\n" + SPEC_TEXT[:1000]
        )
        judgment = judge_agent_behavior(
            user_instruction=self.INSTRUCTION,
            conversation_summary=conversation_summary,
            evaluation_criteria=criteria,
        )

        if judgment.approved:
            return TestResult(
                success=True,
                reason=f"Behavior approved by LLM judge: {{judgment.reasoning}}",
            )

        return TestResult(
            success=False,
            reason=f"Behavior rejected by LLM judge: {{judgment.reasoning}}",
        )
"""


def run_generate_behavior(
    workspace: Workspace,
    spec_filename: str,
    output_dir: Path,
    test_name: str,
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

    instruction_body = (
        f"This is an auto-generated behavior test skeleton for the spec '{spec_name}'.\\n"
        "Please edit INSTRUCTION_BODY to describe a concrete scenario and update\\n"
        "verify_result() with deterministic checks derived from the spec."
    )

    content = TEMPLATE.format(
        spec_name=spec_name,
        spec_path=str(spec_path),
        spec_text=spec_text,
        instruction_body=instruction_body,
        class_name=class_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{test_name}.py"
    output_path.write_text(content, encoding="utf-8")
    return output_path

