"""Implementation of 'specalign compile' command."""

import datetime
from pathlib import Path

import click
import yaml

from specalign.llm_client import LLMClient
from specalign.workspace import Workspace


PROMPT_COMPILATION_SYSTEM = """You are a prompt engineering expert. Your task is to synthesize multiple specification documents into a single, coherent, and effective LLM prompt.

The prompt should:
1. Be clear, concise, and actionable
2. Cover all requirements from the specifications
3. Be structured in a way that the LLM can easily follow
4. Include relevant examples where appropriate
5. Use appropriate tone and formatting

Output only the final prompt text, ready to be used as a system prompt or instruction."""


def compile_prompt(spec_files: list[Path], llm_client: LLMClient) -> str:
    """Compile specifications into a single prompt.

    Args:
        spec_files: List of specification file paths.
        llm_client: LLM client for compilation.

    Returns:
        Compiled prompt text.
    """
    # Read all spec files
    specs_content = []
    for spec_file in spec_files:
        with open(spec_file) as f:
            content = f.read()
            specs_content.append(f"=== {spec_file.name} ===\n{content}")

    all_specs = "\n\n".join(specs_content)

    user_prompt = f"""Please compile the following specifications into a single, effective LLM prompt:

{all_specs}

Create a prompt that an LLM can follow to meet all these specifications. The prompt should be clear, actionable, and comprehensive."""

    compiled_prompt = llm_client.generate(
        prompt=user_prompt,
        system_prompt=PROMPT_COMPILATION_SYSTEM
    )

    return compiled_prompt


def run_compile(workspace: Workspace, model_config_path: Path) -> None:
    """Run the compile command.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration YAML file. If None, uses default model.
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return

    # Get all spec files
    spec_files = workspace.get_all_spec_files()

    if not spec_files:
        click.echo(f"Error: No specification files found in {workspace.specs_dir}", err=True)
        click.echo(f"Create at least one .md file in the {workspace.specs_dir} directory", err=True)
        return

    click.echo(f"Found {len(spec_files)} specification file(s):")
    for spec_file in spec_files:
        click.echo(f"  - {spec_file.name}")

    # Create LLM client for compilation
    click.echo(f"\nCompiling prompt using model config: {model_config_path}...")
    llm_client = LLMClient(model_config_path=model_config_path)
    model_display = str(model_config_path)
    try:
        compiled_prompt = compile_prompt(spec_files, llm_client)

        # Create new prompt directory
        prompt_dir = workspace.create_prompt_dir()
        prompt_number = prompt_dir.name

        # Save prompt
        prompt_file = prompt_dir / "prompt.md"
        with open(prompt_file, "w") as f:
            f.write(compiled_prompt)

        # Save config
        config = {
            "compilation": {
                "model": model_display,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "spec_files": [str(f.relative_to(workspace.root)) for f in spec_files],
            }
        }

        config_file = prompt_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save compilation log
        log_content = f"""Compilation Log
===============

Timestamp: {config['compilation']['timestamp']}
Model: {model_display}

Specifications Used:
"""
        for spec_file in spec_files:
            log_content += f"  - {spec_file.name}\n"

        log_file = prompt_dir / "compilation.log"
        with open(log_file, "w") as f:
            f.write(log_content)

        click.echo(f"\nPrompt compiled successfully!")
        click.echo(f"Saved to: {prompt_dir}/")
        click.echo(f"  - prompt.md (prompt #{prompt_number})")
        click.echo(f"  - config.yaml")
        click.echo(f"  - compilation.log")

        click.echo(f"\nView the prompt: cat {prompt_file}")

    except Exception as e:
        click.echo(f"Error during compilation: {e}", err=True)
        raise
