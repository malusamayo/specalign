"""Command line interface for specalign."""

from pathlib import Path

import click

from specalign.commands.compile import run_compile
from specalign.commands.evaluate import run_evaluate
from specalign.commands.init import run_init
from specalign.workspace import Workspace


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """specalign - Generate, iterate, and maintain LLM prompts."""
    pass


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
def init(path: Path):
    """Initialize a new specalign workspace."""
    workspace = Workspace(path)
    run_init(workspace)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    default="vertex_ai/gemini-2.5-flash",
    help="Model to use for compilation (default: vertex_ai/gemini-2.5-flash)"
)
def compile(path: Path, model: str):
    """Generate a new prompt based on current specifications."""
    workspace = Workspace(path)
    run_compile(workspace, model)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to model configuration YAML file"
)
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to data configuration JSON file"
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to evaluate"
)
@click.option(
    "--prompt",
    type=int,
    default=None,
    help="Specific prompt number to evaluate (defaults to latest)"
)
@click.option(
    "--workers",
    type=int,
    default=10,
    help="Maximum number of parallel workers for sample processing (default: 10)"
)
def evaluate(path: Path, model: Path, data: Path, max_samples: int, prompt: int, workers: int):
    """Run evaluation on a prompt using specified model and data."""
    workspace = Workspace(path)
    run_evaluate(workspace, model, data, max_samples, prompt, workers)


if __name__ == "__main__":
    cli()
