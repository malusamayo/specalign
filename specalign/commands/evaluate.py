"""Implementation of 'specalign evaluate' command."""

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Optional

import click

from specalign.llm_client import LLMClient
from specalign.utils import batch_inference
from specalign.workspace import Workspace


EVALUATION_PROMPT_TEMPLATE = """You are evaluating an LLM response against a specific specification.

SPECIFICATION:
{spec_content}

USER INPUT:
{user_input}

MODEL OUTPUT:
{model_output}

Based on the specification's requirements, evaluation guidelines, and examples, evaluate this response.

Provide your evaluation in the following JSON format:
{{
  "score": 1 or 0 (1 = pass, 0 = fail),
  "rationale": "Brief explanation of why it passed or failed",
  "violations": ["list", "of", "specific", "violation", "types"] or []
}}

Output ONLY the JSON, nothing else."""


def load_data_samples(data_config_path: Path, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    """Load data samples from configured dataset.

    Args:
        data_config_path: Path to data configuration file.
        max_samples: Maximum number of samples to load.

    Returns:
        List of data samples.
    """
    with open(data_config_path) as f:
        data_config = json.load(f)

    data_path = Path(data_config["path"])
    data_format = data_config.get("format", "csv")

    samples = []

    if data_format == "csv":
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                samples.append({"prompt": row.get("prompt", row.get("input", ""))})

    elif data_format == "json":
        with open(data_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data[:max_samples] if max_samples else data
            else:
                samples = [data]

    elif data_format == "jsonl":
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                samples.append(json.loads(line))

    return samples


def evaluate_response(
    spec_name: str,
    spec_content: str,
    user_input: str,
    model_output: str,
    eval_client: LLMClient
) -> dict[str, Any]:
    """Evaluate a single response against a specification.

    Args:
        spec_name: Name of the specification.
        spec_content: Full specification content.
        user_input: User input/prompt.
        model_output: Model's response.
        eval_client: LLM client for evaluation.

    Returns:
        Evaluation result with score, rationale, and violations.
    """
    eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        spec_content=spec_content,
        user_input=user_input,
        model_output=model_output
    )

    response = eval_client.generate(eval_prompt)

    # Parse JSON response
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    try:
        result = json.loads(response.strip())
        return result
    except json.JSONDecodeError:
        click.echo(f"Warning: Could not parse evaluation response for {spec_name}", err=True)
        return {
            "score": 0,
            "rationale": "Evaluation parsing failed",
            "violations": ["evaluation_error"]
        }


def run_evaluate(
    workspace: Workspace,
    model_config_path: Path,
    data_config_path: Path,
    max_samples: Optional[int] = None,
    prompt_number: Optional[int] = None,
    max_workers: int = 10
) -> None:
    """Run the evaluate command.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration.
        data_config_path: Path to data configuration.
        max_samples: Maximum number of samples to evaluate.
        prompt_number: Specific prompt number to use (latest if None).
        max_workers: Maximum number of parallel workers (default: 10).
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return

    # Get prompt to evaluate
    if prompt_number is None:
        prompt_number = workspace.get_next_prompt_number() - 1

    if prompt_number < 1:
        click.echo("Error: No prompts found. Run 'specalign compile' first.", err=True)
        return

    prompt_dir = workspace.prompts_dir / str(prompt_number)
    prompt_file = prompt_dir / "prompt.md"

    if not prompt_file.exists():
        click.echo(f"Error: Prompt #{prompt_number} not found.", err=True)
        return

    with open(prompt_file) as f:
        system_prompt = f.read()

    click.echo(f"Using prompt #{prompt_number}")

    # Load data
    click.echo(f"Loading data from {data_config_path}...")
    samples = load_data_samples(data_config_path, max_samples)
    click.echo(f"Loaded {len(samples)} samples")

    # Load specifications
    spec_files = workspace.get_all_spec_files()
    specs = {}
    for spec_file in spec_files:
        with open(spec_file) as f:
            spec_name = spec_file.stem
            specs[spec_name] = f.read()

    click.echo(f"Evaluating against {len(specs)} specifications:")
    for spec_name in specs:
        click.echo(f"  - {spec_name}")

    # Create LLM clients
    model_client = LLMClient(model_config_path=model_config_path)
    eval_client = LLMClient.create_default_client("vertex_ai/gemini-2.5-flash")

    # Run evaluation
    results = {
        "run_metadata": {
            "command": f"specalign evaluate --model={model_config_path} --data={data_config_path} --max_samples={max_samples or 'all'}",
            "model_path": str(model_config_path),
            "evaluation_model": "vertex_ai/gemini-2.5-flash",
            "data_path": str(data_config_path),
            "max_samples": len(samples),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt_number": prompt_number,
        },
        "specs_evaluated": list(specs.keys()),
        "samples": []
    }

    def generate_response(sample_id: int, user_prompt: str) -> dict[str, Any]:
        """Generate a single model response.

        Args:
            sample_id: Sample identifier.
            user_prompt: User input prompt.

        Returns:
            Dict with sample_id, user_prompt, and model_output.
        """
        try:
            model_output = model_client.generate(user_prompt, system_prompt=system_prompt)
        except Exception as e:
            click.echo(f"\nError generating response for sample {sample_id}: {e}", err=True)
            model_output = f"[Generation error: {e}]"

        return {
            "sample_id": sample_id,
            "user_prompt": user_prompt,
            "model_output": model_output
        }

    def evaluate_sample(sample_id: int, user_prompt: str, model_output: str) -> dict[str, Any]:
        """Evaluate a single sample against all specs.

        Args:
            sample_id: Sample identifier.
            user_prompt: User input prompt.
            model_output: Model's generated output.

        Returns:
            Dict with evaluations for all specs.
        """
        def evaluate_spec(spec_name: str, spec_content: str) -> dict[str, Any]:
            """Evaluate against a single spec."""
            try:
                return evaluate_response(
                    spec_name, spec_content, user_prompt, model_output, eval_client
                )
            except Exception as e:
                click.echo(f"\nError evaluating sample {sample_id} against {spec_name}: {e}", err=True)
                return {
                    "score": 0,
                    "rationale": f"Evaluation error: {e}",
                    "violations": ["evaluation_error"]
                }

        # Evaluate all specs in parallel using batch_inference
        eval_args = [
            {"spec_name": spec_name, "spec_content": spec_content}
            for spec_name, spec_content in specs.items()
        ]

        eval_results = batch_inference(
            evaluate_spec,
            eval_args,
            use_process=False,
            max_workers=len(specs)
        )

        # Convert list of results to dict keyed by spec_name
        evaluations = {
            spec_name: eval_results[i]
            for i, spec_name in enumerate(specs.keys())
        }

        return {
            "sample_id": sample_id,
            "input": {"prompt": user_prompt},
            "model_output": {"text": model_output},
            "evaluations": evaluations
        }

    # Step 1: Generate responses for all samples in parallel
    click.echo(f"Generating responses in parallel (max {max_workers} workers)...")
    generation_args = [
        {
            "sample_id": sample_id,
            "user_prompt": sample.get("prompt", sample.get("input", ""))
        }
        for sample_id, sample in enumerate(samples)
    ]

    generation_results = batch_inference(
        generate_response,
        generation_args,
        use_process=False,
        max_workers=min(len(samples), max_workers)
    )

    # Step 2: Evaluate all samples in parallel
    click.echo(f"Evaluating responses in parallel (max {max_workers} workers)...")
    evaluation_args = [
        {
            "sample_id": gen_result["sample_id"],
            "user_prompt": gen_result["user_prompt"],
            "model_output": gen_result["model_output"]
        }
        for gen_result in generation_results
    ]

    sample_results = batch_inference(
        evaluate_sample,
        evaluation_args,
        use_process=False,
        max_workers=min(len(samples), max_workers)
    )

    # Add all results
    results["samples"] = sample_results

    # Calculate summary statistics
    total_samples = len(results["samples"])
    per_spec_passes = {spec_name: 0 for spec_name in specs}

    for sample in results["samples"]:
        for spec_name, eval_result in sample["evaluations"].items():
            if eval_result["score"] == 1:
                per_spec_passes[spec_name] += 1

    results["summary"] = {
        "total_samples": total_samples,
        "overall_pass_rate": sum(per_spec_passes.values()) / (total_samples * len(specs)) if total_samples > 0 else 0,
        "per_spec_pass_rate": {
            spec_name: count / total_samples if total_samples > 0 else 0
            for spec_name, count in per_spec_passes.items()
        }
    }

    # Save results
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = workspace.results_dir / f"eval_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nEvaluation complete!")
    click.echo(f"Results saved to: {results_file}")
    click.echo(f"\nSummary:")
    click.echo(f"  Total samples: {total_samples}")
    click.echo(f"  Overall pass rate: {results['summary']['overall_pass_rate']:.1%}")
    click.echo(f"  Per-spec pass rates:")
    for spec_name, pass_rate in results["summary"]["per_spec_pass_rate"].items():
        click.echo(f"    - {spec_name}: {pass_rate:.1%}")
