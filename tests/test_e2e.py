"""End-to-end style tests for the main specalign workflow components.

These tests:
- Use temporary workspaces (no writes to the real project tree).
- Mock out LLM calls so they are fast and cost-free.
- Exercise the core flows: init, compile, evaluate, optimize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from specalign.workspace import Workspace
from specalign.commands.init import run_init
from specalign.commands.compile import run_compile
from specalign.commands.evaluate import run_evaluate
from specalign.commands.optimize import run_optimize


class FakeLLMClient:
    """Cheap, deterministic stand‑in for LLMClient used in tests.

    Behavior is keyed off the prompt content so we can reuse the same fake
    for compilation, spec extraction, and evaluation.
    """

    def __init__(self, model_config_path: Path | None = None, model_name: str | None = None):  # type: ignore[unused-argument]
        self.model_config_path = model_config_path
        self.model_name = model_name

    @staticmethod
    def create_default_client(model_name: str = "test/model") -> "FakeLLMClient":
        # Mirrors LLMClient.create_default_client used in init.py
        return FakeLLMClient(model_name=model_name)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:  # type: ignore[override]
        # Spec extraction (used by `init` when a prompt is provided)
        if "You are a specification extraction expert" in prompt:
            return """
<specifications>
  <spec filename="response-style.md">
    <![CDATA[
# Response Style Specification

- The assistant SHALL respond in British English.
    ]]>
  </spec>
  <spec filename="html-formatting.md">
    <![CDATA[
# HTML Formatting Specification

- The assistant SHALL wrap headings in <h3> tags.
    ]]>
  </spec>
</specifications>
""".strip()

        # Prompt compilation (compile.py)
        if "Please compile the following specifications into a single, effective LLM prompt" in prompt:
            return "COMPILED PROMPT FROM SPECS"

        # Evaluation of model outputs against specs (utils.evaluate_responses_against_specs)
        if "You are evaluating an LLM response against a specific specification." in prompt:
            return json.dumps(
                {
                    "score": 1,
                    "rationale": "Meets all requirements.",
                    "violations": [],
                }
            )

        # Default: simulate a generic model completion
        base = "FAKE LLM OUTPUT"
        if system_prompt:
            return f"{base} (system={system_prompt[:32]!r})"
        return base


def _write_model_config(path: Path, name: str = "test/model") -> None:
    path.write_text(
        yaml.dump(
            {
                "model": {
                    "name": name,
                    "temperature": 0.1,
                    "max_tokens": 128,
                }
            },
            default_flow_style=False,
        )
    )


def _write_json_data_file(path: Path, samples: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(samples, indent=2))


def test_init_creates_workspace_and_model_config(monkeypatch, tmp_path: Path) -> None:
    """End‑to‑end test for `init` without real LLM/API calls.

    We mock user interaction so that:
    - A model config is created.
    - No dataset is configured (to keep the test minimal).
    - Specification extraction is skipped.
    """
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    ws = Workspace(workspace_root)

    # Patch prompts/confirmations to avoid interactive I/O.
    import click

    def fake_prompt(text: str, default: Any | None = None, type: Any | None = None):  # type: ignore[unused-argument]
        lower = text.lower()
        if "model name" in lower:
            return "test/model"
        if "temperature" in lower:
            return 0.1
        if "max tokens" in lower:
            return 128
        if "model config filename" in lower:
            return "default"
        # For anything else, just return default to keep behavior simple.
        return default

    def fake_confirm(text: str, default: bool = False) -> bool:  # type: ignore[unused-argument]
        # Skip dataset configuration and spec extraction in this test.
        return False

    monkeypatch.setattr(click, "prompt", fake_prompt)
    monkeypatch.setattr(click, "confirm", fake_confirm)

    # Also patch LLMClient used inside init to avoid any API usage if code path changes.
    import specalign.commands.init as init_mod

    monkeypatch.setattr(init_mod, "LLMClient", FakeLLMClient)

    run_init(ws)

    # Workspace structure should exist.
    assert ws.workspace_root.exists()
    for sub in ("specs", "prompts", "models", "data", "results"):
        assert (ws.workspace_root / sub).exists()

    # Default model config should have been created.
    model_cfg = ws.get_model_config("default")
    assert model_cfg.exists()
    cfg_data = yaml.safe_load(model_cfg.read_text())
    assert cfg_data["model"]["name"] == "test/model"


def test_compile_uses_specs_and_writes_prompt(monkeypatch, tmp_path: Path) -> None:
    """End‑to‑end test for `compile` using a fake LLM."""
    ws_root = tmp_path / "project"
    ws = Workspace(ws_root)
    ws.initialize()

    # Create a simple spec file.
    spec_file = ws.specs_dir / "response-style.md"
    spec_file.write_text("# Response Style\n\n- Use British English.\n")

    # Minimal model config.
    model_cfg = ws.models_dir / "model.yaml"
    _write_model_config(model_cfg, name="test/compile-model")

    # Patch LLMClient used by compile.
    import specalign.commands.compile as compile_mod

    monkeypatch.setattr(compile_mod, "LLMClient", FakeLLMClient)

    run_compile(ws, model_cfg)

    # First compiled prompt should be in prompts/1/prompt.md
    prompt_dir = ws.prompts_dir / "1"
    prompt_file = prompt_dir / "prompt.md"
    assert prompt_file.exists()
    content = prompt_file.read_text()
    assert "COMPILED PROMPT FROM SPECS" in content


def test_evaluate_generates_results(monkeypatch, tmp_path: Path) -> None:
    """End‑to‑end test for `evaluate` on a tiny JSON dataset."""
    ws_root = tmp_path / "project"
    ws = Workspace(ws_root)
    ws.initialize()

    # Seed prompt #1
    seed_prompt_dir = ws.create_prompt_dir(number=1)
    (seed_prompt_dir / "prompt.md").write_text("SYSTEM PROMPT FOR TESTING")

    # One simple spec.
    spec_file = ws.specs_dir / "spec.md"
    spec_file.write_text("# Spec\n\n- The assistant SHALL do something.\n")

    # Model + eval model configs (can be identical).
    model_cfg = ws.models_dir / "model.yaml"
    eval_model_cfg = ws.models_dir / "eval_model.yaml"
    _write_model_config(model_cfg, name="test/eval-model")
    _write_model_config(eval_model_cfg, name="test/eval-model")

    # Tiny JSON dataset with `prompt` field.
    data_file = ws.data_dir / "dataset.json"
    _write_json_data_file(
        data_file,
        [{"prompt": "example input 1"}, {"prompt": "example input 2"}],
    )

    # Data config pointing at the JSON file.
    data_cfg = ws.get_data_config("test_data")
    data_cfg.write_text(
        json.dumps(
            {
                "dataset_name": "test_data",
                "description": "small test dataset",
                "path": str(data_file),
                "format": "json",
            },
            indent=2,
        )
    )

    # Patch LLMClient in both evaluate and utils modules.
    import specalign.commands.evaluate as eval_mod
    import specalign.utils as utils_mod

    monkeypatch.setattr(eval_mod, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(utils_mod, "LLMClient", FakeLLMClient)

    # Also patch tqdm to avoid noisy output during tests.
    import tqdm as tqdm_mod

    monkeypatch.setattr(tqdm_mod, "tqdm", lambda x, total=None: x)  # type: ignore[assignment]

    run_evaluate(
        workspace=ws,
        model_config_path=model_cfg,
        data_config_path=data_cfg,
        max_samples=1,
        prompt_number=1,
        max_workers=2,
        eval_model_config_path=eval_model_cfg,
    )

    # One results file should have been written.
    result_files = list(ws.results_dir.glob("eval_*.json"))
    assert result_files, "No evaluation results were written"
    result = json.loads(result_files[0].read_text())

    assert result["summary"]["total_samples"] == 1
    # Because FakeLLM always returns score=1, pass rate should be 1.0.
    assert result["summary"]["overall_pass_rate"] == 1.0


def test_optimize_creates_new_prompt(monkeypatch, tmp_path: Path) -> None:
    """End‑to‑end test for `optimize` using a fake GEPA backend and fake LLMs."""
    ws_root = tmp_path / "project"
    ws = Workspace(ws_root)
    ws.initialize()

    # Seed prompt #1 (used as base for optimization).
    seed_prompt_dir = ws.create_prompt_dir(number=1)
    (seed_prompt_dir / "prompt.md").write_text("SEED SYSTEM PROMPT")

    # One spec to optimize against.
    spec_file = ws.specs_dir / "spec.md"
    spec_file.write_text("# Spec\n\n- The assistant SHALL do something.\n")

    # Model + eval model configs.
    model_cfg = ws.models_dir / "model.yaml"
    eval_model_cfg = ws.models_dir / "eval_model.yaml"
    _write_model_config(model_cfg, name="test/opt-model")
    _write_model_config(eval_model_cfg, name="test/opt-model")

    # Small JSON dataset so run_optimize can build train/val splits.
    data_file = ws.data_dir / "dataset.json"
    _write_json_data_file(
        data_file,
        [{"prompt": f"example {i}"} for i in range(10)],
    )

    data_cfg = ws.get_data_config("opt_data")
    data_cfg.write_text(
        json.dumps(
            {
                "dataset_name": "opt_data",
                "description": "optimize test dataset",
                "path": str(data_file),
                "format": "json",
            },
            indent=2,
        )
    )

    # Patch LLMClient everywhere it's used in optimize/evaluation flow.
    import specalign.commands.optimize as opt_mod
    import specalign.commands.evaluate as eval_mod
    import specalign.utils as utils_mod

    monkeypatch.setattr(opt_mod, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(eval_mod, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(utils_mod, "LLMClient", FakeLLMClient)

    # Patch GEPA optimize() to avoid heavy optimization logic / network calls.
    class FakeGEPAResult:
        def __init__(self) -> None:
            self.best_candidate = {"system_prompt": "OPTIMIZED SYSTEM PROMPT"}
            self.best_score = 0.9

    def fake_gepa_optimize(
        seed_candidate: dict[str, Any],
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]],
        adapter: Any,
        max_metric_calls: int,
        reflection_lm: str,
        use_wandb: bool,
    ) -> FakeGEPAResult:  # type: ignore[unused-argument]
        # Sanity‑check that we received non‑empty train/val data.
        assert trainset
        assert valset
        assert "system_prompt" in seed_candidate
        return FakeGEPAResult()

    monkeypatch.setattr(opt_mod, "optimize", fake_gepa_optimize)

    # Run with very small sample sizes so the test stays lightweight.
    run_optimize(
        workspace=ws,
        model_config_path=model_cfg,
        data_config_path=data_cfg,
        train_samples=4,
        val_samples=2,
        prompt_number=1,
        max_metric_calls=5,
        reflection_lm="test/reflection-model",
        eval_model_config_path=eval_model_cfg,
        use_wandb=False,
    )

    # A new prompt directory should exist (next number after 1 == 2).
    new_prompt_dir = ws.prompts_dir / "2"
    prompt_file = new_prompt_dir / "prompt.md"
    assert prompt_file.exists()
    assert "OPTIMIZED SYSTEM PROMPT" in prompt_file.read_text()

