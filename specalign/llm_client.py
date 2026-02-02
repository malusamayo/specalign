"""LLM client wrapper."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from dspy import LM

load_dotenv(override=True)


class LLMClient:
    """Wrapper for LLM API calls using dspy."""

    def __init__(self, model_config_path: Optional[Path] = None, model_name: Optional[str] = None):
        """Initialize LLM client.

        Args:
            model_config_path: Path to model configuration YAML file.
            model_name: Direct model name (e.g., 'vertex_ai/gemini-2.5-flash').
                       Used if model_config_path is not provided.
        """
        self.config = {}

        if model_config_path:
            with open(model_config_path) as f:
                config_data = yaml.safe_load(f)
                self.config = config_data.get("model", {})
        elif model_name:
            self.config = {"name": model_name}
        else:
            raise ValueError("Either model_config_path or model_name must be provided")
        
        # Resolve environment variables BEFORE creating LM
        self._resolve_env_vars()
        
        model_name = self.config.get("name")
        if model_name:
            model_name = model_name.strip()  # Remove leading/trailing whitespace
        
        if not model_name:
            raise ValueError("Model name not found in configuration")
        
        kwargs = {}

        if "temperature" in self.config:
            kwargs["temperature"] = self.config["temperature"]
        if "max_tokens" in self.config:
            kwargs["max_tokens"] = self.config["max_tokens"]
        
        # Pass API key to LM if available (dspy/LM uses litellm which reads from env or kwargs)
        if "api_key" in self.config and self.config["api_key"]:
            # Set environment variable for litellm to pick up
            os.environ["OPENAI_API_KEY"] = self.config["api_key"]

        self.lm = LM(model_name, **kwargs)

    def _resolve_env_vars(self) -> None:
        """Resolve environment variables in config values."""
        for key, value in self.config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                self.config[key] = os.getenv(env_var, "")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated text response.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.lm(messages=messages)[0]
        return response

    @staticmethod
    def create_default_client(model_name: str = "vertex_ai/gemini-2.5-flash") -> "LLMClient":
        """Create a default LLM client with specified model.

        Args:
            model_name: Model identifier for litellm.

        Returns:
            Configured LLMClient instance.
        """
        return LLMClient(model_name=model_name)
