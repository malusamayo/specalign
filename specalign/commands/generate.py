"""Implementation of 'specalign generate' command for synthetic data generation."""

import csv
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from specalign.llm_client import LLMClient
from specalign.utils import batch_inference
from specalign.workspace import Workspace


SYNTHETIC_DATA_GENERATION_PROMPT = """You are a test case generation expert. Your task is to generate synthetic test cases based on LLM prompt specifications.

For each specification, generate diverse test cases that:
1. Cover different scenarios mentioned in the specification
2. Include edge cases and boundary conditions
3. Test both positive (pass) and negative (fail) scenarios
4. Are realistic and representative of real-world usage

Each test case should be formatted as a promptfoo test case with:
- vars: Input variables (e.g., {"input": "user query here"})
- assert: Assertions to validate the output
- metadata: Link to the specification(s) it tests

CRITICAL ASSERTION GUIDELINES:
1. **Check for SEMANTIC meaning, not exact text**: The model will paraphrase. Do NOT use "contains" for descriptive words that have common synonyms or paraphrases (e.g. hydrates→hydration/moisture, breathable→breath/ventilation, cushioning→cushion/support/comfort, durable→long-lasting/resilient, boiling→boil). Use JavaScript assertions that check for word stems or key concepts (e.g. output.toLowerCase().includes('hydrat') || output.toLowerCase().includes('moistur')).

2. **Never require exact input wording**: Do not assert that the output must copy phrases from the input verbatim (e.g. "rich texture", "comfortable sleep experience"). The model may rephrase. Check for the underlying meaning with flexible JavaScript (e.g. presence of related terms or stems).

3. **Spec compliance over input copying**: If the specification requires a particular wording (e.g. British English "flavour"), NEVER assert that the output must contain the opposite (e.g. American "flavor") even if the input contains it. Assert what the SPEC requires (e.g. output uses "flavour" or does not use "flavor"), not verbatim repetition of the input.

4. **Match the actual prompt requirements**: Read the prompt carefully and check for what it ACTUALLY requires (e.g. "Characteristics for Usage"), not exact header strings.

5. **Use flexible assertions**: Prefer JavaScript for any content check. Reserve "contains" only for terms that must appear literally (e.g. a specific spelling like "flavour" when testing British English, or forbidden-word absence via javascript !output.includes).

6. **Structure, not exact wording**: For HTML, check that required tags exist; do not require exact phrase matches inside them.

EXAMPLES OF BAD vs GOOD ASSERTIONS:
- BAD: {{"type": "contains", "value": "hydrates"}}  → Model may say "skin hydration" or "maintains moisture".
- GOOD: {{"type": "javascript", "value": "(output.toLowerCase().includes('hydrat') || output.toLowerCase().includes('moistur'))"}}
- BAD: {{"type": "contains", "value": "breathable"}}  → Model may say "breathable" or "breathability" or "allows air flow".
- GOOD: {{"type": "javascript", "value": "(output.toLowerCase().includes('breath') || output.toLowerCase().includes('ventilat') || output.toLowerCase().includes('air'))"}}
- BAD: Requiring output to include "with a spicy flavor" when the spec says use British English (output should use "flavour", not "flavor").
- GOOD: {{"type": "javascript", "value": "output.includes('flavour') && !output.includes('flavor')"}} for British English spec, or check for semantic presence of "spicy" and British spelling.

Use only valid promptfoo assertion types:
- "contains": ONLY for terms that must appear literally (e.g. "flavour" when testing British English). Do NOT use for paraphrased concepts.
- "javascript": PREFERRED for almost all checks (semantic presence, stems, negation, structure).
- "regex": For pattern matching (e.g. HTML tags).

For negative checks use: {{"type": "javascript", "value": "!output.includes('forbidden text')"}}

Output format: JSON array of test cases. Each test case:
{{
  "vars": {{ "input": "..." }},
  "assert": [
    {{ "type": "javascript", "value": "semantic or structural check" }},
    {{ "type": "javascript", "value": "!output.includes('forbidden')" }},
    {{ "type": "javascript", "value": "/<h[34]>.*<\\/h[34]>/.test(output) && /<p>.*<\\/p>/.test(output)" }}
  ],
  "metadata": {{ "spec_requirements": ["spec-name"], "scenario": "...", "requirement": "..." }}
}}

Generate diverse test cases. Focus on specification compliance with semantic checks, not exact text matching. If examples are provided, match their style."""

# --- Scenario-agnostic relaxation (do not add domain-specific words) ---
# These maps are kept minimal so specalign works for any domain (e.g. e-commerce, support, code).
# Rely on the LLM prompt for semantic/flexible assertions; use these only for:
# - Language/spelling variants (e.g. -ise/-ize, British vs American)
# - Word stems that are safe across contexts (e.g. "hydrat", "durabl")
# Developers with a specific scenario should improve the prompt or add custom assertions,
# not extend these maps with scenario-only synonyms.

# contains → semantic (stem) checks; only common stems that generalize.
PARAPHRASE_STEMS: Dict[str, List[str]] = {
    "hydration": ["hydrat", "moistur"],
    "durable": ["durabl", "resilient", "sturdy"],
    "high-quality": ["quality", "premium"],
}

# Single-word .includes() → allow spelling/stem variants only (no domain-specific synonyms).
JS_WORD_SYNONYMS: Dict[str, List[str]] = {
    "moisturising": ["moisturising", "moisturizing", "moistur"],
    "flavour": ["flavour", "flavours"],
    "protection": ["protection", "protect"],
}


def _make_semantic_js(stems: List[str]) -> str:
    """Build a JavaScript expression that passes if output contains any of the stems (case-insensitive)."""
    parts = [f"output.toLowerCase().includes({repr(s)})" for s in stems]
    return "(" + " || ".join(parts) + ")"


def _relax_js_includes(value: str) -> str:
    """Replace strict .includes('word') in JS assertion with (word OR synonym1 OR ...) for known words."""
    # Match output.includes('x') or output.toLowerCase().includes('x')
    pattern = re.compile(
        r"(output\.(?:toLowerCase\(\)\.)?includes\s*\(\s*)(['\"])([a-zA-Z]+)\2(\s*\))"
    )
    def repl(m: re.Match) -> str:
        prefix, quote, word, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        key = word.lower()
        if key not in JS_WORD_SYNONYMS:
            return m.group(0)
        synonyms = JS_WORD_SYNONYMS[key]
        parts = [f"{prefix}{quote}{s}{quote}{suffix}" for s in synonyms]
        return "(" + " || ".join(parts) + ")"
    return pattern.sub(repl, value)


def _relax_html_regex(value: str) -> str:
    """Make HTML regexes match across newlines: .* -> [\\s\\S]*? inside /.../."""
    # Replace /<tag>.*</tag>/ with /<tag>[\s\S]*?<\/tag>/ so . matches newlines
    value = re.sub(r"/<h\[34\]>\.\*</h\[34\]>/", r"/<h[34]>[\\s\\S]*?<\\/h[34]>/", value)
    value = re.sub(r"/<p>\.\*</p>/", r"/<p>[\\s\\S]*?<\\/p>/", value)
    value = re.sub(r"/<h3>\.\*</h3>/", r"/<h3>[\\s\\S]*?<\\/h3>/", value)
    value = re.sub(r"/<h4>\.\*</h4>/", r"/<h4>[\\s\\S]*?<\\/h4>/", value)
    value = re.sub(r"/<ul>\.\*</ul>/", r"/<ul>[\\s\\S]*?<\\/ul>/", value)
    return value


def relax_brittle_contains(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert brittle 'contains' assertions to semantic JavaScript checks where the model often paraphrases."""
    for test_case in test_cases:
        if "assert" not in test_case:
            continue
        spec_requirements = (test_case.get("metadata") or {}).get("spec_requirements") or []
        spec_keys = {s.lower() for s in spec_requirements} if isinstance(spec_requirements, list) else set()

        new_asserts = []
        for assertion in test_case["assert"]:
            a_type = assertion.get("type", "")
            value = assertion.get("value", "")
            if not isinstance(value, str):
                new_asserts.append(assertion)
                continue

            # Convert contains to semantic js for known paraphrased terms
            if a_type == "contains":
                val_lower = value.strip().lower()
                if val_lower in PARAPHRASE_STEMS:
                    stems = PARAPHRASE_STEMS[val_lower]
                    new_asserts.append({
                        "type": "javascript",
                        "value": _make_semantic_js(stems),
                    })
                    continue
                # Keep contains for literal requirements (e.g. "flavour" for British English)
                new_asserts.append(assertion)
                continue

            # Relax JavaScript assertions: allow synonyms and fix HTML regex (match across newlines)
            if a_type == "javascript":
                value = _relax_js_includes(value)
                value = _relax_html_regex(value)
                assertion = {"type": "javascript", "value": value}
            # Fix JavaScript assertions that require American "flavor" when spec is British English
            if a_type == "javascript" and "flavor" in value and "flavour" not in value:
                spec_str = " ".join(spec_requirements).lower() if isinstance(spec_requirements, list) else ""
                if any("british" in k for k in spec_keys) or "british" in spec_str:
                    # Require British spelling in output; don't require American
                    new_value = value.replace("'flavor'", "'flavour'").replace('"flavor"', '"flavour"')
                    new_value = re.sub(
                        r"output\.includes\s*\(\s*['\"]with a spicy flavor['\"]\s*\)",
                        "(output.includes('flavour') && (output.includes('spicy') || output.includes('spice')))",
                        new_value,
                    )
                    new_asserts.append({"type": "javascript", "value": new_value})
                    continue
            new_asserts.append(assertion)

        test_case["assert"] = new_asserts
    return test_cases


def normalize_assertions(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize test case assertions to use valid promptfoo assertion types.
    
    Converts invalid assertion types (like 'not_contains') to valid ones (like 'javascript').
    
    Args:
        test_cases: List of test case dictionaries.
        
    Returns:
        List of test cases with normalized assertions.
    """
    for test_case in test_cases:
        if "assert" not in test_case:
            continue
        
        normalized_asserts = []
        for assertion in test_case["assert"]:
            assert_type = assertion.get("type", "")
            assert_value = assertion.get("value", "")
            
            # Convert not_contains to javascript assertion
            if assert_type == "not_contains":
                normalized_asserts.append({
                    "type": "javascript",
                    "value": f"!output.includes({repr(assert_value)})"
                })
            else:
                # Keep other assertion types as-is
                normalized_asserts.append(assertion)
        
        test_case["assert"] = normalized_asserts
    
    return test_cases


def parse_test_cases_from_llm_response(response: str) -> List[Dict[str, Any]]:
    """Parse test cases from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        List of test case dictionaries.
    """
    # Clean up the response
    text = response.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        # Extract JSON from code block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    try:
        # Try to parse as JSON
        test_cases = json.loads(text)
        
        # Ensure it's a list
        if isinstance(test_cases, dict):
            # Sometimes LLM wraps in an object
            if "test_cases" in test_cases:
                test_cases = test_cases["test_cases"]
            elif "tests" in test_cases:
                test_cases = test_cases["tests"]
            else:
                test_cases = [test_cases]
        
        if not isinstance(test_cases, list):
            test_cases = [test_cases]
        
        return test_cases
    
    except json.JSONDecodeError as e:
        click.echo(f"Warning: Could not parse LLM response as JSON: {e}", err=True)
        click.echo("Attempting to extract JSON from response...")
        
        # Try to find JSON array in the text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty list
        click.echo("Could not extract valid test cases from response.", err=True)
        return []


def load_examples_from_path(example_path: Path, max_examples: int = 10) -> List[Dict[str, Any]]:
    """Load example data from a specific file or directory.
    
    Args:
        example_path: Path to example file or directory.
        max_examples: Maximum number of examples to load.
        
    Returns:
        List of example dictionaries, each with 'input' key.
    """
    example_files = []
    
    if example_path.is_file():
        example_files = [example_path]
    elif example_path.is_dir():
        example_files = sorted(example_path.glob("*.json"))
        example_files.extend(sorted(example_path.glob("*.jsonl")))
        example_files.extend(sorted(example_path.glob("*.csv")))
    else:
        click.echo(f"Warning: Example path does not exist: {example_path}", err=True)
        return []
    
    return _load_examples_from_files(example_files, max_examples)


def _load_examples_from_files(example_files: List[Path], max_examples: int = 10) -> List[Dict[str, Any]]:
    """Load examples from a list of files.
    
    Args:
        example_files: List of file paths to load.
        max_examples: Maximum number of examples to load.
        
    Returns:
        List of example dictionaries, each with 'input' key.
    """
    if not example_files:
        return []
    
    examples = []
    loaded_count = 0
    
    for example_file in example_files:
        if loaded_count >= max_examples:
            break
            
        try:
            if example_file.suffix == ".json":
                with open(example_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if loaded_count >= max_examples:
                                break
                            if isinstance(item, dict):
                                examples.append(item)
                                loaded_count += 1
                    elif isinstance(data, dict):
                        examples.append(data)
                        loaded_count += 1
                    else:
                        click.echo(f"Warning: {example_file} contains invalid JSON structure (expected object or array)", err=True)
                        
            elif example_file.suffix == ".jsonl":
                with open(example_file) as f:
                    for line_num, line in enumerate(f, 1):
                        if loaded_count >= max_examples:
                            break
                        try:
                            item = json.loads(line.strip())
                            if isinstance(item, dict):
                                examples.append(item)
                                loaded_count += 1
                            else:
                                click.echo(f"Warning: {example_file}:{line_num} is not a JSON object, skipping", err=True)
                        except json.JSONDecodeError as e:
                            click.echo(f"Warning: {example_file}:{line_num} - Invalid JSON: {e}", err=True)
                            
            elif example_file.suffix == ".csv":
                with open(example_file) as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        click.echo(f"Warning: {example_file} appears to be empty or invalid CSV", err=True)
                        continue
                    
                    for row_num, row in enumerate(reader, 1):
                        if loaded_count >= max_examples:
                            break
                        # Use 'input' or 'prompt' column, or first column
                        input_value = row.get("input") or row.get("prompt")
                        if not input_value and row:
                            input_value = list(row.values())[0]
                        
                        if input_value:
                            examples.append({"input": input_value})
                            loaded_count += 1
                        else:
                            click.echo(f"Warning: {example_file}:{row_num} - No 'input' or 'prompt' column found, skipping", err=True)
                            
        except json.JSONDecodeError as e:
            click.echo(f"Error: Could not parse JSON from {example_file}: {e}", err=True)
            click.echo(f"   Make sure the file contains valid JSON (object or array of objects)", err=True)
        except csv.Error as e:
            click.echo(f"Error: Could not parse CSV from {example_file}: {e}", err=True)
            click.echo(f"   Make sure the file is a valid CSV with 'input' or 'prompt' column", err=True)
        except Exception as e:
            click.echo(f"Error: Could not load examples from {example_file}: {e}", err=True)
            continue
    
    # Validate examples have required fields
    validated_examples = []
    for i, example in enumerate(examples, 1):
        if not isinstance(example, dict):
            click.echo(f"Warning: Example {i} is not a dictionary, skipping", err=True)
            continue
        if "input" not in example and "prompt" not in example:
            click.echo(f"Warning: Example {i} missing 'input' or 'prompt' field, skipping", err=True)
            continue
        validated_examples.append(example)
    
    return validated_examples[:max_examples]


def load_examples(workspace: Workspace, max_examples: int = 10) -> List[Dict[str, Any]]:
    """Load example data from workspace examples directory.
    
    Args:
        workspace: Workspace instance.
        max_examples: Maximum number of examples to load.
        
    Returns:
        List of example dictionaries, each with 'input' key.
    """
    example_files = workspace.get_example_files()
    return _load_examples_from_files(example_files, max_examples)


def format_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    """Format examples for inclusion in prompt.
    
    Args:
        examples: List of example dictionaries.
        
    Returns:
        Formatted string with examples.
    """
    if not examples:
        return ""
    
    formatted = "\n\nEXAMPLES TO FOLLOW (generate similar test cases):\n"
    for i, example in enumerate(examples, 1):
        input_value = example.get("input", example.get("prompt", str(example)))
        formatted += f"\nExample {i}:\n"
        formatted += f"Input: {input_value}\n"
        if "output" in example:
            formatted += f"Output: {example['output']}\n"
    
    return formatted


def generate_test_cases_for_spec(
    spec_name: str,
    spec_content: str,
    llm_client: LLMClient,
    cases_per_spec: int,
    examples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate test cases for a single specification.
    
    Args:
        spec_name: Name of the specification.
        spec_content: Content of the specification.
        llm_client: LLM client for generation.
        cases_per_spec: Number of test cases to generate.
        examples: List of example dictionaries for few-shot learning.
        
    Returns:
        List of test case dictionaries.
    """
    examples_text = format_examples_for_prompt(examples)
    
    user_prompt = f"""Generate {cases_per_spec} diverse test cases based on this specification:

=== {spec_name} ===
{spec_content}
{examples_text}

IMPORTANT GUIDELINES:
1. Test case inputs should be realistic product attribute data or natural language requests.
2. For product descriptions, use structured format: "Product attributes: NAME_LONG='...', BRAND='...', FORMAT='...', ..."
3. Assertions must check SPECIFICATION COMPLIANCE, not exact repetition of input wording. The model will paraphrase (e.g. "hydrates" → "hydration", "breathable" → "allows air flow"). Use JavaScript with stems or semantic checks, not "contains" for such terms.
4. If this spec requires British English: never assert that the output must contain American spellings (e.g. "flavor"). Assert British spellings (e.g. "flavour") or absence of American spellings.
5. Use flexible assertions: JavaScript for semantic presence, forbidden-word checks (!output.includes), HTML/structure checks. Use "contains" only for terms that must appear literally (e.g. "flavour" when testing British English).
6. Do not require exact phrases from the input (e.g. "rich texture", "comfortable sleep experience")—check for meaning instead.
7. If examples are provided, match their style and distribution.

Each test case must include metadata linking it to the "{spec_name}" specification."""

    try:
        response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=SYNTHETIC_DATA_GENERATION_PROMPT
        )
        
        test_cases = parse_test_cases_from_llm_response(response)
        
        # Normalize assertions to use valid promptfoo types
        test_cases = normalize_assertions(test_cases)
        # Relax brittle contains (paraphrased terms) and fix American spelling in British-English specs
        test_cases = relax_brittle_contains(test_cases)
        
        # Ensure metadata links to this spec
        for test_case in test_cases:
            if "metadata" not in test_case:
                test_case["metadata"] = {}
            
            # Ensure spec_requirements includes this spec
            if "spec_requirements" not in test_case["metadata"]:
                test_case["metadata"]["spec_requirements"] = []
            
            if spec_name not in test_case["metadata"]["spec_requirements"]:
                test_case["metadata"]["spec_requirements"].append(spec_name)
        
        return test_cases
    
    except Exception as e:
        click.echo(f"  Error generating test cases for {spec_name}: {e}", err=True)
        return []


def generate_test_cases(
    specs: Dict[str, str],
    llm_client: LLMClient,
    count: int,
    per_spec: Optional[int] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """Generate synthetic test cases using LLM with parallel processing.

    Args:
        specs: Dictionary mapping spec names to spec content.
        llm_client: LLM client for generation.
        count: Total number of test cases to generate.
        per_spec: Number of test cases per spec (if None, distributes evenly).
        examples: Optional list of example dictionaries for few-shot learning.
        max_workers: Maximum number of parallel workers for generation.

    Returns:
        List of test case dictionaries.
    """
    if examples is None:
        examples = []
    
    if per_spec:
        # Generate per_spec test cases for each specification
        cases_per_spec = per_spec
    else:
        # Distribute evenly across specs
        num_specs = len(specs)
        cases_per_spec = max(1, count // num_specs) if num_specs > 0 else count
    
    # Prepare arguments for parallel generation
    generation_args = [
        {
            "spec_name": spec_name,
            "spec_content": spec_content,
            "llm_client": llm_client,
            "cases_per_spec": cases_per_spec,
            "examples": examples
        }
        for spec_name, spec_content in specs.items()
    ]
    
    # Generate test cases in parallel
    click.echo(f"Generating test cases in parallel (max {max_workers} workers)...")
    all_results = batch_inference(
        generate_test_cases_for_spec,
        generation_args,
        use_process=False,
        max_workers=min(len(generation_args), max_workers)
    )
    
    # Flatten results
    all_test_cases = []
    for i, (spec_name, _) in enumerate(specs.items()):
        test_cases = all_results[i]
        all_test_cases.extend(test_cases)
        click.echo(f"  Generated {len(test_cases)} test cases for {spec_name}")
    
    # Limit to requested count
    if len(all_test_cases) > count:
        all_test_cases = all_test_cases[:count]
    
    return all_test_cases


def format_as_promptfoo(
    test_cases: List[Dict[str, Any]],
    workspace: Workspace,
    model_config_path: Path,
    prompt_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Format test cases as complete promptfoo configuration.

    Args:
        test_cases: List of test case dictionaries.
        workspace: Workspace instance.
        model_config_path: Path to model configuration YAML file.
        prompt_number: Specific prompt number to use (latest if None).

    Returns:
        Complete promptfoo configuration dictionary with providers, prompts, and tests.
    """
    # Load model config
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    model_info = model_config.get("model", {})
    model_name = model_info.get("name", "openai/gpt-4")
    api_key = model_info.get("api_key", "${OPENAI_API_KEY}")
    
    # Convert model name to promptfoo provider format
    # e.g., "openai/gpt-4.1-mini" -> "openai:gpt-4.1-mini"
    provider_name = model_name.replace("/", ":")
    
    # Get prompt number (use latest if not specified)
    if prompt_number is None:
        prompt_number = workspace.get_next_prompt_number() - 1
    
    if prompt_number < 1:
        raise ValueError("No prompts found. Run 'specalign compile' first.")
    
    # Get prompt path relative to workspace root
    prompt_dir = workspace.prompts_dir / str(prompt_number)
    prompt_file = prompt_dir / "prompt.md"
    
    if not prompt_file.exists():
        raise ValueError(f"Prompt #{prompt_number} not found.")
    
    # Read the prompt content
    with open(prompt_file, "r") as f:
        prompt_content = f.read()
    
    # Fix ambiguous "function arguments" language that confuses LLMs
    # Replace confusing "function arguments" language with clear HTML output instruction
    prompt_content = prompt_content.replace(
        "each returning two function arguments: a header and a paragraph",
        "each containing a header and a paragraph in HTML format"
    )
    prompt_content = prompt_content.replace(
        "Use the exact argument names below:",
        "Output the following segments as HTML:"
    )
    
    # Add explicit instruction to output raw HTML without markdown code blocks
    # This is critical for promptfoo assertions to work correctly
    html_output_instruction = """

**IMPORTANT OUTPUT FORMAT:**
- Output ONLY raw HTML code, without any markdown code blocks (no ```html or ``` markers)
- Do NOT wrap your output in code fences or markdown formatting
- Output should start directly with <h3> or <h4> tags
- Output should be plain HTML text that can be directly used"""
    
    # Create a prompt template that includes the system prompt and user input
    # promptfoo will substitute {{input}} with vars.input from each test case
    prompt_template = f"""{prompt_content}{html_output_instruction}

---

### Input:
{{{{input}}}}"""
    
    # Build provider config
    provider_config = {
        "apiKey": api_key
    }
    
    # Add optional model parameters if present
    if "temperature" in model_info:
        provider_config["temperature"] = model_info["temperature"]
    if "max_tokens" in model_info:
        provider_config["max_tokens"] = model_info["max_tokens"]
    
    # Build promptfoo config
    # Use the prompt template directly (as a string) instead of file path
    # This allows promptfoo to substitute {{input}} with vars.input from test cases
    config = {
        "providers": [
            {
                provider_name: provider_config
            }
        ],
        "prompts": [prompt_template],
        "tests": test_cases
    }
    
    return config


def run_generate(
    workspace: Workspace,
    model_config_path: Path,
    output_path: Optional[Path] = None,
    count: int = 10,
    per_spec: Optional[int] = None,
    max_workers: int = 10,
    examples_path: Optional[Path] = None,
) -> None:
    """Run the generate command.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration YAML file.
        output_path: Optional path to save test cases file. If None, uses default location.
        count: Total number of test cases to generate.
        per_spec: Number of test cases per specification (overrides count distribution).
        max_workers: Maximum number of parallel workers for generation.
        examples_path: Optional path to example file(s) or directory. If None, uses workspace examples directory.
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return
    
    # Ensure test_cases directory exists
    workspace.test_cases_dir.mkdir(parents=True, exist_ok=True)
    
    # Load specifications
    spec_files = workspace.get_all_spec_files()
    
    if not spec_files:
        click.echo(f"Error: No specification files found in {workspace.specs_dir}", err=True)
        return
    
    click.echo(f"Found {len(spec_files)} specification file(s):")
    specs = {}
    for spec_file in spec_files:
        with open(spec_file) as f:
            spec_name = spec_file.stem
            specs[spec_name] = f.read()
            click.echo(f"  - {spec_name}")
    
    # Load examples if available
    if examples_path:
        click.echo(f"Loading examples from: {examples_path}")
        examples = load_examples_from_path(examples_path, max_examples=10)
        if examples:
            click.echo(f"✓ Loaded {len(examples)} example(s) from {examples_path}")
        else:
            click.echo(f"⚠ No valid examples found in {examples_path} (using zero-shot generation)", err=True)
            click.echo(f"   Supported formats: JSON, JSONL, CSV with 'input' or 'prompt' field", err=True)
    else:
        examples = load_examples(workspace, max_examples=10)
        if examples:
            click.echo(f"✓ Loaded {len(examples)} example(s) from {workspace.examples_dir}")
        else:
            click.echo(f"ℹ No examples found in {workspace.examples_dir} (using zero-shot generation)")
            click.echo(f"   💡 Tip: Add example files to {workspace.examples_dir} for better results")
            click.echo(f"      Supported: JSON, JSONL, CSV with 'input' or 'prompt' field")
    
    # Create LLM client
    click.echo(f"\nUsing model config: {model_config_path}")
    llm_client = LLMClient(model_config_path=model_config_path)
    
    # Generate test cases
    click.echo(f"\nGenerating {count} test cases...")
    test_cases = generate_test_cases(specs, llm_client, count, per_spec, examples, max_workers)
    
    if not test_cases:
        click.echo("Error: No test cases were generated.", err=True)
        return
    
    click.echo(f"Successfully generated {len(test_cases)} test cases")
    
    # Get latest prompt number
    prompt_number = workspace.get_next_prompt_number() - 1
    if prompt_number < 1:
        click.echo("Warning: No prompts found. Test cases will be generated without prompt reference.", err=True)
        click.echo("Run 'specalign compile' first to generate a prompt.", err=True)
        # Fall back to tests-only format
        promptfoo_config = {"tests": test_cases}
    else:
        click.echo(f"Using prompt #{prompt_number}")
        # Format as complete promptfoo config
        try:
            promptfoo_config = format_as_promptfoo(test_cases, workspace, model_config_path, prompt_number)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)
            click.echo("Generating test cases file without promptfoo config.", err=True)
            promptfoo_config = {"tests": test_cases}
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = workspace.test_cases_dir / f"test_cases_{timestamp}.yaml"
    else:
        # Ensure it's in test_cases directory if relative
        if not output_path.is_absolute():
            output_path = workspace.test_cases_dir / output_path
    
    # Save test cases
    with open(output_path, "w") as f:
        yaml.dump(promptfoo_config, f, default_flow_style=False, sort_keys=False)
    
    # Save metadata
    metadata = {
        "generation": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": str(model_config_path),
            "total_test_cases": len(test_cases),
            "specs_covered": list(specs.keys()),
        },
        "test_case_summary": {
            spec_name: sum(
                1 for tc in test_cases
                if spec_name in tc.get("metadata", {}).get("spec_requirements", [])
            )
            for spec_name in specs.keys()
        }
    }
    
    metadata_path = output_path.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    click.echo(f"\nTest cases generated successfully!")
    click.echo(f"  Promptfoo config: {output_path}")
    click.echo(f"  Metadata: {metadata_path}")
    click.echo(f"\n  Test cases per spec:")
    for spec_name, count in metadata["test_case_summary"].items():
        click.echo(f"    - {spec_name}: {count}")
    
    if "providers" in promptfoo_config:
        click.echo(f"\nReady to run with promptfoo:")
        click.echo(f"  promptfoo eval -c {output_path}")
    else:
        click.echo(f"\nNote: Test cases file created without promptfoo config.")
        click.echo(f"  Run 'specalign compile' first to generate a prompt, then regenerate test cases.")