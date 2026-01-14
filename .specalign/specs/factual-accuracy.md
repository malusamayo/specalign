# factual-accuracy Specification

## Purpose
Ensures only provided and accurate product attributes are used; prohibits fabricated or omitted data.

## Requirements

### Requirement: Use Provided Product Attributes ONLY

All facts, stats, and data in output SHALL come only from the provided product attribute values (like NAME_LONG, BRAND, FORMAT, etc.).

#### Scenario: Product attribute missing or not specified

- **WHEN** generating content
- **THEN** only use provided attributes, and omit (or edit out) any "NO_VALUE_FOUND".

### Requirement: No Health Claims

Do not mention health or health benefits; output SHALL exclude any references to health properties or effects.

#### Scenario: Opportunity to promote health benefits

- **WHEN** generating product text
- **THEN** refrain from any references to health attributes, effects, or implications.

### Requirement: No Value Inference

Output SHALL not infer properties not explicitly present in the attribute data (such as quality, popularity, etc.).

#### Scenario: Potential for property inference

- **WHEN** tempted to infer or speculate about non-provided attributes
- **THEN** avoid and compose output strictly from provided facts.

## Evaluation Guidelines

### Pass Criteria

A response **passes** if it:
- Includes only factual information supplied in the product attributes.
- Omits all health-related claims or statements.
- Omits or edits out any attribute marked as "NO_VALUE_FOUND", keeping only valid attributes.

### Fail Criteria

A response **fails** if it:
- Invents or infers data not supplied.
- Mentions health or health benefits.
- Includes "NO_VALUE_FOUND" tokens or edited sentences are incoherent after removal.

## Examples

### Good Example

`<p>PRODNAME by BRANDNAME is a FORMAT that can be used NUMBER_OF_USES times.</p>`

### Bad Example

`<p>PRODNAME is a great, refreshing product that boosts your health.</p>`
(Makes health claims, value judgements, and uses non-supplied data.)