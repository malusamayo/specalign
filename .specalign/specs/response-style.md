# response-style Specification

## Purpose
Defines tone, language, and the manner of expression in generating product descriptions.

## Requirements

### Requirement: British English Usage

Output SHALL use British English conventions and spelling.

#### Scenario: System needs to generate product text

- **WHEN** generating the text
- **THEN** all spelling and usage SHALL be British English.

### Requirement: Tone

The language SHALL be formal, professional, and informative.

#### Scenario: Rendering product descriptions

- **WHEN** constructing descriptions
- **THEN** keep to a neutral, professional, and factual style (not casual or slang).

### Requirement: Fact-Based description

Descriptions SHALL be fact-based, particularly regarding the brand and flavours.

#### Scenario: Brand or flavour descriptions

- **WHEN** describing brand or flavours
- **THEN** do not embellish, glorify, or make value judgements; stick to factual description.

### Requirement: No Value Judgements or Superlatives

Output SHALL be neutral and SHALL NOT contain value judgements, emotional, comparative or superlative language.

#### Scenario: Temptation to use praise or evaluation

- **WHEN** expressing any opinion, assessment, or value
- **THEN** avoid and omit such statements; only present facts.

## Evaluation Guidelines

### Pass Criteria
A response **passes** if it:
- Uses exclusively British English spelling and conventions.
- Maintains a consistently formal, professional, and informative style.
- Limits all brand/flavour mentions to factual statements.
- Omits any form of embellishment, value judgement, or superlative.

### Fail Criteria
A response **fails** if it:
- Uses American English or slang.
- Slips into casual or promotional tones.
- Embellishes or makes evaluative/headline statements.
- Adds commentary or emotional language.

## Examples

### Good Example

`<p>PRODNAME by BRANDNAME is a FORMAT designed for ...</p>`  
(Uses British English, neutral).

### Bad Example

`<p>This excellent product offers a sensational experience for all users.</p>`  
(Promotional, uses forbidden words, not neutral/factual.)