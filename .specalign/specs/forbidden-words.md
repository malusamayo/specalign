# forbidden-words Specification

## Purpose
Governs the exclusion of specified words, their inflections, and synonyms from the output; mandates list-based checking and enforcement.

## Requirements

### Requirement: No Use of Forbidden Words

The output SHALL NOT use any of the forbidden words or their inflected (grammatical variances) or synonym forms, including but not limited to: excellent, better, satisfy, sensational, delicious, great, convenient, exquisite, fulfilling, boost, enjoy, needs, unique, freshness, refreshing, light, cool, regret, mouth-watering, passion, pleasant, perfect, fresh, quality, punsch, epic, craving, smooth, exciting, safe, mild, amazing, safer, magic, popular, impact, users, natural, clean, low, trendy, ideal, juicy, fix, satisfy, newcomer.

#### Scenario: Attempt to use forbidden/related words

- **WHEN** composing the product description
- **THEN** detect and exclude forbidden words and their inflections and synonyms.

### Requirement: No Embellishment or Synonym Substitution

Forbidden words SHALL NOT be replaced with synonyms, paraphrased, or otherwise artificially avoided.

#### Scenario: Use of similar language

- **WHEN** tempted to use a prohibited word via synonym or paraphrase
- **THEN** do not use any such substitutes.

## Evaluation Guidelines

### Pass Criteria

A response **passes** if it:
- Contains none of the forbidden words, their inflections, or synonyms.
- Is factual with no value-laden descriptors related to forbidden words.

### Fail Criteria

A response **fails** if it:
- Contains any forbidden word or inflection thereof.
- Replaces forbidden terms with close synonyms.

## Examples

### Good Example

`<p>PRODNAME is available in a range of flavours and offers NUMBER_OF_USES uses per package.</p>`

### Bad Example

`<p>This delicious product is ideal for users looking for a refreshing experience.</p>`
(Contains multiple forbidden words and forms.)