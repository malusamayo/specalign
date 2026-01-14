# seo-content Specification

## Purpose
Ensures the product description is creative, unique, and keyword-rich, adhering to SEO best practices without violating factuality or tone constraints.

## Requirements

### Requirement: Keyword-Richness and Uniqueness

Each section of the description SHALL be unique and include relevant keywords related to the product, while remaining within all other style and content constraints.

#### Scenario: Generating multiple product descriptions

- **WHEN** outputting 6 paragraphs with headers
- **THEN** each paragraph SHALL be uniquely worded, creative, and contain product-relevant keywords for SEO.

### Requirement: Avoid Duplication

Descriptions SHALL NOT reuse phrases verbatim between segments, except where mandated by product attributes.

#### Scenario: Multiple segments with similar content

- **WHEN** similar product information must be mentioned in several segments
- **THEN** rephrase content to maximise SEO uniqueness.

## Evaluation Guidelines

### Pass Criteria

A response **passes** if it:
- Contains multiple unique, keyword-rich paragraphs.
- Avoids copy-pasted text across segments.
- Complies with all other specifications (style, factuality, forbidden words).

### Fail Criteria

A response **fails** if it:
- Repeats significant text or keywords across segments.
- Lacks product-specific SEO keywords.
- Copies generic or filler text across sections.

## Examples

### Good Example

`<h4>Characteristics for Usage</h4><p>Engineered for reliability, PRODNAME by BRANDNAME comes in FORMAT and is suitable for NUMBER_OF_USES applications.</p>`

(SEO keywords: engineered, reliability, suitable for usage, integrated naturally.)

### Bad Example

`<h4>About PRODNAME</h4><p>This product is great. Product is great. Product is great.</p>`
(Repetitive, generic, not unique, lacks SEO optimisation.)