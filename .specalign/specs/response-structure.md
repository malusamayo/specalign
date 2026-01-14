# response-structure Specification

## Purpose
Governs the format, structure, segmenting, and function argument assignment of the output, as well as enumeration of content sections.

## Requirements

### Requirement: Output Segments and Arguments

The response SHALL generate product descriptions divided into the following segments with the specified content:

- `segment_1`: Header and paragraph about the product ("About PRODNAME").
- `segment_2`: Header and paragraph regarding product characteristics for usage, including strength and total number of uses.
- `segment_3`: Header and paragraph about flavour(s).
- `segment_5`: Header and paragraph summarising product (category, flavour(s), strength, uses, manufacturer, more about brand).
- `segment_short`: Header and paragraph for a short description specific to the brand.

Each segment SHALL return header and paragraph as function arguments, named as above.

#### Scenario: User requests a product description

- **WHEN** generating the product description
- **THEN** the output SHALL be split into the prescribed segments and assigned as function arguments, with each segment containing only its designated content.

### Requirement: Paragraph and Header Formatting

Headers SHALL use the correct HTML header elements as specified. Paragraphs SHALL use `<p>` elements or `<ul>` where required.

#### Scenario: Outputting a section header and text

- **WHEN** outputting a segment header and paragraph
- **THEN** 
  - The header SHALL be wrapped in the specified `<h3>` or `<h4>` element.
  - The paragraph SHALL be in a `<p>`, or `<ul>` if specified.

### Requirement: Flavour List Deduplication

Flavour lists SHALL remove duplicate values before outputting.

#### Scenario: Flavour list contains duplicate values

- **WHEN** generating the flavour section
- **THEN** flavour entries SHALL be deduplicated.

### Requirement: NO_VALUE_FOUND Handling

Any attribute with the value "NO_VALUE_FOUND" SHALL be omitted, and the text SHALL be adjusted to remain coherent.

#### Scenario: A product attribute is "NO_VALUE_FOUND"

- **WHEN** processing the attribute
- **THEN** 
  - "NO_VALUE_FOUND" SHALL be removed,
  - The sentence or segment SHALL be edited for coherence.

### Requirement: Quotation Mark Policy

Outputs SHALL NOT be wrapped in quote marks unless quotes appear in the input.

#### Scenario: Outputs without quotes in instructions

- **WHEN** rendering product descriptions or text
- **THEN** Do not wrap in `"` or `'`, unless present in input.

## Evaluation Guidelines

### Pass Criteria
A response **passes** if it:
- Returns all prescribed segments named as specified, each with relevant content.
- Wraps headers and paragraphs in the correct HTML elements.
- Assigns segments to the correct function arguments.
- Deduplicates flavours.
- Handles "NO_VALUE_FOUND" by removing and adjusting output for readability.
- Avoids unnecessary quoting.

### Fail Criteria
A response **fails** if it:
- Omits required segments, output is not split as required, or misnames function arguments.
- Fails to wrap section headers/paragraphs in the specified HTML elements.
- Fails to deduplicate flavours.
- Displays "NO_VALUE_FOUND" in output or leaves awkward language after removal.
- Adds unnecessary quotes.

## Examples

### Good Example

```html
<h3>About PRODNAME</h3>
<p>PRODNAME by BRANDNAME is a FORMAT designed for ...</p>
```
Assigned to segment_1.

```html
<h4>Flavour Profile</h4>
<p>PRODNAME is available in the following flavours: FLAVOUR1, FLAVOUR2.</p>
```
Assigned to segment_3.

### Bad Example

```html
<h3>"About PRODNAME"</h3>
<p>"PRODNAME by BRANDNAME is a FORMAT designed for ..."</p>
```
(Headers in quotes: incorrect.)

Or:

```html
<h3>Flavour Profile</h3>
<p>PRODNAME is available in the following flavours: FLAVOUR1, FLAVOUR2, FLAVOUR1.</p>
```
(Duplicate flavour not removed.)