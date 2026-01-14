# html-output Specification

## Purpose
Requires output to be properly formatted with HTML elements as per specification for headers and paragraphs/lists.

## Requirements

### Requirement: HTML Wrapping

Headers SHALL be wrapped in the prescribed `<h3>` or `<h4>` tags. Paragraphs SHALL use `<p>`, unless a different element (e.g., `<ul>`) is explicitly specified.

#### Scenario: Generating a header and descriptive text

- **WHEN** outputting section content
- **THEN** always use the specified HTML tags for each content element.

### Requirement: No Extraneous Markup

Do not surround output or content with extraneous HTML or markup beyond what is specified.

#### Scenario: Temptation to over-format or add wrappers

- **WHEN** outputting the segments
- **THEN** keep only the required HTML elements, no additional divs, spans, etc.

## Evaluation Guidelines

### Pass Criteria

A response **passes** if it:
- Uses only the required HTML tags for structuring output.
- No extra or missing markup.

### Fail Criteria

A response **fails** if it:
- Lacks required HTML wrappers.
- Adds unneeded or extraneous HTML.

## Examples

### Good Example

`<h3>About PRODNAME</h3><p>PRODNAME by BRANDNAME ...</p>`

### Bad Example

`<div><h3>About PRODNAME</h3></div><span><p>PRODNAME by BRANDNAME ...</p></span>`
(Extraneous div and span.)