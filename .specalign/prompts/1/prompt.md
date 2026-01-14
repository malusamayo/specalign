You are to generate a product description composed of multiple distinct segments, strictly adhering to the following comprehensive specifications:

---

### 1. Content Source and Accuracy

- Use **only** the provided product attribute values (e.g., NAME_LONG, BRAND, FORMAT, FLAVOURS, NUMBER_OF_USES, STRENGTH, CATEGORY, BRAND_DESCRIPTION).  
- Do **not** fabricate, infer, or speculate on any data not explicitly given.  
- If any attribute is `"NO_VALUE_FOUND"`, omit it entirely and adjust sentences for coherence without leaving awkward phrasing or placeholders.  
- Do **not** include any health claims, references, or implications regarding health benefits or effects.  
- Refrain from any value judgements, subjective language, or superlatives about the product, brand, or flavours.

---

### 2. Forbidden Language

- Exclude all forbidden words, their inflections, and synonyms. Forbidden terms include but are not limited to:  
  *excellent, better, satisfy, sensational, delicious, great, convenient, exquisite, fulfilling, boost, enjoy, needs, unique, freshness, refreshing, light, cool, regret, mouth-watering, passion, pleasant, perfect, fresh, quality, punsch, epic, craving, smooth, exciting, safe, mild, amazing, safer, magic, popular, impact, users, natural, clean, low, trendy, ideal, juicy, fix, satisfy, newcomer*  
- Do **not** replace forbidden words with synonyms, paraphrases, or euphemisms. Instead, omit or rephrase content to avoid these terms altogether.

---

### 3. Output Structure and Formatting

Produce the output divided into the following named segments, each returning two function arguments: a header and a paragraph. Use the exact argument names below:

- `segment_1`: Header and paragraph titled **"About PRODNAME"** — introduce the product using provided attributes (e.g., product name, brand, format).  
- `segment_2`: Header and paragraph titled **"Characteristics for Usage"** — describe product characteristics relevant to usage, including strength and total number of uses.  
- `segment_3`: Header and paragraph titled **"Flavour Profile"** — list available flavours, deduplicated, separated by commas.  
- `segment_5`: Header and paragraph titled **"Summary"** — summarise key product details including category, flavours, strength, uses, manufacturer, and brand information.  
- `segment_short`: Header and paragraph titled **"Brand Overview"** — provide a brief description specific to the brand.

**Formatting requirements:**

- Headers must be wrapped in `<h3>` tags for `segment_1` and `segment_short`, and `<h4>` tags for `segment_2`, `segment_3`, and `segment_5`.  
- Paragraphs must be wrapped in `<p>` tags, unless explicitly instructed otherwise (no lists required here).  
- Do **not** add any extra or extraneous HTML elements (no `<div>`, `<span>`, quotes, or other wrappers).  
- Do not wrap output text in quotation marks unless quotes appear verbatim in input data.

---

### 4. Language, Tone, and Style

- Write in formal, professional, and neutral tone—factual and informative only.  
- Use British English spelling and conventions throughout.  
- Avoid any slang, casual language, or promotional tone.  
- Do not include any emotional, comparative, or superlative language or embellishments.  
- Descriptions of brand and flavours must be strictly fact-based.

---

### 5. SEO and Uniqueness

- Each segment’s paragraph must be uniquely worded and creative while compliant with all above constraints.  
- Incorporate relevant product-related keywords naturally (e.g., product name, brand, format, strength, flavours, usage).  
- Avoid verbatim repetition of phrases or sentences across segments; rephrase content to maximise uniqueness for SEO.  
- Do not use generic filler text or duplicated content between segments.

---

### 6. Summary of Evaluation Pass Criteria

Your output passes if it:

- Contains only factual information from provided attributes, with no invented data or health claims.  
- Omits or edits out `"NO_VALUE_FOUND"` attributes cleanly.  
- Uses no forbidden words, inflections, or synonyms, and does not substitute with similar terms.  
- Produces five distinct segments with correct function argument names and content per specification.  
- Wraps headers and paragraphs in correct HTML tags only, with no extra markup.  
- Uses British English consistently in a formal, factual style without value judgements.  
- Presents unique, keyword-rich paragraphs avoiding repetition across segments.  
- Does not add unnecessary quotation marks around text.

---

### Example segment header and paragraph (for reference only):

```html
<h3>About PRODNAME</h3>
<p>PRODNAME by BRANDNAME is a FORMAT designed for NUMBER_OF_USES uses.</p>
```

---

### Your task:

Given product attributes, generate the five segments (`segment_1`, `segment_2`, `segment_3`, `segment_5`, and `segment_short`) as described, strictly following all above instructions. Ensure the output is factual, neutral, free of forbidden language, properly formatted in HTML, and optimised uniquely for SEO in British English.