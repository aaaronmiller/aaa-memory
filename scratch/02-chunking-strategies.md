# Topic: Chunking Strategies

## Summary
Seven distinct chunking methods for processing different content types (text, code, mixed-modal) into the RAG system. Includes configuration, routing logic, and the four-layer epistemic scaffolding model.

---

## Conceptual Framework: Four-Layer Epistemic Scaffolding

> **Source:** [opus-prd3-v3.md](../opus-prd3-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

Chunking has evolved into a four-layer system:
1. **Fixed-length chunking** — mechanical, deterministic
2. **Sentence/semantic-unit chunking** — linguistic awareness
3. **Semantic coherence chunking (agentic)** — meaning-aware boundaries
4. **Recursive hierarchical chunking (agentic)** — document-structure-aware

---

## Method 1: Fixed-Size Chunking

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Window tokens:** 512
- **Overlap tokens:** 50
- **Applies to:** configuration files, data files
- **Modalities:** text only
- **Agent required:** No — can be done programmatically

### Implementation Notes
> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- Length-based chunking can be done programmatically without an LLM agent
- Simplest method, serves as fallback for AST chunking failures

---

## Method 2: Sentence-Based Chunking

> **Source:** [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- **Window size:** 3 sentences
- **Min chunk tokens:** 128
- **Max chunk tokens:** 2048
- **Agent required:** No — can be done programmatically

---

## Method 3: Semantic Chunking (Agentic)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- **Similarity threshold:** 0.75
- **Window size:** 3 sentences
- **Boundary detection model:** `Qwen/Qwen3-Embedding-0.6B`
- **Min chunk tokens:** 128
- **Max chunk tokens:** 2048
- **Applies to:** documentation, research papers
- **Modalities:** text only
- **Agent required:** Yes — requires intelligence for boundary detection

### How It Works
> **Source:** [opus-prd3-v3.md](../opus-prd3-v3.md)

Uses embedding similarity between adjacent sentence windows to detect topic shifts. When similarity drops below threshold (0.75), a chunk boundary is placed. The lightweight 0.6B model handles boundary detection cheaply.

---

## Method 4: Recursive Hierarchical Chunking (Agentic)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- **Chunk size tokens:** 1024
- **Overlap tokens:** 100
- **Separators** (in priority order):
  1. `"\n\n"` — Paragraphs
  2. `"\n"` — Lines
  3. `". "` — Sentences
  4. `" "` — Words (last resort)
- **Applies to:** documentation, conversation
- **Modalities:** text only
- **Agent required:** Yes — requires understanding of document structure

---

## Method 5: AST Structural Chunking (Code)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

### Supported Languages & Parsers

| Language | Parser | AST Nodes |
|----------|--------|-----------|
| Python | tree-sitter-python | function_definition, class_definition, decorated_definition |
| TypeScript | tree-sitter-typescript | function_declaration, class_declaration, method_definition, interface_declaration |
| JavaScript | tree-sitter-javascript | function_declaration, class_declaration, method_definition |
| Go | tree-sitter-go | function_declaration, method_declaration, type_declaration |
| Rust | tree-sitter-rust | function_item, impl_item, struct_item, trait_item |
| Java | tree-sitter-java | method_declaration, class_declaration, constructor_declaration, interface_declaration |

### Configuration
- `prepend_parent_context`: true
- `preserve_docstrings`: true
- `preserve_imports`: true
- `extract_dependencies`: true
- `compute_complexity`: true
- `fallback_to_fixed`: true (falls back to fixed-size 512 tokens on parse failure)

### Applies to
- Content types: code
- Modalities: text

---

## Method 6: Multimodal Boundary Detection (NEW)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Visual context window:** 1 paragraph before/after
- **Caption detection:** true
- **Figure reference detection:** true
- **Preserve figure-caption pairs:** true
- **Applies to:** documentation, research papers
- **Modalities:** mixed_text_image, mixed_all

### Purpose
Detects boundaries between text and visual content in mixed documents. Ensures figures, diagrams, and their captions are kept together as coherent chunks.

---

## Method 7: Screenshot-Code Fusion (NEW)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Matching strategies:**
  - `filename_similarity` — match screenshots to code files by name
  - `ocr_text_matching` — extract text from screenshots, match to code
  - `reference_comment_detection` — find code comments referencing screenshots
- **Applies to:** code
- **Modalities:** mixed_text_image

### Purpose
Fuses UI screenshots with the code that generates them, creating cross-modal chunks that link visual output to source code.

---

## Content Type → Chunking Method Routing

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md) (domains section)

| Domain | Chunking Methods |
|--------|-----------------|
| Prompts | semantic, fixed_size |
| Codebase | ast_structural, fixed_size, screenshot_code_fusion |
| Research | recursive_hierarchical, semantic, multimodal_boundary |

---

## Asynchronous / Multi-Agent Chunking

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

### Agent Assignment by Method
- **Fixed-size & Sentence-based:** Programmatic (no LLM needed)
- **Semantic chunking:** Requires LLM intelligence — can use Haiku/Flash-class model
- **Recursive hierarchical:** Requires higher intelligence — Sonnet/Pro-class model recommended

### Key Questions from Requirements
- Can semantic and recursive hierarchical chunking be done in a single pass by one agent, or do they require separate passes?
- The user suggests asynchronous processing across files is ideal given the multi-file corpus

### Model Recommendations for Agentic Chunking
> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- Sonnet/Gemini Pro class: For recursive hierarchical chunking
- Haiku/Gemini Flash class: For semantic chunking
- Free models (e.g., MIMO V2 via OpenRouter): For simpler tasks

---

## Quad Encoding (MemVid-Specific Chunking)

> **Source:** [gemini-prd.md](../gemini-prd.md) (Appendix I)

MemVid uses "Quad Encoding" — encoding the same content at four resolutions:

| Resolution | What it Encodes | Agent Query Type |
|-----------|----------------|-----------------|
| Word (Token) | Keywords & Entities | Exact definitions, variable names |
| Sentence | Discrete Facts | Return types, specific error codes |
| Paragraph | Local Context | How a flow handles edge cases |
| Boundary | Relationships & Flow | What connects between sections |

This is done during sleep-time compute (not real-time) due to 4x embedding cost.

---

## Conflicts / Ambiguities

- **⚠️ Chunk size inconsistency:** chatgpt5.2-prd.md mentions "1.5-3K tokens" for chunks; opus-prd2-v3.md specifies 512 tokens (fixed), 1024 tokens (recursive), 128-2048 tokens (semantic). The AGGREGATION_PLAN.md lists yet another set: "1.5-3k tokens with 200-400 token overlap" for fixed-size. The opus-prd2 YAML config should be treated as authoritative.
- **⚠️ Number of methods:** UNIFIED_PRD.md lists 7 methods; chatgpt5.2-prd.md discusses 4 core methods; opus-prd2-v3.md defines 6 in YAML config. The 7-method list (adding sentence-based as distinct from semantic) is the most complete.
- **⚠️ Agentic vs programmatic:** chatgpt5.2-prd.md suggests semantic and recursive hierarchical need LLM agents; opus-prd2-v3.md treats semantic chunking as algorithmic (embedding similarity threshold). Resolution: semantic chunking uses the lightweight 0.6B model algorithmically, not a full LLM agent.