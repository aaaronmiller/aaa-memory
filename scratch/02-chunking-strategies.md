# Topic 02: Chunking Strategies

## Summary
Implement the 7 chunking methods that form the "four-layer epistemic scaffolding system" for decomposing documents into retrievable units. Each method targets different content types and modalities.

---

## Overview
> Sources: `opus-prd3-v3.md` (foundational theory), `opus-prd2-v3.md` (lines 115–209, YAML config), `docs/UNIFIED_PRD.md` (lines 129–166), `chatgpt5.2-prd.md` (lines 7–9)

The chunking pipeline uses content-type routing to select the appropriate method(s):

| Method | Target Content | Modality | Agentic? |
|--------|---------------|----------|----------|
| Fixed-Size | Data, Config | Text | No |
| Sentence-Based | Pure Text | Text | No |
| Semantic | Documentation, Research | Text | Yes |
| Recursive Hierarchical | Documentation, Conversation | Text | Yes |
| AST Structural | Code | Text | Yes |
| Multimodal Boundary | Mixed docs/research | Mixed | Yes |
| Screenshot-Code Fusion | Code + screenshots | Mixed | Yes |

---

## Method 1: Fixed-Size Chunking
> Sources: `opus-prd2-v3.md` (lines 180–187), `docs/UNIFIED_PRD.md` (lines 137–139), `chatgpt5.2-prd.md` (line 7)

- **Window tokens:** 512
- **Overlap tokens:** 50
- **Applies to:** content_types `["configuration", "data"]`, modalities `["text"]`
- **Implementation:** Programmatic (no LLM needed)
- **Complexity:** Simple token-counting with sliding window

## Method 2: Sentence-Based Chunking
> Sources: `docs/UNIFIED_PRD.md` (lines 141–143), `chatgpt5.2-prd.md` (line 7), `opus-prd3-v3.md`

- **Window size:** 3 sentences
- **Min chunk tokens:** 128
- **Max chunk tokens:** 2048
- **Implementation:** Programmatic (sentence boundary detection, no LLM needed)

## Method 3: Semantic Chunking
> Sources: `opus-prd2-v3.md` (lines 119–129), `docs/UNIFIED_PRD.md` (lines 146–148), `opus-prd3-v3.md`

- **Similarity threshold:** 0.75
- **Window size:** 3 sentences
- **Embedding model:** `boundary_detection` (Qwen3-Embedding-0.6B)
- **Min chunk tokens:** 128
- **Max chunk tokens:** 2048
- **Applies to:** content_types `["documentation", "research_paper"]`, modalities `["text"]`
- **Implementation:** Requires embedding model for similarity computation between adjacent segments. Agentic — needs intelligence to determine semantic boundaries.

**Algorithm:**
1. Split text into sentences
2. Compute embeddings for sliding windows of sentences
3. Calculate cosine similarity between adjacent windows
4. Split at points where similarity drops below threshold (0.75)

## Method 4: Recursive Hierarchical Chunking
> Sources: `opus-prd2-v3.md` (lines 131–143), `docs/UNIFIED_PRD.md` (lines 150–153), `chatgpt5.2-prd.md` (line 7), `opus-prd3-v3.md`

- **Chunk size tokens:** 1024
- **Overlap tokens:** 100
- **Separators (in priority order):**
  1. `"\n\n"` — Paragraphs
  2. `"\n"` — Lines
  3. `". "` — Sentences
  4. `" "` — Words (last resort)
- **Applies to:** content_types `["documentation", "conversation"]`, modalities `["text"]`
- **Implementation:** Agentic — recursive splitting with intelligent boundary selection

## Method 5: AST Structural Chunking (Code)
> Sources: `opus-prd2-v3.md` (lines 145–178), `docs/UNIFIED_PRD.md` (lines 155–158)

- **Languages & Parsers:**
  - Python: `tree-sitter-python` → `[function_definition, class_definition, decorated_definition]`
  - TypeScript: `tree-sitter-typescript` → `[function_declaration, class_declaration, method_definition, interface_declaration]`
  - JavaScript: `tree-sitter-javascript` → `[function_declaration, class_declaration, method_definition]`
  - Go: `tree-sitter-go` → `[function_declaration, method_declaration, type_declaration]`
  - Rust: `tree-sitter-rust` → `[function_item, impl_item, struct_item, trait_item]`
  - Java: `tree-sitter-java` → `[method_declaration, class_declaration, constructor_declaration, interface_declaration]`
- **Options:**
  - `prepend_parent_context`: true
  - `preserve_docstrings`: true
  - `preserve_imports`: true
  - `extract_dependencies`: true
  - `compute_complexity`: true
  - `fallback_to_fixed`: true (fallback window: 512 tokens)
- **Applies to:** content_types `["code"]`, modalities `["text"]`

## Method 6: Multimodal Boundary Detection
> Sources: `opus-prd2-v3.md` (lines 189–198), `docs/UNIFIED_PRD.md` (lines 160–162)

- **Visual context window:** 1 paragraph before/after
- **Caption detection:** true
- **Figure reference detection:** true
- **Preserve figure-caption pairs:** true
- **Applies to:** content_types `["documentation", "research_paper"]`, modalities `["mixed_text_image", "mixed_all"]`

## Method 7: Screenshot-Code Fusion
> Sources: `opus-prd2-v3.md` (lines 200–209), `docs/UNIFIED_PRD.md` (lines 164–166)

- **Matching strategies:**
  - `filename_similarity`
  - `ocr_text_matching`
  - `reference_comment_detection`
- **Applies to:** content_types `["code"]`, modalities `["mixed_text_image"]`

---

## Implementation Tasks

1. Create `src/chunking/fixed_size.py` — token-counting sliding window
2. Create `src/chunking/sentence_based.py` — sentence boundary detection + grouping
3. Create `src/chunking/semantic.py` — embedding-based similarity boundary detection
4. Create `src/chunking/recursive_hierarchical.py` — recursive separator-based splitting
5. Create `src/chunking/ast_structural.py` — tree-sitter based code chunking (6 languages)
6. Create `src/chunking/multimodal_boundary.py` — image/figure boundary detection in mixed docs
7. Create `src/chunking/screenshot_code_fusion.py` — screenshot-to-code matching
8. Create `src/chunking/router.py` — content-type router that selects appropriate chunking method(s)

---

## Conflicts & Ambiguities

1. **Sentence-based chunking config location:** `docs/UNIFIED_PRD.md` lists sentence-based as Method 2 with specific params, but `opus-prd2-v3.md` YAML config does not include a separate `sentence_based` section — it's folded into the semantic chunking config's `window_size: 3` sentences. Clarify whether sentence-based is a standalone method or part of semantic.

2. **Chunking method count:** `docs/UNIFIED_PRD.md` says "6 chunking methods" in the roadmap (Stage 2) but lists 7 methods. The YAML config in `opus-prd2-v3.md` has 6 methods (no standalone sentence-based).

3. **Agentic vs programmatic:** `chatgpt5.2-prd.md` (lines 18–23) discusses which methods need LLM intelligence vs programmatic. Fixed-size and sentence-based are programmatic. Semantic and recursive hierarchical are agentic. The question of whether semantic + recursive can share a single agent pass is raised but not definitively answered.

4. **Token sizes diverge:** `docs/AGGREGATION_PLAN.md` (line 163) mentions "1.5-3k tokens with 200-400 token overlap" for fixed-size, but `opus-prd2-v3.md` specifies 512 tokens with 50 overlap. The YAML config is more authoritative.

5. **Quad Encoding integration:** `gemini-prd.md` (Appendix I, lines 295–368) describes a "Quad Encoding" approach where the same content is chunked at 4 resolutions (Word, Sentence, Paragraph, Boundary). This is separate from the 7 chunking methods and applies during the MemVid archival process. Needs clarification on how these interact.
