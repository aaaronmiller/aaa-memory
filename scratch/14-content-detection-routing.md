# Topic 14: Content Detection & Modality Routing

## Summary
Implement the content type detection and modality classification system that determines how each input file should be processed, which chunking methods to apply, and which domain it belongs to.

---

## Overview
> Sources: `opus-prd2-v3.md` (lines 305–309, chunking `applies_to` sections), `chatgpt5.2-prd.md` (lines 36, 26–27), `docs/UNIFIED_PRD.md` (Implementation Roadmap Stage 1: "Build multimodal content detector")

The content detection system is the first stage of the ingestion pipeline. It classifies each input file by content type, modality, and domain, then routes it to the appropriate processing pipeline.

---

## Content Types
> Source: `docs/SCHEMA_REFERENCE.md` (lines 525–533)

```typescript
type ContentType = 
  | 'code' 
  | 'documentation' 
  | 'research_paper'
  | 'prompt'
  | 'configuration'
  | 'data'
  | 'conversation'
  | 'mixed';
```

## Modalities
> Source: `docs/SCHEMA_REFERENCE.md` (line 535)

```typescript
type Modality = 'text' | 'image' | 'video' | 'audio' | 'screenshot' | 'diagram';
```

---

## Detection Logic

### File Extension → Content Type Mapping

| Extension | Content Type | Domain | Modality |
|-----------|-------------|--------|----------|
| `.py` | code | codebase | text |
| `.ts`, `.js` | code | codebase | text |
| `.go`, `.rs`, `.java` | code | codebase | text |
| `.md` | documentation OR research_paper | research/prompts | text |
| `.yaml`, `.yml`, `.json`, `.toml` | configuration | codebase | text |
| `.docx` | documentation | research | text |
| `.pdf` | research_paper | research | text (+ potential images) |
| `.png`, `.jpg`, `.svg` | mixed | varies | image |
| `.mp4`, `.webm` | mixed | varies | video |

### Content-Based Detection (beyond extension)

For ambiguous files (e.g., `.md` could be documentation, prompt, or research):
1. Check file path for domain hints (e.g., `/prompts/`, `/research/`, `/src/`)
2. Analyze content structure (headings, code blocks, conversation patterns)
3. Check for multimodal references (image links, embedded diagrams)

---

## Routing Rules
> Source: `opus-prd2-v3.md` (chunking `applies_to` sections throughout)

| Content Type | Modality | Chunking Methods | Domain |
|-------------|----------|-----------------|--------|
| code | text | ast_structural, fixed_size | codebase |
| code | mixed_text_image | ast_structural, screenshot_code_fusion | codebase |
| documentation | text | semantic, recursive_hierarchical | research |
| documentation | mixed_text_image | multimodal_boundary, semantic | research |
| research_paper | text | recursive_hierarchical, semantic | research |
| research_paper | mixed_all | multimodal_boundary, recursive_hierarchical | research |
| prompt | text | semantic, fixed_size | prompts |
| configuration | text | fixed_size | codebase |
| data | text | fixed_size | codebase |
| conversation | text | recursive_hierarchical | prompts |

---

## Implementation Tasks

1. Create `src/detection/content_detector.py` — Classify files by content type based on extension + content analysis
2. Create `src/detection/modality_detector.py` — Detect modalities present in a file (text, images, video, etc.)
3. Create `src/detection/domain_router.py` — Map detected content to appropriate domain (prompts/codebase/research)
4. Create `src/detection/language_detector.py` — Detect natural language (ISO 639-1) and programming language
5. Create `src/detection/mime_detector.py` — MIME type detection for proper file handling
6. Implement PDF text/image extraction pipeline
7. Implement DOCX text extraction pipeline

---

## Conflicts & Ambiguities

1. **Markdown classification:** `.md` files could be prompts, documentation, or research. The routing depends on directory structure or content analysis. No explicit rules are provided — need heuristics.

2. **Multi-method chunking:** Some content types get multiple chunking methods (e.g., code gets both `ast_structural` and `fixed_size`). It's unclear whether both methods run in parallel producing separate chunk sets, or if one is primary and the other is fallback. The AST config has `fallback_to_fixed: true` suggesting fixed-size is a fallback for code, but for research, both semantic and recursive_hierarchical seem to run independently.

3. **Image-only files:** Pure image files (`.png`, `.jpg`) are listed as supported modalities but there's no standalone "image chunking" method. They would be embedded directly via Qwen3-VL-Embedding-8B without chunking, or attached to nearby text chunks via multimodal boundary detection.
