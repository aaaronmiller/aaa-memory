# Topic: Domain Configuration & Content Routing

## Summary
Configuration for the three content domains (prompts, codebase, research), including per-domain chunking methods, storage files, retention policies, and content type detection.

---

## Domain Definitions

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

### Domain: Prompts
```yaml
- name: "prompts"
  description: "User inputs and prompts to LLMs"
  storage: "prompts.mp4"
  chunking_methods:
    - "semantic"
    - "fixed_size"
  retention: "30_days_rolling"
  multimodal: false
```

### Domain: Codebase
```yaml
- name: "codebase"
  description: "Multi-repository source code and configs"
  storage: "codebase.mp4"
  chunking_methods:
    - "ast_structural"
    - "fixed_size"
    - "screenshot_code_fusion"
  retention: "version_controlled"
  multimodal: true
  cross_reference: true
```

### Domain: Research
```yaml
- name: "research"
  description: "Research papers, documentation, diagrams"
  storage: "research.mp4"
  chunking_methods:
    - "recursive_hierarchical"
    - "semantic"
    - "multimodal_boundary"
  retention: "permanent"
  multimodal: true
```

---

## Content Type Detection

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md), [opus-prd2-v3.md](../opus-prd2-v3.md)

### Supported File Types
- **Markdown (.md)** — Primary format for writing/documentation
- **Python (.py)** — Code
- **JavaScript/TypeScript (.js/.ts)** — Code
- **DOCX** — Documents (not large portion)
- **PDF** — Research papers (not large portion)
- **Config files** — Various formats

### Content Type Mapping

| File Type | Content Type | Domain | Chunking Methods |
|-----------|-------------|--------|-----------------|
| .md (writing) | documentation | research | recursive_hierarchical, semantic, multimodal_boundary |
| .md (prompts) | prompt | prompts | semantic, fixed_size |
| .py, .js, .ts | code | codebase | ast_structural, fixed_size |
| .json, .yaml, .toml | configuration | codebase | fixed_size |
| .pdf | research_paper | research | recursive_hierarchical, semantic |
| .docx | documentation | research | recursive_hierarchical, semantic |

---

## Multi-Repository Setup

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- Codebase is stored in a multi-repo setup (not monorepo)
- Each repository should be tracked separately for provenance
- Git metadata (commit SHA, branch, author) captured per chunk

---

## Corpus Size Estimates

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md), [gemini-prd.md](../gemini-prd.md)

| Metric | Value |
|--------|-------|
| Initial text corpus | ~35MB |
| Initial documents (gemini estimate) | 500 docs × 150 pages = 75,000 pages |
| Weekly growth (gemini estimate) | +100 docs × 150 pages = +15,000 pages/week |

---

## Retention Policies

| Domain | Policy | Description |
|--------|--------|-------------|
| Prompts | 30-day rolling | Older prompts archived to MemVid |
| Codebase | Version-controlled | Tied to git history, never deleted |
| Research | Permanent | Always retained |

---

## Implementation Requirements

1. Implement content type detector (file extension + content analysis)
2. Build domain router (content type → domain → chunking methods)
3. Configure per-domain MemVid files
4. Implement retention policy enforcement
5. Build multi-repo ingestion support with git provenance tracking
6. Create domain-specific embedding instructions (instruction-aware model)

---

## Conflicts / Ambiguities

- **⚠️ Corpus size discrepancy:** chatgpt5.2-prd.md says ~35MB of text files; gemini-prd.md estimates 75,000 pages initially with 15,000 pages/week growth. These may refer to different corpora or different time horizons.
- **⚠️ Prompt vs documentation detection:** Both prompts and documentation can be markdown files. The routing logic needs a way to distinguish user prompts from documentation (possibly by source directory or metadata).
- **⚠️ Code tokenization:** chatgpt5.2-prd.md asks whether different tokenization is needed for config vs script vs library files. The opus-prd2 config uses fixed_size for configs and ast_structural for code, which implicitly answers this.
