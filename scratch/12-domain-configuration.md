# Topic 12: Domain Configuration

## Summary
Configure the three content domains (prompts, codebase, research) with domain-specific chunking methods, storage files, retention policies, and multimodal settings.

---

## Overview
> Sources: `opus-prd2-v3.md` (lines 397–429), `chatgpt5.2-prd.md` (lines 26–27, 31, 35–36)

The system handles three distinct content domains, each with tailored processing pipelines.

---

## Domain Definitions
> Source: `opus-prd2-v3.md` (lines 400–429)

### Domain 1: Prompts
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

- **Content:** User prompts to LLMs, conversation logs
- **Chunking:** Semantic + fixed-size (text-only)
- **Retention:** 30-day rolling window
- **Multimodal:** No

### Domain 2: Codebase
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

- **Content:** Source code from multiple repositories, config files
- **Chunking:** AST structural + fixed-size + screenshot-code fusion
- **Retention:** Version-controlled (tied to git history)
- **Multimodal:** Yes (screenshots, diagrams)
- **Cross-reference:** Enabled (link code to docs)
- **Repository structure:** Multi-repo setup

> Source: `chatgpt5.2-prd.md` (lines 26–27, 36)
- Code files: Python, TypeScript/JavaScript
- Config, script, and library files may benefit from different tokenization
- Monorepo vs multi-repo: multi-repo setup confirmed

### Domain 3: Research
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

- **Content:** Research papers, documentation, writing, diagrams
- **Chunking:** Recursive hierarchical + semantic + multimodal boundary
- **Retention:** Permanent
- **Multimodal:** Yes (figures, tables, equations, diagrams)

> Source: `chatgpt5.2-prd.md` (lines 35–36)
- Input types: Mostly markdown, some Python/JS, some DOCX and PDF
- Total corpus: ~35MB of text files

---

## File Type Handling
> Source: `chatgpt5.2-prd.md` (line 36)

| File Type | Domain | Notes |
|-----------|--------|-------|
| `.md` (Markdown) | Research/Prompts | Primary format |
| `.py` (Python) | Codebase | AST chunking |
| `.ts`/`.js` (TypeScript/JavaScript) | Codebase | AST chunking |
| `.docx` (Word) | Research | Needs conversion |
| `.pdf` (PDF) | Research | Needs extraction |
| Config files (`.yaml`, `.json`, `.toml`) | Codebase | Fixed-size chunking |

---

## Implementation Tasks

1. Create `src/config/domains.py` — Domain configuration loader and validator
2. Create `src/config/domain_router.py` — Route files to appropriate domain based on path/type
3. Implement per-domain MemVid file management (separate MP4 per domain)
4. Implement retention policies (30-day rolling, version-controlled, permanent)
5. Add DOCX and PDF text extraction support
6. Configure cross-referencing between codebase and research domains

---

## Conflicts & Ambiguities

1. **Separate databases vs unified:** Each domain has its own MemVid file (`prompts.mp4`, `codebase.mp4`, `research.mp4`), but it's unclear if they share a single SQLite metadata database or have separate ones. A single SQLite DB with `corpus_id` filtering is more practical.

2. **Retention implementation:** "30_days_rolling" for prompts means old prompts are deleted. "version_controlled" for codebase means chunks are tied to git commits. "permanent" for research means never deleted. The retention logic needs to handle all three modes.

3. **Config file chunking:** `chatgpt5.2-prd.md` asks "do I want different tokenization for config, script, and library files?" The answer in `opus-prd2-v3.md` is: config files use fixed-size chunking, code files use AST structural. But the boundary between "config" and "code" needs to be defined (e.g., `.yaml` = config, `.py` = code).
