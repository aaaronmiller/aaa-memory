# aaa-memory Constitution

## Core Principles

1. **Local-first, zero cloud dependency** — All storage is on-device. Free-tier APIs are fallbacks only. No vendor lock-in.
2. **Markdown as source of truth** — Every wiki page, every element, every decision is plain text with YAML frontmatter. Git-trackable, human-readable, diffable.
3. **Raw is immutable, derived is rebuildable** — `raw_text` never changes. Summaries, embeddings, and topic labels can be regenerated from raw at any time.
4. **Single encode at extraction time** — Embeddings computed once, stored with the element, reused by ClawMem, Graphiti, and MemVid V2. No re-encoding.
5. **Sub-indexes are the shard** — Master `index.md` points to sub-indexes, not individual articles. Each sub-index stays under ~700 items for Karpathy pointer navigation.
6. **Provenance chains are mandatory** — Every element tracks its origin: source file → extraction → wiki page → Graphiti episode → MemVid archive.
7. **Never pay for software** — All components are open-source or free-tier. No commercial dependencies.

## Quality Gates

- Every spec must pass the requirements checklist before planning
- Every plan must resolve all NEEDS CLARIFICATION markers before implementation
- No implementation details in specs (WHAT only)
- No user requirements in plans (HOW only)
