# wiki-memory → aaa-memory Integration Audit

_Generated 2026-06-20. All 31 markdown files in wiki-memory read._

---

## Architecture Comparison

### wiki-memory spec (ARCHITECTURE.md v4.1.0)

```
Hot:   ClawMem (REST API on :7438) — main searchable knowledge base
Warm:  Wiki pages (YAML frontmatter + markdown) — human-readable, permanent
Cold:  MemVid V2 (.mv2 files) — compressed archival, monthly snapshots
```

### aaa-memory (as built)

```
Hot:   aaa-memory vault (SQLite FTS5) — fast writes, session capture
Warm:  ClawMem (REST API on :7438) — indexed docs
Cold:  MemVid V2 — NOT IMPLEMENTED
```

### ⚠️ FUNCTIONAL DIFFERENCE: Tier roles are swapped

In wiki-memory, **ClawMem is the hot tier** — the primary searchable knowledge base. Wiki pages are the warm tier (human-readable output). In aaa-memory, **the vault is the hot tier** and ClawMem is the warm tier.

**Impact**: Wiki-memory's data flow is `Session → ClawMem → Dream Agent → Wiki pages`. aaa-memory's is `Session → Vault → Dream Agent → Wiki pages + ClawMem`. The vault adds an extra layer that wiki-memory doesn't have.

**Why it matters**: Wiki-memory's dream agent reads directly from ClawMem (`GET /documents/:docid`). Our dream agent reads from both ClawMem AND the vault. This means:
- Double storage for session data (vault + ClawMem)
- Dream agent has two sources to reconcile
- ClawMem may not have all session data that the vault has

---

## Feature-by-Feature Audit

### 1. Dream Agent (Phase 0-6)

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Phase 0: Budget** | `idle × 0.25`, cap 7200s, dynamic intake/refine ratio | Same formula, same cap | ✅ Yes |
| **Phase 1: Extract** | Read from ClawMem REST API | Reads from ClawMem + vault + raw/ | ⚠️ Different (extra vault source) |
| **Phase 2: Refine** | 4-factor weighted: 35% consistency, 25% freshness, 25% cross-ref, 15% evidence | Same formula | ✅ Yes |
| **Phase 2: Council** | Adversarial deliberation (8-10 personas, convergence detection) | Stub that always returns True | ✅ Same (both stubs) |
| **Phase 3: Compile** | YAML frontmatter, [[wikilinks]], provenance | Same | ✅ Yes |
| **Phase 4: Pattern Detect** | 3+ patterns → auto-skill | Same | ✅ Yes |
| **Phase 5: Re-index** | POST /reindex to ClawMem | Same | ✅ Yes |
| **Phase 6: Improve** | S-tier reference comparison, embedding-guided | Structural fixes only (missing metadata) | ✅ Same (both scaffolding) |

### 2. ClawMem Integration

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **REST API** | `localhost:7438` | Same | ✅ Yes |
| **Health check** | `GET /health` | Same | ✅ Yes |
| **Search** | `POST /search` with FTS + vector | FTS only (no vector embeddings running) | ⚠️ Different (no vectors) |
| **Document list** | `GET /documents?pattern=*` | Same | ✅ Yes |
| **Reindex** | `POST /reindex` | Same | ✅ Yes |
| **Embedding server** | `localhost:8088` (llama.cpp or cloud) | Not running | ⚠️ Different (FTS only) |

### 3. Wiki Page Compilation

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Location** | `~/ai-wiki/pages/` | Same | ✅ Yes |
| **Categories** | concepts, entities, sources, queries | Same | ✅ Yes |
| **YAML frontmatter** | title, created, updated, tags, confidence, status, sources, wikilinks | Same fields | ✅ Yes |
| **[[wikilinks]]** | Cross-references between pages | Not implemented in compilation | ❌ Different |
| **Provenance** | `sources: [clawmem://docid/...]` | Same | ✅ Yes |

### 4. Confidence Scoring

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Formula** | 0.35×consistency + 0.25×freshness + 0.25×cross-ref + 0.15×evidence | Same | ✅ Yes |
| **Auto-accept** | > 0.8 | Same (but lowered to 0.3 for testing) | ⚠️ Different threshold |
| **Flag for review** | 0.5 - 0.8 | Same | ✅ Yes |
| **Reject** | < 0.5 | Same | ✅ Yes |
| **Council trigger** | 0.5 - 0.6 AND contradicts high-confidence | Same (but stub) | ✅ Same (both stubs) |

### 5. Retrieval Pipeline

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Intent router** | 6 signals: recent, entity, relational, archival, factual, ambiguous | Simple keyword-based classification | ⚠️ Different (simpler) |
| **Tier routing** | Recent → ClawMem only, Historical → ClawMem + MemVid | Hot + Wiki + ClawMem | ⚠️ Different (extra tier) |
| **RRF fusion** | Reciprocal rank fusion across tiers | Not implemented (simple concatenation) | ❌ Different |
| **Token budget** | 2000 tokens max | Not implemented | ❌ Different |

### 6. Auto-Skill Creation

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Threshold** | 3+ patterns | Same | ✅ Yes |
| **Output** | SKILL.md in Pi skills directory | Same | ✅ Yes |
| **Pattern types** | code-review, deployment, testing, debugging, database, api-development, research | Same | ✅ Yes |

### 7. Systemd Services

| Service | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **clawmem-serve** | `clawmem serve --port 7438` | Same | ✅ Yes |
| **clawmem-watcher** | `clawmem watch` | Same | ✅ Yes |
| **dream timer** | systemd idle timer (30min check) | Not implemented (manual trigger only) | ❌ Different |
| **embed timer** | Daily embedding refresh | Not implemented | ❌ Different |

### 8. Cold Storage (MemVid)

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Format** | .mv2 tar containers with HDF5 vectors | Not implemented | ✅ Same (neither implemented) |
| **Schedule** | Monthly snapshots | Not implemented | ✅ Same |
| **CLI** | `mv2 create/restore/query` | Not implemented | ✅ Same |

### 9. Council Deliberation

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Architecture** | 8-10 personas, 4 rounds, convergence detection | Stub (always True) | ✅ Same (both stubs) |
| **Models** | Advocate (deepseek-v4-flash) + Skeptic (claude-sonnet-4) | Not implemented | ✅ Same |
| **Verdict** | Accept/Reject/Flag for Human | Always accepts | ✅ Same |

### 10. S-Tier Improvement Engine

| Feature | wiki-memory spec | aaa-memory built | Same behavior? |
|---------|-----------------|------------------|----------------|
| **Reference corpus** | Curated exemplar documents per type | Not implemented | ✅ Same (neither implemented) |
| **Embedding comparison** | cosine distance to reference | Not implemented | ✅ Same |
| **Rubric analysis** | 6 criteria with weighted scoring | Not implemented | ✅ Same |
| **LLM refinement** | Targeted improvements from rubric gaps | Not implemented | ✅ Same |
| **Structural fixes** | Missing metadata, broken links | Same | ✅ Yes |

---

## Summary of Functional Differences

### Same behavior (no difference):
1. Budget allocation formula
2. Confidence scoring formula
3. Wiki page compilation (YAML frontmatter)
4. Auto-skill creation
5. ClawMem REST API integration
6. Council deliberation (both stubs)
7. S-tier improvement (both structural-only)
8. MemVid cold storage (neither implemented)

### Different behavior (user would notice):

| # | Difference | Impact | Severity |
|---|-----------|--------|----------|
| 1 | **Extra vault tier** — aaa-memory adds a vault layer that wiki-memory doesn't have | Double storage, extra complexity | 🟡 Medium |
| 2 | **[[wikilinks]] not generated** — wiki-memory spec requires cross-references between pages | Pages are isolated, no navigation | 🟡 Medium |
| 3 | **No intent router** — wiki-memory defines 6-signal intent classification | All queries hit all tiers | 🟢 Low |
| 4 | **No RRF fusion** — wiki-memory specifies reciprocal rank fusion | Results not properly ranked | 🟡 Medium |
| 5 | **No token budget** — wiki-memory caps retrieval at 2000 tokens | Context overflow possible | 🟢 Low |
| 6 | **No dream timer** — wiki-memory specifies systemd idle timer | Must trigger manually | 🟡 Medium |
| 7 | **No vector embeddings** — ClawMem embedding server not running | FTS-only search | 🟡 Medium |
| 8 | **Lower confidence threshold** — aaa-memory uses 0.3 (testing) vs wiki-memory's 0.8 | More claims accepted | 🟢 Low (configurable) |

---

## Recommendations

1. **Add [[wikilinks]] to wiki page compilation** — the spec requires cross-references between pages. This is a significant gap in knowledge graph formation.

2. **Implement RRF fusion** — the retrieval pipeline should properly rank results across tiers.

3. **Install dream timer** — systemd idle timer for automatic dream cycles.

4. **Start ClawMem embedding server** — enables vector search beyond FTS.

5. **Consider simplifying architecture** — the vault tier may be redundant if ClawMem is the primary knowledge base. Wiki-memory's simpler `Session → ClawMem → Wiki pages` flow is cleaner.
