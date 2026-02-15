# Topic 11: Quality Assurance & Error Handling

## Summary
Implement chunk validation, coherence scoring, outlier detection, verification queries, and error handling procedures for the ingestion and retrieval pipelines.

---

## Overview
> Sources: `opus-prd2-v3.md` (lines 431–479), `docs/SCHEMA_REFERENCE.md` (lines 810–840, QualityMetadata interface)

---

## Chunk Validation
> Source: `opus-prd2-v3.md` (lines 434–439)

```yaml
quality:
  validation:
    validate_all_chunks: true
    min_coherence_score: 0.6
    min_completeness_score: 0.5
    flag_outlier_embeddings: true
```

- **Coherence score:** Measures how semantically coherent a chunk is (min 0.6)
- **Completeness score:** Measures whether chunk contains complete thoughts (min 0.5)
- **Outlier detection:** Flag embeddings that are statistical outliers in the vector space

### Quality Metadata per Chunk
> Source: `docs/SCHEMA_REFERENCE.md` (lines 813–840)

```typescript
interface QualityMetadata {
  confidence_score: number;            // 0-1, overall quality confidence
  validation_status: 'pending' | 'validated' | 'flagged' | 'rejected';
  validation_details?: {
    validator: string;
    passed_checks: string[];
    failed_checks: string[];
    warnings: string[];
  };
  error_flags: Array<{
    error_type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    auto_fixed: boolean;
  }>;
  review_status: 'unreviewed' | 'auto_approved' | 'human_reviewed' | 'needs_review';
  chunking_quality: {
    coherence_score: number;
    completeness_score: number;
    boundary_quality: number;
  };
  embedding_quality?: {
    reconstruction_error?: number;
    outlier_score?: number;
  };
}
```

---

## Verification Queries
> Source: `opus-prd2-v3.md` (lines 441–457)

Post-ingestion verification queries to validate the system works correctly:

| Query | Type | Expected Domain | Max Latency |
|-------|------|----------------|-------------|
| "database connection setup" | text | codebase | 550ms |
| "system architecture diagram" | text | image/mixed modalities | 600ms |
| "Find code that implements this UI" + image | mixed | codebase | 700ms |

---

## Error Handling
> Source: `opus-prd2-v3.md` (lines 459–479)

### API Rate Limiting
```yaml
api_rate_limit:
  initial_backoff_seconds: 5
  max_backoff_seconds: 60
  max_retries: 5
```

### Embedding Failures
```yaml
embedding_failure:
  retry_count: 3
  fallback_to_text_only: true
```
- Retry up to 3 times
- If multimodal embedding fails, fall back to text-only model (Qwen3-Embedding-8B)

### Parse Failures
```yaml
parse_failure:
  log_file: "ingestion_errors.log"
  continue_on_error: true
  quarantine_failed: true
```
- Log errors to `ingestion_errors.log`
- Continue processing other files on error
- Quarantine failed files for manual review

### Multimodal Failures
```yaml
multimodal_failure:
  fallback_to_text_only: true
  log_visual_errors: true
```
- Fall back to text-only processing if visual processing fails
- Log visual processing errors separately

---

## Implementation Tasks

1. Create `src/quality/validator.py` — Chunk validation pipeline (coherence, completeness, boundary quality)
2. Create `src/quality/outlier_detector.py` — Embedding outlier detection in vector space
3. Create `src/quality/verification.py` — Post-ingestion verification query runner
4. Create `src/quality/error_handler.py` — Centralized error handling with retry/backoff/fallback logic
5. Create `src/quality/quarantine.py` — Failed file quarantine management
6. Implement coherence scoring algorithm (likely using embedding similarity of chunk parts)
7. Implement completeness scoring (sentence boundary analysis)

---

## Conflicts & Ambiguities

1. **Coherence scoring method:** The schema defines `coherence_score` but doesn't specify the algorithm. Common approaches: compute embedding similarity between first and second half of chunk, or use an LLM to rate coherence. Need to decide.

2. **Outlier detection method:** `flag_outlier_embeddings: true` but no algorithm specified. Options: z-score based, isolation forest, or simple distance-from-centroid. Need to decide.

3. **Validation timing:** Should validation run during ingestion (blocking) or as a post-processing step? The `validate_all_chunks: true` config suggests during ingestion, but this adds latency.
