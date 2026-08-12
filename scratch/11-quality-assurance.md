# Topic: Quality Assurance & Error Handling

## Summary
Chunk validation, coherence scoring, verification queries, outlier detection, and error handling procedures for the RAG v3.0 system.

---

## Chunk Validation

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

```yaml
quality:
  validation:
    validate_all_chunks: true
    min_coherence_score: 0.6
    min_completeness_score: 0.5
    flag_outlier_embeddings: true
```

- All chunks are validated after creation
- Minimum coherence score: 0.6 (how well the chunk holds together semantically)
- Minimum completeness score: 0.5 (whether the chunk contains a complete thought)
- Outlier embeddings are flagged for review

---

## Quality Metadata (Dimension 11)

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

Each chunk carries quality metadata:

| Field | Type | Description |
|-------|------|-------------|
| confidence_score | number | Overall confidence (0-1) |
| validation_status | enum | valid, warning, error, pending |
| error_flags | string[] | List of detected issues |
| review_status | enum | auto_approved, needs_review, reviewed, rejected |
| chunking_quality.coherence_score | number | Semantic coherence |
| chunking_quality.completeness_score | number | Thought completeness |
| chunking_quality.boundary_quality | number | How clean the chunk boundaries are |

---

## Verification Queries

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

Post-ingestion verification queries to validate the system works correctly:

| Query | Type | Expected | Max Latency |
|-------|------|----------|-------------|
| "database connection setup" | text | codebase domain | 550ms |
| "system architecture diagram" | text | image/mixed modalities | 600ms |
| "Find code that implements this UI" + test_screenshot.png | mixed | codebase domain | 700ms |

---

## Error Handling

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

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
- Fall back to text-only Qwen3-Embedding-8B if multimodal embedding fails

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
- Fall back to text-only processing
- Log visual processing errors separately

---

## Implementation Requirements

1. Implement chunk coherence scoring algorithm
2. Implement chunk completeness scoring algorithm
3. Build outlier embedding detection (statistical outlier in vector space)
4. Create validation pipeline that runs after each chunk creation
5. Implement verification query test suite
6. Build exponential backoff retry logic for API calls
7. Implement fallback chain (multimodal → text-only)
8. Create error quarantine system for failed files
9. Build ingestion error logging

---

## Conflicts / Ambiguities

- **⚠️ Scoring algorithms undefined:** The documents specify minimum scores (0.6 coherence, 0.5 completeness) but don't define how these scores are computed. Implementation needs to determine the scoring methodology (e.g., embedding-based coherence, LLM-based completeness).
- **⚠️ Outlier detection method:** "flag_outlier_embeddings" is specified but the detection method (z-score, IQR, isolation forest, etc.) is not defined.
