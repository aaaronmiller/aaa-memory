# Topic: Metadata Schema (12 Dimensions)

## Summary
The 12-dimensional chunk metadata schema for the RAG v3.0 system, including TypeScript interfaces and YAML configuration for enabling/disabling dimensions.

---

## Schema Overview

> **Source:** [opus-prd1-v3.md](../opus-prd1-v3.md) (Phase 2), [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

The metadata schema has 12 dimensions, each capturing a different aspect of chunk information:

```
1. IDENTITY       — Unique identification and versioning
2. PROVENANCE     — Complete audit trail
3. CONTENT        — What the chunk contains
4. STRUCTURE      — How the chunk was created
5. HIERARCHY      — Document structure preservation
6. SEMANTIC       — Extracted meaning and classification
7. CODE_SPECIFIC  — Code-only metadata (AST, complexity, imports)
8. MULTIMODAL     — Cross-modal relationships
9. EMBEDDING      — Vector representation metadata
10. GRAPH         — Knowledge graph relationships
11. QUALITY       — Quality metrics and validation
12. RETRIEVAL     — Retrieval analytics and feedback
```

---

## Dimension 1: IDENTITY

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md), [opus-prd1-v3.md](../opus-prd1-v3.md)

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | string | UUID v7 (time-sortable) |
| content_hash | string | SHA-256 of raw content (deduplication) |
| version | number | Incremental version for updates |
| parent_chunk_id | string/null | If this is a sub-chunk |
| root_document_id | string | Original document this came from |
| corpus_id | string | Which corpus/domain (prompts/code/research) |

**Config:** `generate_uuid_v7: true`, `compute_content_hash: true`

---

## Dimension 2: PROVENANCE

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md), [opus-prd1-v3.md](../opus-prd1-v3.md)

| Field | Type | Description |
|-------|------|-------------|
| source_uri | string | file://path or https://url |
| source_type | enum | local_file, git_repo, web_url, api, user_upload |
| git_metadata | object | repository, commit_sha, branch, timestamp, author, file_path |
| author | object | name, email, organization |
| license | string | SPDX identifier |
| created_at | string | ISO 8601 |
| modified_at | string | ISO 8601 |
| ingested_at | string | ISO 8601 |
| ingestion_pipeline_version | string | e.g., "3.0.0" |

**Config:** `git_integration: true`, `track_authors: true`, `track_license: true`

---

## Dimension 3: CONTENT

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| content_type | enum | code, documentation, research_paper, prompt, configuration, data, conversation, mixed |
| modalities | Modality[] | text, image, video, audio, screenshot, diagram |
| primary_modality | Modality | Dominant modality |
| language.natural | string | ISO 639-1 (e.g., 'en') |
| language.programming | string | e.g., 'python', 'typescript' |
| mime_type | string | e.g., 'text/markdown' |
| byte_size | number | Size in bytes |
| encoding | string | e.g., 'utf-8' |

**Config:** `detect_language: true`, `detect_modalities: true`

---

## Dimension 4: STRUCTURE

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| chunking_method | enum | fixed_size, sentence_based, semantic, recursive_hierarchical, ast_structural, multimodal_boundary, manual |
| chunking_config | object | target_tokens, overlap_tokens, similarity_threshold, separators |
| token_count | number | Token count |
| char_count | number | Character count |
| word_count | number | Word count |
| line_count | number | Line count |
| overlap.previous_chunk_id | string | Previous chunk reference |
| overlap.previous_overlap_tokens | number | Overlap with previous |
| overlap.next_chunk_id | string | Next chunk reference |
| overlap.next_overlap_tokens | number | Overlap with next |
| boundaries.start_offset | number | Byte offset in source |
| boundaries.end_offset | number | End byte offset |
| boundaries.start_line | number | Start line number |
| boundaries.end_line | number | End line number |

**Config:** `track_overlaps: true`, `track_boundaries: true`

---

## Dimension 5: HIERARCHY

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| depth_level | number | 0=root, 1=section, 2=subsection... |
| section_path | string[] | e.g., ["Chapter 1", "Introduction", "Background"] |
| heading_text | string | Current section heading |
| parent_heading | string | Parent section heading |
| document_position | object | section_index, chunk_index_in_section, total_chunks_in_section, global_chunk_index, total_document_chunks |
| sibling_chunk_ids | string[] | Other chunks at same level |
| child_chunk_ids | string[] | Sub-chunks if hierarchical |

**Config:** `max_depth: 10`, `track_siblings: true`

---

## Dimension 6: SEMANTIC

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| topic_cluster_id | string | Cluster assignment from topic modeling |
| topic_keywords | string[] | Top keywords for this topic |
| topic_confidence | number | Confidence score |
| entities | NamedEntity[] | Extracted named entities with type, confidence, offsets |
| keywords | object[] | term, tfidf_score, is_technical |
| summary | string | Auto-generated 1-2 sentence summary |
| intent_classification | object | primary_intent (explanation/tutorial/reference), confidence |
| sentiment | object | polarity (-1 to 1), subjectivity (0 to 1) |
| reading_level | string | technical, beginner, expert |

**Entity Types:** PERSON, ORG, PRODUCT, TECH, CONCEPT, LOCATION, DATE, CODE_ELEMENT

**Config:** `extract_entities: true`, `extract_keywords: true`, `generate_summaries: true`, `classify_intent: true`

---

## Dimension 7: CODE_SPECIFIC

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| ast_node_type | enum | module, class_definition, function_definition, method_definition, etc. |
| parent_scope | string | e.g., "ClassName.method_name" |
| fully_qualified_name | string | e.g., "module.ClassName.method_name" |
| signature | string | Function/method signature |
| return_type | string | Return type |
| parameters | object[] | name, type, default_value |
| imports | object[] | module, items, is_relative |
| exports | string[] | Exported symbols |
| docstring | object | summary, params, returns, raises, examples |
| complexity | object | cyclomatic, cognitive, lines_of_code, lines_of_comments |
| dependencies | object | internal (same codebase), external (packages) |
| test_coverage | object | covered, test_file, coverage_percentage |

**Config:** `extract_docstrings: true`, `compute_complexity: true`, `track_dependencies: true`, `track_test_coverage: false`

---

## Dimension 8: MULTIMODAL

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| visual_elements | VisualElement[] | figure, table, diagram, screenshot, equation, chart |
| referenced_images | string[] | Image chunk IDs referenced |
| referenced_code_blocks | string[] | Code chunk IDs referenced |
| referenced_videos | string[] | Video chunk IDs referenced |
| cross_modal_links | CrossModalLink[] | Links between modalities |
| diagram_analysis | object | diagram_type, extracted_nodes, extracted_relationships |
| ocr_extraction | object | full_text, confidence, language_detected |

**CrossModalLink relationship types:** references, illustrates, implements, documents, derives_from, related_to

**Config:** `extract_visual_elements: true`, `run_ocr: true`, `detect_diagram_types: true`, `build_cross_modal_links: true`

---

## Dimension 9: EMBEDDING

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| model_id | string | e.g., "qwen/qwen3-vl-embedding-8b" |
| model_version | string | Model version |
| native_dimensions | number | Original output dims (e.g., 4096) |
| stored_dimensions | number | After MRL truncation (e.g., 1024) |
| mrl_truncated | boolean | Whether MRL was applied |
| quantization | string | bf16, fp16, int8, int4 |
| instruction_used | string | The instruction prefix used |
| embedding_hash | string | Hash of the embedding vector |
| embedded_at | string | ISO 8601 timestamp |

**Config:** `track_model_version: true`, `compute_embedding_hash: true`

---

## Dimension 10: GRAPH

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| incoming_refs | object[] | source_chunk_id, relationship_type, weight |
| outgoing_refs | object[] | target_chunk_id, relationship_type, weight |
| semantic_neighbors | object[] | chunk_id, similarity_score, model_id |
| coreference_chain | string | Coreference chain ID |
| dependency_graph | object | upstream_ids, downstream_ids |

**Config:** `compute_semantic_neighbors: true`, `neighbor_top_k: 10`, `track_coreferences: false` (expensive, optional)

---

## Dimension 11: QUALITY

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| confidence_score | number | Overall confidence (0-1) |
| validation_status | enum | valid, warning, error, pending |
| error_flags | string[] | List of detected issues |
| review_status | enum | auto_approved, needs_review, reviewed, rejected |
| chunking_quality | object | coherence_score, completeness_score, boundary_quality |

**Config:** `validate_chunks: true`, `compute_coherence: true`

---

## Dimension 12: RETRIEVAL

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Field | Type | Description |
|-------|------|-------------|
| access_count | number | Times retrieved |
| retrieval_success_rate | number | How often selected after retrieval |
| user_feedback_score | number | Aggregated user feedback |
| freshness_decay | number | Time-based relevance decay |
| last_accessed_at | string | ISO 8601 |

**Config:** `track_access: true`, `track_feedback: true`, `compute_freshness_decay: true`

---

## YAML Configuration Reference

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

All 12 dimensions can be individually enabled/disabled via YAML config under `metadata.dimensions.<dimension>.enabled`.

---

## Implementation Requirements

1. Define TypeScript interfaces for all 12 dimensions (reference code in SCHEMA_REFERENCE.md)
2. Implement Pydantic models (Python) matching the TypeScript interfaces
3. Build metadata extraction pipeline for each dimension
4. Create configuration loader for enabling/disabling dimensions
5. Implement content_hash computation (SHA-256)
6. Implement UUID v7 generation for chunk_id

---

## Conflicts / Ambiguities

- **⚠️ Schema completeness:** The UNIFIED_PRD.md schema overview shows fewer fields per dimension than the full TypeScript interfaces in SCHEMA_REFERENCE.md. The TypeScript interfaces are the authoritative source.
- **⚠️ Hierarchy fields:** The overview diagram shows `sibling_ids[]` but the TypeScript interface uses `sibling_chunk_ids` and adds `child_chunk_ids`. Use the TypeScript interface names.