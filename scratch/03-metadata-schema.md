# Topic 03: Metadata Schema (12 Dimensions)

## Summary
Implement the 12-dimensional chunk metadata schema that provides full provenance tracking, semantic graph linkage, quality scoring, and cross-modal relationship mapping for every chunk in the system.

---

## Overview
> Sources: `chatgpt5.2-prd.md` (lines 176–214, TypeScript interfaces lines 217–500+), `opus-prd1-v3.md`, `docs/SCHEMA_REFERENCE.md` (lines 18–57, TypeScript lines 476–897), `opus-prd2-v3.md` (lines 211–295), `docs/UNIFIED_PRD.md` (lines 169–207)

The metadata schema has 12 dimensions. Each chunk carries a complete `ChunkMetadata` object containing all applicable dimensions.

---

## Dimension Details

### 1. IDENTITY — Unique identification and versioning
- `chunk_id`: string — UUID v7 (time-sortable)
- `content_hash`: string — SHA-256 of raw content (for deduplication)
- `version`: number — Incremental version for updates
- `parent_chunk_id`: string | null — If this is a sub-chunk
- `root_document_id`: string — Original document this came from
- `corpus_id`: string — Which corpus/domain (prompts/code/research)

**Config:** `generate_uuid_v7: true`, `compute_content_hash: true`

### 2. PROVENANCE — Complete audit trail
- `source_uri`: string — file://path or https://url
- `source_type`: enum — local_file | git_repo | web_url | api | user_upload
- `git_metadata?`: object — repository, commit_sha, branch, commit_timestamp, commit_author, file_path_in_repo
- `author?`: object — name, email, organization
- `license?`: string — SPDX identifier
- `created_at`: string — ISO 8601
- `modified_at`: string — ISO 8601
- `ingested_at`: string — ISO 8601
- `ingestion_pipeline_version`: string

**Config:** `git_integration: true`, `track_authors: true`, `track_license: true`

### 3. CONTENT — What this chunk contains
- `content_type`: enum — code | documentation | research_paper | prompt | configuration | data | conversation | mixed
- `modalities`: Modality[] — text | image | video | audio | screenshot | diagram
- `primary_modality`: Modality
- `language`: object — natural (ISO 639-1), programming (e.g., 'python')
- `mime_type`: string
- `byte_size`: number
- `encoding`: string (e.g., 'utf-8')

**Config:** `detect_language: true`, `detect_modalities: true`

### 4. STRUCTURE — How this chunk was created
- `chunking_method`: enum — fixed_size | sentence_based | semantic | recursive_hierarchical | ast_structural | multimodal_boundary | manual
- `chunking_config`: object — target_tokens, overlap_tokens, similarity_threshold, separators
- `token_count`: number
- `char_count`: number
- `word_count`: number
- `line_count`: number
- `overlap`: object — previous_chunk_id, previous_overlap_tokens, next_chunk_id, next_overlap_tokens
- `boundaries`: object — start_offset, end_offset, start_line, end_line

**Config:** `track_overlaps: true`, `track_boundaries: true`

### 5. HIERARCHY — Document structure preservation
- `depth_level`: number — 0 = root, 1 = section, 2 = subsection...
- `section_path`: string[] — e.g., ["Chapter 1", "Introduction", "Background"]
- `heading_text?`: string
- `parent_heading?`: string
- `document_position`: object — section_index, chunk_index_in_section, total_chunks_in_section, global_chunk_index, total_document_chunks
- `sibling_chunk_ids`: string[]
- `child_chunk_ids`: string[]

**Config:** `max_depth: 10`, `track_siblings: true`

### 6. SEMANTIC — Extracted meaning and classification
- `topic_cluster_id`: string
- `topic_keywords`: string[]
- `topic_confidence`: number
- `entities`: NamedEntity[] — text, type (PERSON|ORG|PRODUCT|TECH|CONCEPT|LOCATION|DATE|CODE_ELEMENT), confidence, offsets, linked_entity_id
- `keywords`: Array<{term, tfidf_score, is_technical}>
- `summary`: string — Auto-generated 1-2 sentence summary
- `intent_classification`: object — primary_intent, confidence
- `sentiment?`: object — polarity (-1 to 1), subjectivity (0 to 1)
- `reading_level?`: string

**Config:** `extract_entities: true`, `extract_keywords: true`, `generate_summaries: true`, `classify_intent: true`

### 7. CODE_SPECIFIC — For code chunks only
- `ast_node_type`: enum — module | class_definition | function_definition | method_definition | decorator | import_statement | variable_declaration | type_definition | interface | enum | constant
- `parent_scope`: string
- `fully_qualified_name`: string
- `signature?`: string
- `return_type?`: string
- `parameters?`: Array<{name, type, default_value}>
- `imports`: Array<{module, items, is_relative}>
- `exports?`: string[]
- `docstring?`: object — summary, params, returns, raises, examples
- `complexity`: object — cyclomatic, cognitive, lines_of_code, lines_of_comments
- `dependencies`: object — internal[], external[]
- `test_coverage?`: object — covered, test_file, coverage_percentage

**Config:** `extract_docstrings: true`, `compute_complexity: true`, `track_dependencies: true`, `track_test_coverage: false`

### 8. MULTIMODAL — Cross-modal relationships
- `visual_elements`: VisualElement[] — element_id, element_type (figure|table|diagram|screenshot|equation|chart), caption, alt_text, ocr_text, bounding_box, source_url
- `referenced_images`: string[]
- `referenced_code_blocks`: string[]
- `referenced_videos`: string[]
- `cross_modal_links`: CrossModalLink[] — link_id, source/target chunk_id, source/target modality, relationship_type (references|illustrates|implements|documents|derives_from|related_to), confidence, anchor_text
- `diagram_analysis?`: object — diagram_type, extracted_nodes, extracted_relationships
- `ocr_extraction?`: object — full_text, confidence, language_detected

**Config:** `extract_visual_elements: true`, `run_ocr: true`, `detect_diagram_types: true`, `build_cross_modal_links: true`

### 9. EMBEDDING — Vector representation metadata
- `model_id`: string
- `model_version`: string
- `native_dimensions`: number (e.g., 4096)
- `stored_dimensions`: number (e.g., 1024 after MRL truncation)
- `mrl_truncated`: boolean
- `quantization`: object — applied, method, bits
- `instruction_used?`: string
- `embedding_hash`: string
- `embedded_at`: string — ISO 8601
- `embedding_latency_ms`: number
- `input_modalities_embedded`: Modality[]

**Config:** `track_model_version: true`, `compute_embedding_hash: true`

### 10. GRAPH — Knowledge graph relationships
- `incoming_references`: GraphReference[] — chunk_id, relationship_type, weight, evidence
- `outgoing_references`: GraphReference[]
- `semantic_neighbors`: Array<{chunk_id, similarity_score, computed_at}>
- `coreference_chain?`: object — chain_id, entity, mentions[]
- `dependency_position?`: object — topological_order, is_leaf, is_root, depth_from_root

**Config:** `compute_semantic_neighbors: true`, `neighbor_top_k: 10`, `track_coreferences: false` (expensive, optional)

### 11. QUALITY — Validation and confidence scores
- `confidence_score`: number (0-1)
- `validation_status`: enum — pending | validated | flagged | rejected
- `validation_details?`: object — validator, passed_checks, failed_checks, warnings
- `error_flags`: Array<{error_type, severity (low|medium|high|critical), message, auto_fixed}>
- `review_status`: enum — unreviewed | auto_approved | human_reviewed | needs_review
- `reviewed_by?`: string
- `reviewed_at?`: string
- `chunking_quality`: object — coherence_score, completeness_score, boundary_quality
- `embedding_quality?`: object — reconstruction_error, outlier_score

**Config:** `validate_chunks: true`, `compute_coherence: true`

### 12. RETRIEVAL — Usage analytics and optimization
- `access_count`: number
- `retrieval_success_rate`: number
- `user_feedback`: object — upvotes, downvotes, average_rating
- `query_patterns`: Array<{query_cluster_id, frequency}>
- `freshness`: object — content_age_days, decay_factor (0-1), is_evergreen
- `last_accessed_at?`: string — ISO 8601
- `performance`: object — average_retrieval_rank, average_rerank_score, appeared_in_top_k

**Config:** `track_access: true`, `track_feedback: true`, `compute_freshness_decay: true`

---

## Complete TypeScript Interface
> Source: `docs/SCHEMA_REFERENCE.md` (lines 873–896)

```typescript
interface ChunkMetadata {
  identity: ChunkIdentity;
  provenance: ChunkProvenance;
  content: ChunkContent;
  structure: ChunkStructure;
  hierarchy: ChunkHierarchy;
  semantic: ChunkSemantic;
  code?: CodeMetadata;           // Only for code chunks
  multimodal: MultimodalMetadata;
  embedding: EmbeddingMetadata;
  graph: ChunkGraph;
  quality: QualityMetadata;
  retrieval: RetrievalMetadata;
  custom?: Record<string, unknown>;  // Extension point
}
```

---

## Implementation Tasks

1. Define Pydantic models (Python) for all 12 metadata dimensions
2. Define TypeScript interfaces (matching the schema reference)
3. Implement UUID v7 generation for chunk_id
4. Implement SHA-256 content hashing for deduplication
5. Implement metadata population pipeline (each dimension populated at different stages)
6. Create validation logic for required vs optional fields per content type

---

## Conflicts & Ambiguities

1. **Schema version in YAML vs TypeScript:** Both `opus-prd2-v3.md` and `docs/SCHEMA_REFERENCE.md` agree on v3.0.0. No conflict.
2. **`root_document_id` vs `corpus_id`:** Identity dimension has both. `root_document_id` is the specific source document; `corpus_id` is the domain (prompts/code/research). These are complementary, not redundant.
3. **Code-specific dimension is optional:** Only populated for code chunks. The `ChunkMetadata` interface marks it as `code?: CodeMetadata`.
