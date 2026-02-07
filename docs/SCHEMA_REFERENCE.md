# Schema Reference Documentation
## Consolidated Schema Definitions from RAG v3.0 Architecture

**Version:** 3.0.0  
**Date:** January 9, 2026  
**Author:** Ice-ninja / Sliither  

---

## Table of Contents
1. [Chunk Metadata Schema](#chunk-metadata-schema)
2. [SQLite Database Schema](#sqlite-database-schema)
3. [Configuration Schema](#configuration-schema)
4. [TypeScript Interfaces](#typescript-interfaces)

---

## Chunk Metadata Schema

This schema is designed for cutting-edge RAG with full provenance tracking, semantic graph linkage, quality scoring, and cross-modal relationship mapping.

### Schema Overview: 12 Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHUNK METADATA SCHEMA v3.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. IDENTITY          │  2. PROVENANCE        │  3. CONTENT               │
│  - chunk_id (UUID)    │  - source_uri         │  - content_type           │
│  - content_hash       │  - git_commit_sha     │  - modalities[]           │
│  - version            │  - author             │  - language               │
│  - parent_chunk_id    │  - created_at         │  - mime_type              │
│                       │  - modified_at        │  - byte_size              │
├─────────────────────────────────────────────────────────────────────────────┤
│  4. STRUCTURE         │  5. HIERARCHY         │  6. SEMANTIC              │
│  - chunk_method       │  - depth_level        │  - topic_cluster_id       │
│  - token_count        │  - section_path[]     │  - entities[]             │
│  - char_count         │  - heading_text       │  - keywords[]             │
│  - overlap_prev       │  - parent_heading     │  - summary                │
│  - overlap_next       │  - sibling_ids[]      │  - intent_class           │
├─────────────────────────────────────────────────────────────────────────────┤
│  7. CODE_SPECIFIC     │  8. MULTIMODAL        │  9. EMBEDDING             │
│  - ast_node_type      │  - referenced_images[]│  - model_id               │
│  - parent_scope       │  - referenced_code[]  │  - dimensions             │
│  - signature          │  - cross_modal_links[]│  - mrl_truncated          │
│  - imports[]          │  - visual_elements[]  │  - quantization           │
│  - complexity_score   │  - ocr_text           │  - embedding_hash         │
│  - docstring          │  - diagram_type       │  - embedded_at            │
├─────────────────────────────────────────────────────────────────────────────┤
│  10. GRAPH            │  11. QUALITY          │  12. RETRIEVAL            │
│  - incoming_refs[]    │  - confidence_score   │  - access_count           │
│  - outgoing_refs[]    │  - validation_status  │  - retrieval_success_rate │
│  - semantic_neighbors[]│ - error_flags[]      │  - user_feedback_score    │
│  - coreference_chain  │  - review_status      │  - freshness_decay        │
│  - dependency_graph   │  - chunking_quality   │  - last_accessed_at       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SQLite Database Schema

Optimized for MemVid video-encoded storage with full metadata.

### Core chunks table with essential fields for fast retrieval

```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    corpus_id TEXT NOT NULL,
    root_document_id TEXT NOT NULL,
    
    -- Content
    raw_content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    modalities TEXT NOT NULL,           -- JSON array
    primary_modality TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    
    -- Structure
    chunking_method TEXT NOT NULL,
    parent_chunk_id TEXT,
    depth_level INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    
    -- Full metadata as JSON
    metadata_json TEXT NOT NULL,        -- Complete ChunkMetadata object
    
    -- Indexes
    FOREIGN KEY (parent_chunk_id) REFERENCES chunks(chunk_id)
);
```

### Embeddings table (separate for multiple embedding versions)

```sql
CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    mrl_truncated INTEGER DEFAULT 0,
    quantization TEXT,
    
    -- Vector stored as blob (or reference to MemVid frame)
    vector BLOB,
    memvid_frame_index INTEGER,         -- If stored in MemVid
    memvid_file TEXT,                   -- Which .mp4 file
    
    -- Metadata
    instruction_used TEXT,
    embedded_at TEXT NOT NULL,
    embedding_hash TEXT NOT NULL,
    
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
```

### Graph relationships (normalized for query efficiency)

```sql
CREATE TABLE chunk_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    evidence TEXT,
    created_at TEXT NOT NULL,
    
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(chunk_id)
);
```

### Semantic neighbors (precomputed for fast similar-chunk retrieval)

```sql
CREATE TABLE semantic_neighbors (
    chunk_id TEXT NOT NULL,
    neighbor_chunk_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    computed_at TEXT NOT NULL,
    model_id TEXT NOT NULL,
    
    PRIMARY KEY (chunk_id, neighbor_chunk_id, model_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (neighbor_chunk_id) REFERENCES chunks(chunk_id)
);
```

### Cross-modal links (for multimodal retrieval)

```sql
CREATE TABLE cross_modal_links (
    link_id TEXT PRIMARY KEY,
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    source_modality TEXT NOT NULL,
    target_modality TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    anchor_text TEXT,
    
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(chunk_id)
);
```

### Entities (for entity-centric retrieval)

```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT,
    knowledge_base_id TEXT              -- Link to external KB
);

CREATE TABLE chunk_entities (
    chunk_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    mention_text TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    confidence REAL NOT NULL,
    
    PRIMARY KEY (chunk_id, entity_id, start_offset),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);
```

### Retrieval analytics

```sql
CREATE TABLE retrieval_events (
    event_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    query_text TEXT,
    query_embedding_hash TEXT,
    retrieval_rank INTEGER,
    rerank_score REAL,
    was_selected INTEGER,               -- Did user select this result
    user_feedback INTEGER,              -- -1, 0, 1
    timestamp TEXT NOT NULL,
    
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
```

### MemVid index mapping

```sql
CREATE TABLE memvid_indices (
    memvid_file TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    
    PRIMARY KEY (memvid_file, frame_index),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id)
);
```

### Indexes for common queries

```sql
CREATE INDEX idx_chunks_corpus ON chunks(corpus_id);
CREATE INDEX idx_chunks_content_type ON chunks(content_type);
CREATE INDEX idx_chunks_document ON chunks(root_document_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_model ON embeddings(model_id);
CREATE INDEX idx_relationships_source ON chunk_relationships(source_chunk_id);
CREATE INDEX idx_relationships_target ON chunk_relationships(target_chunk_id);
CREATE INDEX idx_relationships_type ON chunk_relationships(relationship_type);
CREATE INDEX idx_neighbors_similarity ON semantic_neighbors(similarity_score DESC);
CREATE INDEX idx_cross_modal_source ON cross_modal_links(source_chunk_id);
CREATE INDEX idx_cross_modal_modality ON cross_modal_links(source_modality, target_modality);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_chunk_entities_entity ON chunk_entities(entity_id);
```

---

## Configuration Schema

### Embedding Models Configuration

```yaml
embedding:
  # PRIMARY: Qwen3-VL-Embedding-8B (Released Jan 7-8, 2026)
  primary:
    model_id: "Qwen/Qwen3-VL-Embedding-8B"
    model_type: "multimodal"
    parameters: 8140000000  # 8.14B
    layers: 36
    architecture: "qwen3_vl"
    
    dimensions:
      native: 4096
      mrl_supported: true
      mrl_options: [256, 512, 1024, 2048, 4096]
      storage: 1024          # Truncated for MemVid efficiency
      retrieval: 2048        # Higher precision for queries
    
    context:
      max_length: 32768
      default_length: 8192
    
    vision:
      min_pixels: 4096
      max_pixels: 1843200    # 1280x1440
      total_video_pixels: 7864320
      default_fps: 1.0
      default_frames: 64
      max_frames: 64
    
    supported_modalities:
      - text
      - image
      - video
      - screenshot
      - mixed_text_image
      - mixed_text_video
      - mixed_all
    
    quantization:
      supported: true
      options: ["bf16", "fp16", "int8", "int4"]
      recommended: "bf16"
    
    instruction_aware: true
    
    inference:
      torch_dtype: "bfloat16"
      attn_implementation: "flash_attention_2"
      device_map: "auto"
    
    benchmarks:
      mmeb_v2: 77.8
      mmeb_v2_rank: 1
      mmteb: 67.88
      image_retrieval: 80.0
      video_retrieval: 67.1
      visdoc_retrieval: 82.4
```

### Chunking Strategies Configuration

```yaml
chunking:
  # Method 1: Semantic (Pure Text)
  semantic:
    enabled: true
    similarity_threshold: 0.75
    window_size: 3  # sentences
    embedding_model: "boundary_detection"
    min_chunk_tokens: 128
    max_chunk_tokens: 2048
    applies_to:
      content_types: ["documentation", "research_paper"]
      modalities: ["text"]

  # Method 2: Recursive Hierarchical (Pure Text)
  recursive_hierarchical:
    enabled: true
    chunk_size_tokens: 1024
    overlap_tokens: 100
    separators:
      - "\n\n"    # Paragraphs
      - "\n"      # Lines
      - ". "      # Sentences
      - " "       # Words (last resort)
    applies_to:
      content_types: ["documentation", "conversation"]
      modalities: ["text"]

  # Method 3: AST Structural (Pure Code)
  ast_structural:
    enabled: true
    languages:
      python:
        parser: "tree-sitter-python"
        nodes: ["function_definition", "class_definition", "decorated_definition"]
      typescript:
        parser: "tree-sitter-typescript"
        nodes: ["function_declaration", "class_declaration", "method_definition", "interface_declaration"]
      javascript:
        parser: "tree-sitter-javascript"
        nodes: ["function_declaration", "class_declaration", "method_definition"]
      go:
        parser: "tree-sitter-go"
        nodes: ["function_declaration", "method_declaration", "type_declaration"]
      rust:
        parser: "tree-sitter-rust"
        nodes: ["function_item", "impl_item", "struct_item", "trait_item"]
      java:
        parser: "tree-sitter-java"
        nodes: ["method_declaration", "class_declaration", "constructor_declaration", "interface_declaration"]
    
    prepend_parent_context: true
    preserve_docstrings: true
    preserve_imports: true
    extract_dependencies: true
    compute_complexity: true
    fallback_to_fixed: true
    fallback_window_tokens: 512
    
    applies_to:
      content_types: ["code"]
      modalities: ["text"]
```

### Metadata Schema Configuration

```yaml
metadata:
  schema_version: "3.0.0"
  
  # Enable/disable metadata dimensions
  dimensions:
    identity:
      enabled: true
      generate_uuid_v7: true
      compute_content_hash: true
    
    provenance:
      enabled: true
      git_integration: true
      track_authors: true
      track_license: true
    
    content:
      enabled: true
      detect_language: true
      detect_modalities: true
    
    structure:
      enabled: true
      track_overlaps: true
      track_boundaries: true
    
    hierarchy:
      enabled: true
      max_depth: 10
      track_siblings: true
    
    semantic:
      enabled: true
      extract_entities: true
      extract_keywords: true
      generate_summaries: true
      classify_intent: true
      entity_types:
        - PERSON
        - ORG
        - PRODUCT
        - TECH
        - CONCEPT
        - LOCATION
        - DATE
        - CODE_ELEMENT
    
    code_specific:
      enabled: true
      extract_docstrings: true
      compute_complexity: true
      track_dependencies: true
      track_test_coverage: false
    
    multimodal:
      enabled: true
      extract_visual_elements: true
      run_ocr: true
      detect_diagram_types: true
      build_cross_modal_links: true
    
    embedding:
      enabled: true
      track_model_version: true
      compute_embedding_hash: true
    
    graph:
      enabled: true
      compute_semantic_neighbors: true
      neighbor_top_k: 10
      track_coreferences: false  # Expensive, optional
    
    quality:
      enabled: true
      validate_chunks: true
      compute_coherence: true
    
    retrieval:
      enabled: true
      track_access: true
      track_feedback: true
      compute_freshness_decay: true
```

---

## TypeScript Interfaces

### Complete Chunk Metadata Interface

```typescript
// ============================================================================
// CUTTING-EDGE RAG METADATA SCHEMA v3.0
// Enterprise Knowledge Graph with Full Provenance
// ============================================================================

// -----------------------------------------------------------------------------
// 1. IDENTITY: Unique identification and versioning
// -----------------------------------------------------------------------------
interface ChunkIdentity {
  chunk_id: string;                    // UUID v7 (time-sortable)
  content_hash: string;                // SHA-256 of raw content (deduplication)
  version: number;                     // Incremental version for updates
  parent_chunk_id: string | null;      // If this is a sub-chunk
  root_document_id: string;            // Original document this came from
  corpus_id: string;                   // Which corpus/domain (prompts/code/research)
}

// -----------------------------------------------------------------------------
// 2. PROVENANCE: Complete audit trail
// -----------------------------------------------------------------------------
interface ChunkProvenance {
  source_uri: string;                  // file://path or https://url
  source_type: 'local_file' | 'git_repo' | 'web_url' | 'api' | 'user_upload';
  git_metadata?: {
    repository: string;                // e.g., "github.com/user/repo"
    commit_sha: string;                // Full 40-char SHA
    branch: string;
    commit_timestamp: string;          // ISO 8601
    commit_author: string;
    file_path_in_repo: string;
  };
  author?: {
    name: string;
    email?: string;
    organization?: string;
  };
  license?: string;                    // SPDX identifier
  created_at: string;                  // ISO 8601 when chunk was created
  modified_at: string;                 // ISO 8601 last modification
  ingested_at: string;                 // ISO 8601 when ingested into RAG
  ingestion_pipeline_version: string;  // e.g., "3.0.0"
}

// -----------------------------------------------------------------------------
// 3. CONTENT: What this chunk contains
// -----------------------------------------------------------------------------
type ContentType = 
  | 'code' 
  | 'documentation' 
  | 'research_paper'
  | 'prompt'
  | 'configuration'
  | 'data'
  | 'conversation'
  | 'mixed';

type Modality = 'text' | 'image' | 'video' | 'audio' | 'screenshot' | 'diagram';

interface ChunkContent {
  content_type: ContentType;
  modalities: Modality[];              // What modalities are present
  primary_modality: Modality;          // Dominant modality
  language: {
    natural?: string;                  // ISO 639-1 (e.g., 'en', 'zh')
    programming?: string;              // e.g., 'python', 'typescript'
  };
  mime_type: string;                   // e.g., 'text/markdown', 'image/png'
  byte_size: number;
  encoding: string;                    // e.g., 'utf-8'
}

// -----------------------------------------------------------------------------
// 4. STRUCTURE: How this chunk was created
// -----------------------------------------------------------------------------
type ChunkingMethod = 
  | 'fixed_size'
  | 'sentence_based'
  | 'semantic'
  | 'recursive_hierarchical'
  | 'ast_structural'
  | 'multimodal_boundary'
  | 'manual';

interface ChunkStructure {
  chunking_method: ChunkingMethod;
  chunking_config: {
    target_tokens?: number;
    overlap_tokens?: number;
    similarity_threshold?: number;
    separators?: string[];
  };
  token_count: number;
  char_count: number;
  word_count: number;
  line_count: number;
  overlap: {
    previous_chunk_id?: string;
    previous_overlap_tokens: number;
    next_chunk_id?: string;
    next_overlap_tokens: number;
  };
  boundaries: {
    start_offset: number;              // Byte offset in source
    end_offset: number;
    start_line?: number;
    end_line?: number;
  };
}

// -----------------------------------------------------------------------------
// 5. HIERARCHY: Document structure preservation
// -----------------------------------------------------------------------------
interface ChunkHierarchy {
  depth_level: number;                 // 0 = root, 1 = section, 2 = subsection...
  section_path: string[];              // ["Chapter 1", "Introduction", "Background"]
  heading_text?: string;               // Current section heading
  parent_heading?: string;             // Parent section heading
  document_position: {
    section_index: number;             // Which section (0-indexed)
    chunk_index_in_section: number;    // Position within section
    total_chunks_in_section: number;
    global_chunk_index: number;        // Position in entire document
    total_document_chunks: number;
  };
  sibling_chunk_ids: string[];         // Other chunks at same level
  child_chunk_ids: string[];           // Sub-chunks if hierarchical
}

// -----------------------------------------------------------------------------
// 6. SEMANTIC: Extracted meaning and classification
// -----------------------------------------------------------------------------
interface NamedEntity {
  text: string;
  type: 'PERSON' | 'ORG' | 'PRODUCT' | 'TECH' | 'CONCEPT' | 'LOCATION' | 'DATE' | 'CODE_ELEMENT';
  confidence: number;
  start_offset: number;
  end_offset: number;
  linked_entity_id?: string;           // Link to knowledge base entity
}

interface ChunkSemantic {
  topic_cluster_id: string;            // Cluster assignment from topic modeling
  topic_keywords: string[];            // Top keywords for this topic
  topic_confidence: number;
  entities: NamedEntity[];
  keywords: Array<{
    term: string;
    tfidf_score: number;
    is_technical: boolean;
  }>;
  summary: string;                     // Auto-generated 1-2 sentence summary
  intent_classification: {
    primary_intent: string;            // e.g., 'explanation', 'tutorial', 'reference'
    confidence: number;
  };
  sentiment?: {
    polarity: number;                  // -1 to 1
    subjectivity: number;              // 0 to 1
  };
  reading_level?: string;              // e.g., 'technical', 'beginner', 'expert'
}

// -----------------------------------------------------------------------------
// 7. CODE-SPECIFIC: For code chunks only
// -----------------------------------------------------------------------------
type ASTNodeType = 
  | 'module'
  | 'class_definition'
  | 'function_definition'
  | 'method_definition'
  | 'decorator'
  | 'import_statement'
  | 'variable_declaration'
  | 'type_definition'
  | 'interface'
  | 'enum'
  | 'constant';

interface CodeMetadata {
  ast_node_type: ASTNodeType;
  parent_scope: string;                // e.g., "ClassName.method_name"
  fully_qualified_name: string;        // e.g., "module.ClassName.method_name"
  signature?: string;                  // Function/method signature
  return_type?: string;
  parameters?: Array<{
    name: string;
    type?: string;
    default_value?: string;
  }>;
  imports: Array<{
    module: string;
    items: string[];
    is_relative: boolean;
  }>;
  exports?: string[];
  docstring?: {
    summary: string;
    params?: Record<string, string>;
    returns?: string;
    raises?: string[];
    examples?: string[];
  };
  complexity: {
    cyclomatic: number;
    cognitive: number;
    lines_of_code: number;
    lines_of_comments: number;
  };
  dependencies: {
    internal: string[];                // References within same codebase
    external: string[];                // External package dependencies
  };
  test_coverage?: {
    covered: boolean;
    test_file?: string;
    coverage_percentage?: number;
  };
}

// -----------------------------------------------------------------------------
// 8. MULTIMODAL: Cross-modal relationships
// -----------------------------------------------------------------------------
interface VisualElement {
  element_id: string;
  element_type: 'figure' | 'table' | 'diagram' | 'screenshot' | 'equation' | 'chart';
  caption?: string;
  alt_text?: string;
  ocr_text?: string;                   // Extracted text from image
  bounding_box?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  source_url?: string;
}

interface CrossModalLink {
  link_id: string;
  source_chunk_id: string;
  target_chunk_id: string;
  source_modality: Modality;
  target_modality: Modality;
  relationship_type: 
    | 'references'                     // Text references image
    | 'illustrates'                    // Image illustrates text
    | 'implements'                     // Code implements documentation
    | 'documents'                      // Doc documents code
    | 'derives_from'                   // Derived/transformed content
    | 'related_to';                    // General semantic relationship
  confidence: number;
  anchor_text?: string;                // The text that creates the link
}

interface MultimodalMetadata {
  visual_elements: VisualElement[];
  referenced_images: string[];         // Image chunk IDs referenced
  referenced_code_blocks: string[];    // Code chunk IDs referenced
  referenced_videos: string[];         // Video chunk IDs referenced
  cross_modal_links: CrossModalLink[];
  diagram_analysis?: {
    diagram_type: 'flowchart' | 'sequence' | 'architecture' | 'erd' | 'uml' | 'other';
    extracted_nodes?: string[];
    extracted_relationships?: string[];
  };
  ocr_extraction?: {
    full_text: string;
    confidence: number;
    language_detected: string;
  };
}

// -----------------------------------------------------------------------------
// 9. EMBEDDING: Vector representation metadata
// -----------------------------------------------------------------------------
interface EmbeddingMetadata {
  model_id: string;                    // e.g., "qwen/qwen3-vl-embedding-8b"
  model_version: string;
  native_dimensions: number;           // Original output dims (e.g., 4096)
  stored_dimensions: number;           // After MRL truncation (e.g., 1024)
  mrl_truncated: boolean;
  quantization: {
    applied: boolean;
    method?: 'int8' | 'int4' | 'fp16' | 'bf16';
    bits?: number;
  };
  instruction_used?: string;           // Task instruction if instruction-aware
  embedding_hash: string;              // Hash of the embedding vector
  embedded_at: string;                 // ISO 8601
  embedding_latency_ms: number;
  input_modalities_embedded: Modality[];
}

// -----------------------------------------------------------------------------
// 10. GRAPH: Knowledge graph relationships
// -----------------------------------------------------------------------------
interface GraphReference {
  chunk_id: string;
  relationship_type: string;
  weight: number;                      // Strength of relationship
  evidence?: string;                   // Why this link exists
}

interface ChunkGraph {
  incoming_references: GraphReference[];    // Chunks that reference this one
  outgoing_references: GraphReference[];    // Chunks this one references
  semantic_neighbors: Array<{
    chunk_id: string;
    similarity_score: number;
    computed_at: string;
  }>;
  coreference_chain?: {
    chain_id: string;
    entity: string;
    mentions: Array<{
      chunk_id: string;
      mention_text: string;
      offset: number;
    }>;
  };
  dependency_position?: {
    topological_order: number;         // For code dependency graphs
    is_leaf: boolean;
    is_root: boolean;
    depth_from_root: number;
  };
}

// -----------------------------------------------------------------------------
// 11. QUALITY: Validation and confidence scores
// -----------------------------------------------------------------------------
type ValidationStatus = 'pending' | 'validated' | 'flagged' | 'rejected';
type ReviewStatus = 'unreviewed' | 'auto_approved' | 'human_reviewed' | 'needs_review';

interface QualityMetadata {
  confidence_score: number;            // 0-1, overall quality confidence
  validation_status: ValidationStatus;
  validation_details?: {
    validator: string;                 // Which validation ran
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
  review_status: ReviewStatus;
  reviewed_by?: string;
  reviewed_at?: string;
  chunking_quality: {
    coherence_score: number;           // How semantically coherent is this chunk
    completeness_score: number;        // Does it contain complete thoughts
    boundary_quality: number;          // How good are the chunk boundaries
  };
  embedding_quality?: {
    reconstruction_error?: number;
    outlier_score?: number;            // How unusual is this embedding
  };
}

// -----------------------------------------------------------------------------
// 12. RETRIEVAL: Usage analytics and optimization
// -----------------------------------------------------------------------------
interface RetrievalMetadata {
  access_count: number;                // How many times retrieved
  retrieval_success_rate: number;      // When retrieved, was it useful?
  user_feedback: {
    upvotes: number;
    downvotes: number;
    average_rating: number;
  };
  query_patterns: Array<{
    query_cluster_id: string;          // What types of queries retrieve this
    frequency: number;
  }>;
  freshness: {
    content_age_days: number;
    decay_factor: number;              // 0-1, how much to downweight old content
    is_evergreen: boolean;             // Does this content age?
  };
  last_accessed_at?: string;           // ISO 8601
  performance: {
    average_retrieval_rank: number;    // Where does this typically rank
    average_rerank_score: number;
    appeared_in_top_k: number;         // How often in top-k results
  };
}

// =============================================================================
// COMPLETE CHUNK METADATA INTERFACE
// =============================================================================
interface ChunkMetadata {
  // Core dimensions
  identity: ChunkIdentity;
  provenance: ChunkProvenance;
  content: ChunkContent;
  structure: ChunkStructure;
  hierarchy: ChunkHierarchy;
  semantic: ChunkSemantic;
  
  // Specialized dimensions
  code?: CodeMetadata;                 // Only for code chunks
  multimodal: MultimodalMetadata;
  embedding: EmbeddingMetadata;
  
  // Relationship dimensions
  graph: ChunkGraph;
  
  // Operational dimensions
  quality: QualityMetadata;
  retrieval: RetrievalMetadata;
  
  // Extension point
  custom?: Record<string, unknown>;    // Domain-specific extensions
}
``` 