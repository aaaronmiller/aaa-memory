


CLAUDE OUTPUT:
---

## date: 2026-01-09 15:45:00 PST ver: 3.0.0 author: the penitent Sliither model: claude-opus-4-5-20250514 tags: [rag, qwen3-vl-embedding, multimodal, knowledge-graph, memvid, agentic-swarm, cutting-edge, 2026]

# Cutting-Edge RAG Architecture v3.0: Multimodal Agentic Swarms with Knowledge Graph Metadata (January 2026)

## Executive Summary

This is the definitive architecture for Ice-ninja's next-generation RAG system, incorporating the **Qwen3-VL-Embedding-8B** multimodal model released January 7-8, 2026. This architecture represents the absolute state-of-the-art as of January 9, 2026, featuring unified text-image-video embeddings, a sophisticated knowledge graph metadata schema rivaling enterprise systems, and agentic orchestration via headless Claude Code.

**What Makes This Cutting-Edge:**

- **Qwen3-VL-Embedding-8B**: MMEB-V2 rank #1 (77.8 score), multimodal unified vector space
- **Qwen3-VL-Reranker-8B**: Cross-attention precision reranking for query-document pairs
- **Knowledge Graph Metadata**: 12-dimension schema with provenance chains, semantic linkage, quality scoring
- **Cross-Modal Retrieval**: Query with text, retrieve images/code/video (or any combination)
- **MemVid Multimodal**: Video-encoded storage supporting mixed-modality chunks

---

## Phase 1: Embedding Model Stack (January 2026 SOTA)

### Primary: Qwen3-VL-Embedding-8B

Released **January 7-8, 2026** (arXiv:2601.04720). This is NOT the text-only Qwen3-Embedding-8B from June 2025. This is a **vision-language multimodal** embedding model.

|Specification|Qwen3-VL-Embedding-8B|Qwen3-VL-Embedding-2B|
|---|---|---|
|Parameters|8.14B|2.13B|
|Layers|36|28|
|Context Length|32,768 tokens|32,768 tokens|
|Embedding Dimensions|4096|2048|
|MMEB-V2 Score|**77.8** (Rank #1)|73.2|
|MMTEB Score|67.88|63.87|
|MRL Support|✅ Flexible dimensions|✅|
|Quantization|✅|✅|
|Instruction-Aware|✅|✅|

**Supported Input Modalities:**

- Pure text
- Pure image
- Pure video
- Text + image (mixed)
- Text + video (mixed)
- Image + video (mixed)
- Text + image + video (mixed)
- Screenshots (treated as images with OCR awareness)

**Architecture: Dual-Tower**

- Single/mixed-modal input → high-dimensional semantic vector
- Extracts `[EOS]` token hidden state from last layer as final representation
- Enables efficient independent encoding for large-scale retrieval

### Reranker: Qwen3-VL-Reranker-8B

Two-stage retrieval pipeline requires precision reranking after initial recall.

|Specification|Qwen3-VL-Reranker-8B|Qwen3-VL-Reranker-2B|
|---|---|---|
|Parameters|8.14B|2.13B|
|Architecture|Single-Tower|Single-Tower|
|Input|(Query, Document) pairs|(Query, Document) pairs|
|Output|Relevance score|Relevance score|
|Mechanism|Cross-Attention|Cross-Attention|

**Architecture: Single-Tower**

- Receives `(Query, Document)` pair (both can be mixed-modal)
- Cross-Attention for deep inter-modal interaction
- Relevance score via `yes`/`no` token generation probability

### Comparison: VL-Embedding vs Text-Only Embedding

|Capability|Qwen3-VL-Embedding-8B|Qwen3-Embedding-8B|
|---|---|---|
|Text Embedding|✅|✅|
|Code Embedding|✅|✅|
|Image Embedding|✅|❌|
|Video Embedding|✅|❌|
|Screenshot/Diagram|✅ (OCR-aware)|❌|
|Mixed Modal|✅|❌|
|MTEB Score|67.88|**70.58**|
|MMEB-V2 Score|**77.8**|N/A|

**Decision:** Use **Qwen3-VL-Embedding-8B** as primary for Ice-ninja's multi-domain corpus which includes:

- Code (benefits from screenshot/diagram understanding)
- Research papers (figures, tables, equations as images)
- Documentation (architectural diagrams, UI screenshots)

For pure text retrieval benchmarks, the text-only model scores higher on MTEB, but the VL model's multimodal capabilities are essential for real-world mixed content.

### Model Initialization

```python
import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker

# Primary Embedding Model
embedder = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-8B",
    max_length=8192,
    min_pixels=4096,
    max_pixels=1843200,        # 1280×1440 resolution
    total_pixels=7864320,      # Max total for video frames
    fps=1.0,                   # Video sampling rate
    num_frames=64,
    max_frames=64,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Precision Reranker
reranker = Qwen3VLReranker(
    model_name_or_path="Qwen/Qwen3-VL-Reranker-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

---

## Phase 2: Enterprise Knowledge Graph Metadata Schema

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

### Complete TypeScript Interface

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

### SQLite Schema for MemVid Integration

```sql
-- =============================================================================
-- CUTTING-EDGE RAG SCHEMA v3.0
-- Optimized for MemVid video-encoded storage with full metadata
-- =============================================================================

-- Core chunks table with essential fields for fast retrieval
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

-- Embeddings table (separate for multiple embedding versions)
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

-- Graph relationships (normalized for query efficiency)
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

-- Semantic neighbors (precomputed for fast similar-chunk retrieval)
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

-- Cross-modal links (for multimodal retrieval)
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

-- Entities (for entity-centric retrieval)
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

-- Retrieval analytics
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

-- MemVid index mapping
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

-- Indexes for common queries
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

## Phase 3: Multimodal Chunking Strategy

### Content Router with Modality Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL CONTENT ROUTER                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Input Analysis:                                                        │
│  1. Detect file type and MIME                                          │
│  2. Identify modalities present (text, images, video, mixed)           │
│  3. Route to appropriate chunking pipeline                             │
└─────────────────────────────────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬─────────────┬──────────────┐
    │         │         │             │              │
    ▼         ▼         ▼             ▼              ▼
 PURE      PURE      PURE         MIXED          MIXED
 TEXT      CODE      IMAGE        TEXT+IMG       CODE+IMG
   │         │         │             │              │
   ▼         ▼         ▼             ▼              ▼
Semantic   AST      Single       Multimodal    Screenshot
Chunking  Parsing   Chunk       Boundary       +Code
                               Detection       Fusion
```

### Method 1: Semantic Chunking (Pure Text)

Unchanged from v2 - uses Qwen3-Embedding-0.6B for boundary detection with 0.75 similarity threshold.

### Method 2: Recursive Hierarchical (Pure Text)

Unchanged from v2 - separator hierarchy with 1024 token target.

### Method 3: AST Structural (Pure Code)

Enhanced with parent scope prepending and dependency extraction.

### Method 4: Fixed-Size (Data/Config)

Unchanged - 512 tokens, 50 overlap.

### Method 5: Multimodal Boundary Detection (NEW)

For documents with mixed text and images:

```python
class MultimodalChunker:
    def __init__(self, embedder: Qwen3VLEmbedder):
        self.embedder = embedder
        
    def chunk_mixed_document(self, document: Document) -> List[MultimodalChunk]:
        """
        Chunk documents containing text, images, and diagrams.
        Preserves figure-caption relationships and diagram context.
        """
        chunks = []
        
        # 1. Identify visual elements and their positions
        visual_elements = self.extract_visual_elements(document)
        
        # 2. For each visual element, find surrounding context
        for element in visual_elements:
            # Get caption if present
            caption = self.find_caption(element, document)
            
            # Get surrounding text (1 paragraph before/after)
            context = self.extract_context(element, document, window=1)
            
            # Create multimodal chunk
            chunk = MultimodalChunk(
                modalities=['text', element.modality],
                text_content=f"{context.before}\n\n{caption}\n\n{context.after}",
                visual_content=element.content,
                cross_modal_links=[
                    CrossModalLink(
                        relationship_type='illustrates',
                        anchor_text=caption
                    )
                ]
            )
            chunks.append(chunk)
        
        # 3. Chunk remaining pure text using semantic method
        text_only_sections = self.extract_text_only_sections(document, visual_elements)
        for section in text_only_sections:
            text_chunks = self.semantic_chunk(section)
            chunks.extend(text_chunks)
        
        return chunks
```

### Method 6: Screenshot + Code Fusion (NEW)

For code with associated UI screenshots or architecture diagrams:

```python
class ScreenshotCodeFusion:
    """
    Handles code files that have associated visual documentation.
    Common in:
    - UI component libraries (code + rendered preview)
    - Architecture docs (code + system diagram)
    - API documentation (code + request/response screenshots)
    """
    
    def fuse_code_and_visuals(
        self, 
        code_file: str, 
        visual_files: List[str]
    ) -> List[FusedChunk]:
        # Parse code with AST
        code_chunks = self.ast_chunker.chunk(code_file)
        
        # Match visuals to code chunks based on:
        # 1. Filename similarity
        # 2. OCR text matching
        # 3. Reference comments in code
        for chunk in code_chunks:
            matching_visuals = self.find_matching_visuals(chunk, visual_files)
            
            if matching_visuals:
                # Create fused multimodal chunk
                chunk.add_visual_elements(matching_visuals)
                chunk.modalities.extend(['screenshot', 'diagram'])
        
        return code_chunks
```

---

## Phase 4: Agentic Swarm Architecture (Updated)

### Multimodal-Aware Agent Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│              HEADLESS ORCHESTRATOR (Claude Code)                        │
│                   orchestration_logic_v3.md                             │
│                                                                         │
│   Capabilities:                                                         │
│   - Multimodal content detection                                       │
│   - Cross-modal relationship discovery                                 │
│   - Quality validation with visual verification                        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
     ▼                        ▼                        ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  MODALITY   │       │   CONTENT   │       │   GRAPH     │
│  DETECTOR   │       │   ROUTER    │       │   BUILDER   │
│             │       │             │       │             │
│ - MIME type │       │ - Code      │       │ - Entity    │
│ - Visual    │       │ - Text      │       │   extract   │
│   elements  │       │ - Mixed     │       │ - Link      │
│ - OCR       │       │ - Visual    │       │   discover  │
└─────────────┘       └─────────────┘       └─────────────┘
     │                        │                        │
     ▼                        ▼                        ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    CODE     │       │    TEXT     │       │  MULTIMODAL │
│ SPECIALIST  │       │ SPECIALIST  │       │  SPECIALIST │
│             │       │             │       │             │
│ - AST Parse │       │ - Semantic  │       │ - Boundary  │
│ - Methods   │       │ - Recursive │       │   detection │
│   3 & 4     │       │   1 & 2     │       │ - Fusion    │
└─────────────┘       └─────────────┘       └─────────────┘
     │                        │                        │
     └────────────────────────┼────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   INTEGRATION   │
                    │     AGENT       │
                    │                 │
                    │ - Embed with    │
                    │   Qwen3-VL-8B   │
                    │ - Cluster       │
                    │ - MemVid encode │
                    │ - Metadata gen  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    MEMVID       │
                    │  MULTIMODAL     │
                    │    STORE        │
                    │                 │
                    │ codebase.mp4    │
                    │ research.mp4   │
                    │ prompts.mp4    │
                    └─────────────────┘
```

### Multimodal Embedding Call

```python
# Example: Embedding mixed-modal chunks with Qwen3-VL-Embedding-8B

inputs = [
    # Pure text
    {
        "text": "The UserDB class handles database connections...",
        "instruction": "Represent this code documentation for retrieval"
    },
    # Pure code (embedded as text with code instruction)
    {
        "text": "class UserDB:\n    def __init__(self):\n        ...",
        "instruction": "Represent this code snippet for retrieval"
    },
    # Image (architecture diagram)
    {
        "image": "/path/to/architecture_diagram.png",
        "instruction": "Represent this system architecture diagram"
    },
    # Mixed: code + screenshot
    {
        "text": "The login component renders as shown:",
        "image": "/path/to/login_screenshot.png",
        "instruction": "Represent this UI component documentation"
    },
    # Video (demo recording)
    {
        "video": "/path/to/feature_demo.mp4",
        "instruction": "Represent this feature demonstration video"
    }
]

# All embedded into unified 4096-dim vector space
embeddings = embedder.process(inputs)
```

---

## Phase 5: Two-Stage Retrieval Pipeline

### Stage 1: Multimodal Recall

```python
class MultimodalRetriever:
    def __init__(
        self,
        embedder: Qwen3VLEmbedder,
        memvid_stores: Dict[str, MemVidStore]
    ):
        self.embedder = embedder
        self.stores = memvid_stores
    
    def retrieve(
        self,
        query: MultimodalQuery,
        top_k: int = 100,
        domains: List[str] = None
    ) -> List[RetrievalResult]:
        """
        Multimodal retrieval supporting any query modality.
        
        Query can be:
        - Text: "How does the authentication flow work?"
        - Image: Upload a screenshot, find similar UI
        - Mixed: "Find code that implements this" + diagram
        """
        # Embed query (supports any modality combination)
        query_input = {
            "instruction": self._get_instruction(query),
            **query.to_dict()
        }
        query_embedding = self.embedder.process([query_input])[0]
        
        # Search across requested domains
        results = []
        for domain in (domains or self.stores.keys()):
            store = self.stores[domain]
            domain_results = store.search(
                query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            results.extend(domain_results)
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
```

### Stage 2: Cross-Attention Reranking

```python
class MultimodalReranker:
    def __init__(self, reranker: Qwen3VLReranker):
        self.reranker = reranker
    
    def rerank(
        self,
        query: MultimodalQuery,
        candidates: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Precision reranking using cross-attention.
        
        Both query and documents can be multimodal.
        """
        rerank_input = {
            "instruction": "Assess relevance of the document to the query",
            "query": query.to_dict(),
            "documents": [c.chunk.to_dict() for c in candidates],
            "fps": 1.0,
            "max_frames": 64
        }
        
        scores = self.reranker.process(rerank_input)
        
        # Re-sort by rerank score
        for i, candidate in enumerate(candidates):
            candidate.rerank_score = scores[i]
        
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[:top_k]
```

### Complete Pipeline

```python
class CuttingEdgeRAGPipeline:
    def __init__(self):
        self.embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-8B")
        self.reranker = Qwen3VLReranker("Qwen/Qwen3-VL-Reranker-8B")
        self.retriever = MultimodalRetriever(self.embedder, self.memvid_stores)
        self.reranking = MultimodalReranker(self.reranker)
    
    def query(
        self,
        query: Union[str, MultimodalQuery],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        # Stage 1: Broad recall (100 candidates)
        candidates = self.retriever.retrieve(query, top_k=100)
        
        # Stage 2: Precision reranking (top 10)
        results = self.reranking.rerank(query, candidates, top_k=top_k)
        
        # Update retrieval analytics
        self._update_analytics(query, results)
        
        return results
```

---

## Phase 6: Implementation Roadmap (Updated)

### Stage 1: Foundation (Week 1-2)

- [ ] Set up Qwen3-VL-Embedding-8B environment (flash_attention_2)
- [ ] Implement metadata schema and SQLite tables
- [ ] Build multimodal content detector
- [ ] Configure MemVid with H.265

### Stage 2: Chunking Pipeline (Week 3-4)

- [ ] Implement all 6 chunking methods
- [ ] Build multimodal boundary detection
- [ ] Add screenshot-code fusion logic
- [ ] Create cross-modal link discovery

### Stage 3: Agentic Swarm (Week 5-6)

- [ ] Build headless orchestrator with modality routing
- [ ] Implement specialized agents
- [ ] Add entity extraction and graph building
- [ ] Create quality validation pipeline

### Stage 4: Retrieval System (Week 7-8)

- [ ] Implement two-stage retrieval pipeline
- [ ] Add Qwen3-VL-Reranker integration
- [ ] Build retrieval analytics
- [ ] Performance optimization (target <550ms)

---

## Appendix A: Cost Analysis (Updated)

|Component|Model|Cost|
|---|---|---|
|Embedding|Qwen3-VL-Embedding-8B|~$0.03/1M tokens*|
|Reranking|Qwen3-VL-Reranker-8B|~$0.05/1M tokens*|
|Ingestion (35MB)|One-time|~$0.10|
|Queries (10K/day, annual)|-|~$5.00|

*Estimated - not yet on OpenRouter, requires self-hosting or wait for API availability.

---

## Appendix B: Key References

1. **Qwen3-VL-Embedding-8B** (arXiv:2601.04720, January 8, 2026) - MMEB-V2 rank #1, multimodal
2. **Qwen3-VL-Reranker-8B** - Cross-attention precision reranking
3. **Qwen3-VL Foundation** - 32K context, text+image+video understanding
4. **MemVid** - Video-encoded vector storage
5. **cAST** (arXiv:2506.15655) - AST code chunking

---

_Document prepared by the penitent Sliither for Ice-ninja's ACTUAL cutting-edge RAG deployment._ _This supersedes v2.0 with proper Qwen3-VL-Embedding integration and enterprise metadata._



