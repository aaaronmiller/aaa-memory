# Topic 04: Database Schema (SQLite)

## Summary
Implement the SQLite database schema optimized for MemVid video-encoded storage with full metadata support, including tables for chunks, embeddings, relationships, entities, retrieval analytics, and MemVid index mapping.

---

## Overview
> Sources: `docs/SCHEMA_REFERENCE.md` (lines 61–252), `docs/UNIFIED_PRD.md` (Implementation Roadmap Stage 1)

The database uses SQLite as the primary metadata store. Vector storage is handled by MemVid (MP4 files) and/or FAISS, with the SQLite database maintaining the index mapping and all non-vector metadata.

---

## Tables

### 1. `chunks` — Core chunks table
> Source: `docs/SCHEMA_REFERENCE.md` (lines 67–97)

```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    corpus_id TEXT NOT NULL,
    root_document_id TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    modalities TEXT NOT NULL,           -- JSON array
    primary_modality TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    chunking_method TEXT NOT NULL,
    parent_chunk_id TEXT,
    depth_level INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,        -- Complete ChunkMetadata object
    FOREIGN KEY (parent_chunk_id) REFERENCES chunks(chunk_id)
);
```

### 2. `embeddings` — Multiple embedding versions per chunk
> Source: `docs/SCHEMA_REFERENCE.md` (lines 102–122)

```sql
CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    mrl_truncated INTEGER DEFAULT 0,
    quantization TEXT,
    vector BLOB,                        -- Or reference to MemVid frame
    memvid_frame_index INTEGER,         -- If stored in MemVid
    memvid_file TEXT,                   -- Which .mp4 file
    instruction_used TEXT,
    embedded_at TEXT NOT NULL,
    embedding_hash TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
```

### 3. `chunk_relationships` — Graph relationships
> Source: `docs/SCHEMA_REFERENCE.md` (lines 127–139)

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

### 4. `semantic_neighbors` — Precomputed similarity
> Source: `docs/SCHEMA_REFERENCE.md` (lines 144–155)

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

### 5. `cross_modal_links` — Multimodal retrieval
> Source: `docs/SCHEMA_REFERENCE.md` (lines 160–173)

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

### 6. `entities` and `chunk_entities` — Entity-centric retrieval
> Source: `docs/SCHEMA_REFERENCE.md` (lines 178–198)

```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT,
    knowledge_base_id TEXT
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

### 7. `retrieval_events` — Analytics tracking
> Source: `docs/SCHEMA_REFERENCE.md` (lines 203–217)

```sql
CREATE TABLE retrieval_events (
    event_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    query_text TEXT,
    query_embedding_hash TEXT,
    retrieval_rank INTEGER,
    rerank_score REAL,
    was_selected INTEGER,
    user_feedback INTEGER,              -- -1, 0, 1
    timestamp TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
```

### 8. `memvid_indices` — MemVid video-encoded storage mapping
> Source: `docs/SCHEMA_REFERENCE.md` (lines 221–232)

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

---

## Indexes
> Source: `docs/SCHEMA_REFERENCE.md` (lines 237–252)

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

## Implementation Tasks

1. Create `src/db/schema.sql` — Complete DDL script with all tables and indexes
2. Create `src/db/connection.py` — SQLite connection manager with WAL mode
3. Create `src/db/repositories/chunks.py` — CRUD operations for chunks table
4. Create `src/db/repositories/embeddings.py` — CRUD for embeddings table
5. Create `src/db/repositories/relationships.py` — CRUD for chunk_relationships and semantic_neighbors
6. Create `src/db/repositories/entities.py` — CRUD for entities and chunk_entities
7. Create `src/db/repositories/retrieval.py` — CRUD for retrieval_events
8. Create `src/db/repositories/memvid.py` — CRUD for memvid_indices
9. Create migration system for schema versioning

---

## Conflicts & Ambiguities

1. **SQLite vs FalkorDB:** `gemini-prd.md` (line 269) specifies FalkorDB (via Bolt Protocol) for the Graphiti warm memory layer, while the schema reference uses SQLite for everything. These serve different purposes: SQLite is the chunk metadata store; FalkorDB is the knowledge graph for warm memory. Both are needed.
2. **Vector storage:** The `embeddings` table has both a `vector BLOB` column and `memvid_frame_index`/`memvid_file` columns. A chunk's embedding can be stored either inline as a BLOB or referenced via MemVid frame. The system should support both modes.
3. **FAISS vs SQLite for vector search:** The actual vector similarity search happens in FAISS/HNSW, not SQLite. SQLite stores metadata and the mapping. This is implied but not explicitly stated in one place.
