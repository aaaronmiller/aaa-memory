# Topic: Database Schema (SQLite)

## Summary
SQLite database schema optimized for MemVid video-encoded storage, including all tables, relationships, and indexes for the RAG v3.0 system.

---

## Tables Overview

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

| Table | Purpose |
|-------|---------|
| chunks | Core chunk storage with essential fields |
| embeddings | Multiple embedding versions per chunk, MemVid integration |
| chunk_relationships | Normalized graph relationships |
| semantic_neighbors | Precomputed similar-chunk retrieval |
| cross_modal_links | Multimodal retrieval support |
| entities | Entity definitions |
| chunk_entities | Entity-to-chunk mapping with offsets |
| retrieval_events | Analytics tracking |
| memvid_indices | MemVid video-encoded storage mapping |

---

## Table: chunks

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

**Design notes:**
- `metadata_json` stores the complete 12-dimension metadata as JSON for flexibility
- Core fields are denormalized for fast queries without JSON parsing
- `modalities` stored as JSON array string

---

## Table: embeddings

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

**Design notes:**
- Supports both direct BLOB storage and MemVid frame references
- Multiple embeddings per chunk (different models, dimensions)

---

## Table: chunk_relationships

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

---

## Table: semantic_neighbors

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

---

## Table: cross_modal_links

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

---

## Tables: entities & chunk_entities

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

---

## Table: retrieval_events

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

---

## Table: memvid_indices

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

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

## Implementation Requirements

1. Create SQLite database initialization script with all tables
2. Create migration system for schema versioning
3. Implement data access layer (DAL) with CRUD operations for each table
4. Add JSON validation for `metadata_json` and `modalities` fields
5. Implement content_hash-based deduplication logic
6. Build query helpers for common access patterns (by corpus, by document, by content_type)

---

## Conflicts / Ambiguities

- **⚠️ SQLite vs other databases:** The schema is SQLite-specific, but gemini-prd.md mentions FalkorDB (via Bolt Protocol) for Graphiti and Qdrant/FAISS for vector search. The SQLite schema appears to be for the chunk metadata store only, not for vector search or graph queries.
- **⚠️ Vector storage:** The embeddings table stores vectors as BLOB, but actual vector similarity search would use FAISS/HNSW indexes (see MemVid topic). SQLite is the metadata store, not the vector search engine.
