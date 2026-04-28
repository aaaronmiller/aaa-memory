-- ============================================================================
-- aaa-memory: Canonical Data Schema v2.0
-- Compatible with SQLite (primary) and PostgreSQL (optional)
-- Single encode at extraction time. Raw immutable. Derived rebuildable.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. TURNS — The atomic unit: a single exchange
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    turn_id          TEXT PRIMARY KEY,           -- UUID v7, sortable by time
    source_system    TEXT NOT NULL,              -- 'claude_code', 'openclaw', 'hermes', 'qwen', 'opencode', 'codex', 'chatgpt', 'gemini', 'claude_web'
    account_id       TEXT,                        -- For multi-account scenarios
    workspace_id     TEXT,                        -- Project root path or workspace identifier
    repo_id          TEXT,                        -- Git remote URL or local repo path
    project_id       TEXT,                        -- Logical project name (e.g., 'claude-code-proxy', 'data-kiln')
    session_id       TEXT NOT NULL,               -- Agent session ID
    turn_number      INT NOT NULL,                -- Position within session
    role             TEXT NOT NULL CHECK (role IN ('user', 'model', 'tool', 'system')),
    raw_text         TEXT NOT NULL,               -- Immutable original text
    normalized_text  TEXT,                        -- Sanitized, filler-stripped version
    summary_short    TEXT,                        -- 2-3 line summary for collapsed view
    summary_full     TEXT,                        -- Full summary for expanded view
    timestamp_utc    TEXT NOT NULL,               -- ISO 8601 with timezone
    model_name       TEXT,                        -- 'claude-sonnet-4-6', 'gpt-4o', etc.
    provider         TEXT,                        -- 'anthropic', 'openai', 'google', 'local'
    token_count_input  INT DEFAULT 0,
    token_count_output INT DEFAULT 0,
    tool_calls       TEXT,                        -- JSON array of {name, input, output}
    files_touched    TEXT,                        -- JSON array of file paths
    commands_run     TEXT,                        -- JSON array of shell commands
    git_branch       TEXT,                        -- Active branch at time of turn
    intent_category  TEXT,                        -- Classified at ingestion time
    topic_labels     TEXT,                        -- JSON array of topic tags
    failure_mode     TEXT,                        -- NULL or: 'partial_fix', 'regression', 'misunderstanding', 'context_loss'
    embedding_vector_id TEXT,                     -- Foreign key to embeddings table
    created_at       TEXT DEFAULT (datetime('now','utc')),
    schema_version   INT DEFAULT 2
);

-- ---------------------------------------------------------------------------
-- 2. ELEMENTS — Extracted knowledge from raw turns
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS elements (
    element_id       TEXT PRIMARY KEY,           -- UUID v7
    source_turn_id   TEXT REFERENCES turns(turn_id),
    element_type     TEXT NOT NULL CHECK (element_type IN ('decision', 'pattern', 'code', 'prompt', 'fact', 'concept')),
    title            TEXT NOT NULL,
    description      TEXT,
    content          TEXT NOT NULL,               -- The actual extracted content
    confidence       TEXT DEFAULT 'medium' CHECK (confidence IN ('high', 'medium', 'low')),
    importance       REAL DEFAULT 0.5,            -- 0.0-1.0 baseline importance
    tags             TEXT,                        -- JSON array of tags
    related_links    TEXT,                        -- JSON array of [[wikilink]] targets
    source_file      TEXT,                        -- Original file this was extracted from
    source_type      TEXT,                        -- 'transcript', 'prd', 'youtube', 'paper', 'web_chat'
    extracted_at     TEXT DEFAULT (datetime('now','utc')),
    embedding_vector_id TEXT REFERENCES embeddings(embedding_id),
    wiki_path        TEXT,                        -- Where this element was compiled to (e.g., 'wiki/decisions/auth-pattern.md')
    schema_version   INT DEFAULT 2
);

-- ---------------------------------------------------------------------------
-- 3. EMBEDDINGS — Pre-computed at extraction time, reused everywhere
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id     TEXT PRIMARY KEY,           -- UUID v7
    entity_type      TEXT NOT NULL CHECK (entity_type IN ('turn', 'element', 'wiki_page', 'skill')),
    entity_id        TEXT NOT NULL,               -- References the source entity
    vector           BLOB NOT NULL,               -- sqlite-vec float32 array
    model_name       TEXT NOT NULL,              -- 'qwen3-embedding-8b', 'embedding-gemma-300m', 'jina-v3'
    model_dim        INT NOT NULL,               -- 768, 1024, etc.
    embedded_at      TEXT DEFAULT (datetime('now','utc')),
    chunk_index      INT DEFAULT 0,              -- For multi-chunk entities
    chunk_total      INT DEFAULT 1,
    UNIQUE(entity_type, entity_id, chunk_index)
);

-- ---------------------------------------------------------------------------
-- 4. WIKI PAGES — Karpathy pointer-based knowledge
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wiki_pages (
    page_id          TEXT PRIMARY KEY,           -- UUID v7
    path             TEXT NOT NULL UNIQUE,        -- Relative to ~/knowledge/wiki/ (e.g., 'projects/claude-code-proxy/index.md')
    title            TEXT NOT NULL,
    sub_index       TEXT,                        -- 'projects', 'research', 'concepts', 'prompts', 'code', 'decisions'
    page_type        TEXT NOT NULL CHECK (page_type IN ('index', 'article', 'concept', 'entity', 'comparison', 'source-summary')),
    content_hash     TEXT,                        -- SHA-256 of file content for change detection
    sources          TEXT,                        -- JSON array of raw/ files this was compiled from
    related_pages    TEXT,                        -- JSON array of [[wikilink]] targets
    created_at       TEXT DEFAULT (datetime('now','utc')),
    updated_at       TEXT DEFAULT (datetime('now','utc')),
    confidence       TEXT DEFAULT 'medium' CHECK (confidence IN ('high', 'medium', 'low')),
    pinned           INT DEFAULT 0,              -- index.md pages get pinned (baseline importance 0.9)
    access_count     INT DEFAULT 0,              -- Track retrieval frequency for sleep-time compute
    last_accessed    TEXT,
    schema_version   INT DEFAULT 2
);

-- ---------------------------------------------------------------------------
-- 5. KNOWLEDGE EDGES — Relationships between concepts (Graphiti proxy)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_edges (
    edge_id          TEXT PRIMARY KEY,           -- UUID v7
    source_node      TEXT NOT NULL,              -- concept, project, skill, tool, file
    target_node      TEXT NOT NULL,
    edge_type        TEXT NOT NULL CHECK (edge_type IN ('relates_to', 'part_of', 'solved_by', 'caused', 'supersedes', 'contradicts', 'derived_from', 'used_in')),
    weight           REAL DEFAULT 1.0,
    evidence_turns   TEXT,                        -- JSON array of turn_ids that support this edge
    confidence       REAL,
    created_at       TEXT DEFAULT (datetime('now','utc')),
    UNIQUE(source_node, target_node, edge_type)
);

-- ---------------------------------------------------------------------------
-- 6. EXTRACTED SKILLS — Reusable patterns from interaction history
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extracted_skills (
    skill_id         TEXT PRIMARY KEY,           -- UUID v7
    skill_name       TEXT NOT NULL,
    description      TEXT,
    canonical_prompt TEXT,                        -- Parameterized template
    source_turns     TEXT,                        -- JSON array of turn_ids
    source_elements  TEXT,                        -- JSON array of element_ids
    usage_count      INT DEFAULT 0,
    success_rate     REAL,
    project_scope    TEXT DEFAULT 'global',       -- 'global' or specific project
    created_at       TEXT DEFAULT (datetime('now','utc')),
    last_used        TEXT,
    promoted_to_command INT DEFAULT 0             -- Whether this became a slash command
);

-- ---------------------------------------------------------------------------
-- 7. SLASH COMMAND CANDIDATES — Common operations detected across sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slash_command_candidates (
    candidate_id     TEXT PRIMARY KEY,           -- UUID v7
    command_name     TEXT NOT NULL,
    description      TEXT,
    steps            TEXT,                        -- JSON array of {type, content}
    source_turns     TEXT,                        -- JSON array of turn_ids
    frequency        INT,
    project_scope    TEXT,
    confidence       REAL
);

-- ---------------------------------------------------------------------------
-- 8. RETRIEVAL QUERY LOG — For self-improvement
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrieval_queries (
    query_id         TEXT PRIMARY KEY,           -- UUID v7
    query_text       TEXT NOT NULL,
    query_type       TEXT CHECK (query_type IN ('semantic', 'sql', 'hybrid', 'graph')),
    filters          TEXT,                        -- JSON: {project, date_range, model, ...}
    results_returned INT,
    results_clicked  TEXT,                        -- JSON array of turn_ids the user expanded
    satisfaction     TEXT CHECK (satisfaction IN ('helpful', 'not_helpful')),
    latency_ms       INT,
    created_at       TEXT DEFAULT (datetime('now','utc'))
);

-- ---------------------------------------------------------------------------
-- 9. SCHEMA EVOLUTION LOG
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_evolution (
    evolution_id     TEXT PRIMARY KEY,           -- UUID v7
    schema_version   INT NOT NULL,
    migration_sql    TEXT,
    justification    TEXT,
    triggered_by     TEXT,                        -- 'auto_detect' or manual note
    applied_at       TEXT DEFAULT (datetime('now','utc')),
    rolled_back      INT DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- 10. INGESTION STATE — Resume on crash
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_state (
    source_file      TEXT PRIMARY KEY,           -- Full path of the source file
    last_offset      INT DEFAULT 0,              -- Byte offset for resume
    last_turn_id     TEXT,                        -- Last successfully ingested turn
    status           TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'complete', 'error')),
    error_message    TEXT,
    updated_at       TEXT DEFAULT (datetime('now','utc'))
);

-- ---------------------------------------------------------------------------
-- INDEXES — Optimized for the retrieval pipeline
-- ---------------------------------------------------------------------------

-- Turn search by project (metadata filter)
CREATE INDEX IF NOT EXISTS idx_turns_project ON turns(project_id, timestamp_utc DESC);

-- Turn search by source system + session
CREATE INDEX IF NOT EXISTS idx_turns_source_session ON turns(source_system, session_id);

-- Turn search by topic (metadata filter on JSON)
CREATE INDEX IF NOT EXISTS idx_turns_topics ON turns(topic_labels) WHERE topic_labels IS NOT NULL;

-- Element search by type + confidence
CREATE INDEX IF NOT EXISTS idx_elements_type_confidence ON elements(element_type, confidence);

-- Element search by source file
CREATE INDEX IF NOT EXISTS idx_elements_source ON elements(source_file);

-- Wiki pages by sub-index (for Karpathy pointer navigation)
CREATE INDEX IF NOT EXISTS idx_wiki_sub_index ON wiki_pages(sub_index, path);

-- Knowledge edges by source node (for graph traversal)
CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_node, edge_type);

-- Knowledge edges by target node (for reverse traversal)
CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_node, edge_type);

-- Skills by usage (for sleep-time compute prioritization)
CREATE INDEX IF NOT EXISTS idx_skills_usage ON extracted_skills(usage_count DESC, last_used DESC);

-- Retrieval queries by satisfaction (for self-improvement analysis)
CREATE INDEX IF NOT EXISTS idx_retrieval_satisfaction ON retrieval_queries(satisfaction, created_at DESC);

-- ---------------------------------------------------------------------------
-- FULL-TEXT SEARCH — SQLite FTS5 for lexical retrieval
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    normalized_text,
    summary_short,
    commands_run,
    files_touched,
    topic_labels,
    content='turns',
    content_rowid='rowid'  -- SQLite auto-increment rowid
);

-- FTS5 triggers for automatic sync
CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, normalized_text, summary_short, commands_run, files_touched, topic_labels)
    VALUES (new.rowid, new.normalized_text, new.summary_short, new.commands_run, new.files_touched, new.topic_labels);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_delete AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, normalized_text, summary_short, commands_run, files_touched, topic_labels)
    VALUES('delete', old.rowid, old.normalized_text, old.summary_short, old.commands_run, old.files_touched, old.topic_labels);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_update AFTER UPDATE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, normalized_text, summary_short, commands_run, files_touched, topic_labels)
    VALUES('delete', old.rowid, old.normalized_text, old.summary_short, old.commands_run, old.files_touched, old.topic_labels);
    INSERT INTO turns_fts(rowid, normalized_text, summary_short, commands_run, files_touched, topic_labels)
    VALUES (new.rowid, new.normalized_text, new.summary_short, new.commands_run, new.files_touched, new.topic_labels);
END;

-- ---------------------------------------------------------------------------
-- VIRTUAL TABLE — sqlite-vec for vector similarity search
-- ---------------------------------------------------------------------------
-- Created dynamically by ClawMem's vec0 module at runtime:
--   CREATE VIRTUAL TABLE vec_embeddings USING vec0(
--     embedding_id TEXT PRIMARY KEY,
--     vector FLOAT[1024]  -- matches Qwen3-Embedding-8B dimension
--   );
--
-- For EmbeddingGemma-300M: FLOAT[768]
-- For Jina v3: FLOAT[1024]
--
-- The adapter creates the appropriate vec0 table based on the embedding model.
-- ---------------------------------------------------------------------------
