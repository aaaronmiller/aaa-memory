# Repository Inventory
## Complete List of PRD, Schema, and Documentation Files

**Date:** February 10, 2026
**Version:** 1.0

---

## 1. Product Requirement Documents (PRDs)

### 1.1 Main PRD Files

| File Path | Description |
|-----------|-------------|
| `chatgpt5.2-prd.md` | Original RAG architecture requirements document with detailed specifications for multimodal chunking strategies, embedding models (Qwen3-VL-Embedding-8B), and agentic deployment. Contains requirements for handling 35MB of text files in a RAG system with multiple chunking methods (semantic, fixed size, sentence, recursive hierarchical). |
| `gemini-prd.md` | Autodidactic Omni-Loop system specification with ByteRover, Graphiti, and MemVid components. Describes a full-cycle autonomous memory and self-improvement system with three-tiered memory hierarchy (Hot/Warm/Cold) and sleep-time compute for iterative refinement. |
| `opus-prd1-v3.md` | Cutting-edge RAG architecture v3.0 specification focusing on Qwen3-VL-Embedding-8B multimodal model released January 7-8, 2026. Includes comprehensive knowledge graph metadata schema with 12 dimensions and cross-modal retrieval capabilities. |
| `opus-prd2-v3.md` | Configuration specification for RAG v3.0 architecture. Contains detailed YAML configurations for embedding models, chunking strategies, metadata schema, orchestration, MemVid storage, retrieval configuration, and domain settings. |
| `opus-prd3-v3.md` | Foundational theory of multi-pass chunking and embedding model selection for 2026 RAG systems. Explores the multimodal revolution in RAG 3.0 and provides in-depth examination of embedding model stack and chunking methodologies. |

### 1.2 Unified Documentation

| File Path | Description |
|-----------|-------------|
| `docs/UNIFIED_PRD.md` | Unified Product Requirements Document that consolidates information from all source PRDs into a single comprehensive specification. Contains executive summary, core philosophy, system components, architecture overview, embedding model stack, chunking strategies, metadata schema, implementation roadmap, and cost analysis. |

---

## 2. Schema Files

### 2.1 Main Schema Reference

| File Path | Description |
|-----------|-------------|
| `docs/SCHEMA_REFERENCE.md` | Consolidated schema definitions from RAG v3.0 Architecture. Contains complete chunk metadata schema (12 dimensions), SQLite database schema with optimized tables for MemVid storage, configuration schema in YAML format, and TypeScript interfaces for all metadata components. |

### 2.2 Schema Components Detailed

#### Chunk Metadata Schema (12 Dimensions)
- **IDENTITY**: Unique identification (chunk_id, content_hash, version, parent_chunk_id, root_document_id, corpus_id)
- **PROVENANCE**: Audit trail (source_uri, git_commit_sha, author, created_at, modified_at, ingested_at)
- **CONTENT**: Content description (content_type, modalities, language, mime_type, byte_size)
- **STRUCTURE**: Chunking information (chunk_method, token_count, char_count, overlap information)
- **HIERARCHY**: Document structure (depth_level, section_path, heading_text, sibling_ids)
- **SEMANTIC**: Meaning extraction (topic_cluster_id, entities, keywords, summary, intent_class)
- **CODE_SPECIFIC**: Code metadata (ast_node_type, parent_scope, signature, imports, complexity_score, docstring)
- **MULTIMODAL**: Cross-modal relationships (referenced_images, referenced_code, cross_modal_links, visual_elements, ocr_text, diagram_type)
- **EMBEDDING**: Vector representation (model_id, dimensions, mrl_truncated, quantization, embedding_hash, embedded_at)
- **GRAPH**: Knowledge graph relationships (incoming_refs, outgoing_refs, semantic_neighbors, coreference_chain, dependency_graph)
- **QUALITY**: Quality metrics (confidence_score, validation_status, error_flags, review_status, chunking_quality)
- **RETRIEVAL**: Retrieval analytics (access_count, retrieval_success_rate, user_feedback_score, freshness_decay, last_accessed_at)

#### Database Schema
- **chunks table**: Core chunks table with essential fields for fast retrieval
- **embeddings table**: Separate table for multiple embedding versions with MemVid integration
- **chunk_relationships table**: Normalized graph relationships for query efficiency
- **semantic_neighbors table**: Precomputed similar-chunk retrieval
- **cross_modal_links table**: Multimodal retrieval support
- **entities and chunk_entities tables**: Entity-centric retrieval
- **retrieval_events table**: Analytics tracking
- **memvid_indices table**: MemVid video-encoded storage mapping

#### Configuration Schema
- **Embedding Models**: Detailed configuration for Qwen3-VL-Embedding-8B and related models
- **Chunking Strategies**: Configuration for all 7 chunking methods (fixed-size, sentence-based, semantic, recursive hierarchical, AST structural, multimodal boundary, screenshot-code fusion)
- **Metadata Schema**: Enable/disable configuration for all 12 metadata dimensions

#### TypeScript Interfaces
- Complete type definitions for all metadata components matching the database schema

---

## 3. Other Relevant Documentation

### 3.1 Technical Implementation Guides

| File Path | Description |
|-----------|-------------|
| `gemini-prd.md` (Appendix sections) | Contains technical implementation details for local multi-strategy RAG with MP4 encoding, including complete Python code examples for MP4 RAG encoder, quad encoding advantages, and agent retrieval logic. |

### 3.2 Architecture Documentation

| File Path | Description |
|-----------|-------------|
| `opus-prd1-v3.md` (later sections) | Detailed technical architecture documentation with implementation guidelines, model initialization code examples, and complete schema specifications. |
| `opus-prd3-v3.md` (various sections) | In-depth exploration of multimodal foundations, embedding model stack mechanics, and multi-pass chunking theory with implementation guidelines. |

---

## 4. Summary

This repository contains a comprehensive set of documentation for a cutting-edge RAG (Retrieval-Augmented Generation) system architecture v3.0, centered around the Qwen3-VL-Embedding-8B multimodal model. The documentation includes:

1. **Multiple PRD perspectives** covering different aspects of the system (chunking strategies, autodidactic loop, configuration)
2. **Detailed schema definitions** with both database-level and type-level specifications
3. **Unified documentation** that consolidates all information into a single coherent specification
4. **Technical implementation guides** with code examples and configuration details

The system architecture features:
- Multimodal embeddings supporting text, images, video, and mixed content
- Advanced chunking strategies (fixed-size, sentence-based, semantic, recursive hierarchical, AST structural, multimodal boundary, screenshot-code fusion)
- Three-tiered memory system (ByteRover/Hot, Graphiti/Warm, MemVid/Cold)
- Knowledge graph metadata with 12-dimensional schema
- Cross-modal retrieval capabilities
- Agentic orchestration via headless Claude Code
- MemVid video-encoded storage for efficient vector storage