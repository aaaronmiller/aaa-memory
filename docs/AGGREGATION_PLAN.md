# RAG v3.0 Aggregation Plan

## 1. Complete Inventory of Source Files

### 1.1 `chatgpt5.2-prd.md`
This file contains requirements for a RAG system focusing on ingestion of approximately 35MB of text files. Key elements include:
- Use of latest embedding models (Qwen3 embedding 8B or 2B, or upcoming Gemini text embedding 001)
- Implementation of multiple chunking methods: semantic, fixed size, sentence, and recursive hierarchical chunking
- Asynchronous chunking approach
- Consideration of multiple models for different tasks (Sonnet, Haiku, Gemini Pro/Flash, Mimo V2)
- Integration with MemVid for video-based storage
- Separate handling for writing, coding, and research paper data
- Cost analysis for embedding and retrieval operations

### 1.2 `gemini-prd.md`
This document describes the "Autodidactic Omni-Loop" system using ByteRover, Graphiti, and MemVid components. Key aspects include:
- Local-first, zero-cost architectural framework for transforming Claude Code into a self-improving AGI
- Three-tiered memory hierarchy (Hot/Warm/Cold) with ByteRover (Hot), Graphiti (Warm), and MemVid (Cold)
- "Sleep-Time Compute" for iterative refinement without human intervention
- Bidirectional learning system
- Tribunal critic component with adversarial personas
- Mutator evolution component using genetic algorithms
- Taste Oracle quality gate with vector-based novelty detection
- Five strategic integrations: Hypergraph Knowledge Representation, Active Inference Curiosity Module, Formal Verification Gate, Automated Model Merging, and Contrastive Value Alignment

### 1.3 `opus-prd1-v3.md`
This file presents a cutting-edge RAG architecture v3.0 focusing on multimodal agentic swarms with knowledge graph metadata. Major components include:
- Use of Qwen3-VL-Embedding-8B multimodal model (released January 7-8, 2026)
- Sophisticated knowledge graph metadata schema with 12 dimensions
- Cross-modal retrieval capabilities
- MemVid multimodal video-encoded storage
- Detailed technical specifications for embedding models and rerankers
- Comprehensive TypeScript interfaces for metadata schema

### 1.4 `opus-prd2-v3.md`
This is a configuration specification for the RAG v3.0 architecture. Key elements include:
- Detailed configuration for embedding models (Qwen3-VL-Embedding-8B, boundary detection model, Qwen3-VL-Reranker-8B, fallback text-only model)
- Chunking strategies configuration (semantic, recursive hierarchical, AST structural, fixed size, multimodal boundary, screenshot-code fusion)
- Metadata schema configuration with enable/disable options for each dimension
- Orchestration settings including concurrency and batching parameters
- MemVid storage configuration
- Retrieval configuration with recall and reranking stages
- Domain-specific configurations (prompts, codebase, research)
- Quality assurance settings

### 1.5 `opus-prd3-v3.md`
This document covers the foundational theory of multi-pass chunking and embedding model selection. Main topics include:
- The multimodal turn and why RAG 3.0 exists
- In-depth examination of the embedding model stack
- Foundational theory of multi-pass chunking with four-layer epistemic scaffolding system
- Taxonomy of chunk types (fixed-length, sentence-based, semantic, recursive hierarchical)
- History of multi-pass chunking from 2020-2026
- Mechanics of the four chunking modes
- Implementation guidelines for 2026

### 1.6 `docs/SCHEMA_REFERENCE.md`
This file provides consolidated schema definitions from the RAG v3.0 architecture. It includes:
- Chunk metadata schema with 12 dimensions
- SQLite database schema optimized for MemVid video-encoded storage
- Configuration schema for embedding models, chunking strategies, and metadata
- TypeScript interfaces for complete chunk metadata

## 2. Proposed Structure/Outline for Unified PRD

The unified PRD should follow this structure:

1. **Executive Summary**
   - Project overview and objectives
   - Key innovations and differentiators

2. **Foundational Concepts**
   - The multimodal turn and why RAG 3.0 exists
   - Evolution from previous RAG generations
   - Core architectural principles

3. **System Architecture**
   - High-level architecture diagram
   - Component breakdown (ByteRover, Graphiti, MemVid, Tribunal, Mutator, Taste Oracle)
   - Memory hierarchy (Hot/Warm/Cold)
   - Data lifecycle and transition layers

4. **Embedding Model Stack**
   - Primary model (Qwen3-VL-Embedding-8B) specifications
   - Reranker model (Qwen3-VL-Reranker-8B) specifications
   - Fallback and auxiliary models
   - Model initialization and configuration

5. **Chunking Strategies**
   - Fixed-length chunking
   - Sentence-based chunking
   - Semantic chunking (agentic)
   - Recursive hierarchical chunking (agentic)
   - AST structural chunking (for code)
   - Multimodal boundary chunking
   - Screenshot-code fusion chunking

6. **Metadata Schema**
   - 12-dimensional metadata schema overview
   - Detailed breakdown of each dimension
   - TypeScript interfaces

7. **Database Schema**
   - Core chunks table
   - Embeddings table
   - Graph relationships
   - Semantic neighbors
   - Cross-modal links
   - Entities and retrieval analytics

8. **Storage and Retrieval**
   - MemVid video-encoded storage
   - Retrieval pipeline (recall, hybrid search, reranking, cross-modal)
   - Performance targets and optimization strategies

9. **Orchestration and Concurrency**
   - Headless operation
   - Concurrency settings
   - Batching strategies
   - MCP server configurations

10. **Quality Assurance**
    - Validation mechanisms
    - Verification queries
    - Error handling procedures

11. **Implementation Roadmap**
    - Phased approach
    - Dependencies and prerequisites
    - Testing and validation procedures

12. **Appendices**
    - Configuration reference
    - API specifications
    - Troubleshooting guide

## 3. Comprehensive Checklist of Major Topics, Features, and Requirements

### 3.1 Core Architecture Elements
- [ ] Multimodal RAG architecture
- [ ] Three-tiered memory hierarchy (Hot/Warm/Cold)
- [ ] Sleep-time compute for autonomous improvement
- [ ] Bidirectional learning system
- [ ] Agentic orchestration
- [ ] Headless operation capability

### 3.2 Memory Components
- [ ] ByteRover (Hot Memory) filesystem-based active context
- [ ] Graphiti (Warm Memory) temporal knowledge graph
- [ ] MemVid (Cold Memory) deep archive with H.265 compressed video storage
- [ ] Memory transition protocols (Digest: Hot→Warm, Freeze: Warm→Cold)

### 3.3 Embedding Models
- [ ] Qwen3-VL-Embedding-8B as primary model
- [ ] Qwen3-VL-Reranker-8B for precision reranking
- [ ] Qwen3-VL-Embedding-2B as lightweight alternative
- [ ] Qwen3-Embedding-0.6B for boundary detection
- [ ] Qwen3-Embedding-8B as text-only fallback
- [ ] Gemini Text-Embedding-001 consideration
- [ ] Model benchmark specifications (MMEB-V2, MMTEB scores)

### 3.4 Chunking Strategies
- [ ] Fixed-size chunking (1.5-3k tokens with 200-400 token overlap)
- [ ] Sentence-based chunking (3-7 sentence bundles, 300-700 tokens)
- [ ] Semantic chunking (800-2000 tokens, agentic)
- [ ] Recursive hierarchical chunking (agentic, deep)
- [ ] AST structural chunking for code
- [ ] Multimodal boundary chunking
- [ ] Screenshot-code fusion chunking

### 3.5 Metadata Schema (12 Dimensions)
- [ ] Identity (chunk_id, content_hash, version, parent_chunk_id, root_document_id, corpus_id)
- [ ] Provenance (source_uri, git_commit_sha, author, created_at, modified_at, ingested_at)
- [ ] Content (content_type, modalities, language, mime_type, byte_size)
- [ ] Structure (chunk_method, token_count, char_count, overlap, boundaries)
- [ ] Hierarchy (depth_level, section_path, heading_text, parent_heading, document_position, siblings, children)
- [ ] Semantic (topic_cluster_id, entities, keywords, summary, intent_classification)
- [ ] Code-specific (ast_node_type, parent_scope, signature, imports, complexity_score, docstring)
- [ ] Multimodal (visual_elements, referenced_images, referenced_code, cross_modal_links, ocr_text, diagram_type)
- [ ] Embedding (model_id, dimensions, mrl_truncated, quantization, embedding_hash, embedded_at)
- [ ] Graph (incoming_refs, outgoing_refs, semantic_neighbors, coreference_chain, dependency_graph)
- [ ] Quality (confidence_score, validation_status, error_flags, review_status, chunking_quality)
- [ ] Retrieval (access_count, retrieval_success_rate, user_feedback_score, freshness_decay, last_accessed_at)

### 3.6 Database Schema Elements
- [ ] Core chunks table
- [ ] Embeddings table
- [ ] Chunk relationships table
- [ ] Semantic neighbors table
- [ ] Cross-modal links table
- [ ] Entities and chunk_entities tables
- [ ] Retrieval events table
- [ ] MemVid indices table
- [ ] Indexes for common queries

### 3.7 Storage and Retrieval
- [ ] MemVid encoder configuration (HEVC codec, CRF 18)
- [ ] Vector configuration (input 4096 dimensions, storage 1024 dimensions)
- [ ] Recall stage configuration (top_k: 100, similarity_threshold: 0.5)
- [ ] Hybrid search (vector_weight: 0.7, keyword_weight: 0.3)
- [ ] Reranking stage (top_k_input: 100, top_k_output: 10)
- [ ] Cross-modal retrieval support
- [ ] Performance targets (max_latency_ms: 550)

### 3.8 Orchestration and Concurrency
- [ ] Headless operation with checkpoint intervals
- [ ] Concurrency settings for different worker types
- [ ] Batching configurations
- [ ] MCP server configurations (filesystem, git, embedding, memvid, entity)

### 3.9 Quality Assurance
- [ ] Chunk validation mechanisms
- [ ] Coherence and completeness scoring
- [ ] Outlier embedding detection
- [ ] Verification queries for different domains
- [ ] Error handling procedures (rate limiting, embedding failures, parse failures)

### 3.10 Strategic Integrations
- [ ] Hypergraph Knowledge Representation
- [ ] Active Inference Curiosity Module
- [ ] Formal Verification Gate
- [ ] Automated Model Merging
- [ ] Contrastive Value Alignment

### 3.11 Sleep-Time Refinement Loops
- [ ] Simulator (Correction) loop
- [ ] Professor (Synthesis) loop
- [ ] Evolutionary Forge (Creation) loop

### 3.12 Implementation Requirements
- [ ] Asynchronous chunking capabilities
- [ ] Multi-model orchestration
- [ ] Cost analysis and optimization
- [ ] Separate handling for different data types (writing, coding, research)
- [ ] Video-based storage integration (MemVid)
- [ ] Cross-modal retrieval capabilities

## 4. Overlap Analysis

Several significant overlaps exist between the source files:

### 4.1 Embedding Models
All files reference Qwen3-VL-Embedding models, though with varying levels of detail:
- `opus-prd1-v3.md` and `opus-prd2-v3.md` provide the most detailed technical specifications
- `chatgpt5.2-prd.md` mentions Qwen3 embedding models but with less technical detail
- `opus-prd3-v3.md` discusses the theoretical foundations

### 4.2 Chunking Strategies
Multiple files discuss various chunking approaches:
- `chatgpt5.2-prd.md` lists semantic, fixed size, sentence, and recursive hierarchical chunking as requirements
- `opus-prd1-v3.md` and `opus-prd2-v3.md` provide detailed configurations for these and additional methods
- `opus-prd3-v3.md` offers theoretical foundations and historical context

### 4.3 Metadata Schema
The 12-dimensional metadata schema appears in both `opus-prd1-v3.md` and `docs/SCHEMA_REFERENCE.md`:
- `opus-prd1-v3.md` provides the conceptual framework
- `docs/SCHEMA_REFERENCE.md` offers concrete implementations

### 4.4 MemVid Storage
MemVid is mentioned across multiple files:
- `chatgpt5.2-prd.md` suggests using it for storage compression
- `gemini-prd.md` describes it as Cold Memory with QR-encoded video storage
- `opus-prd1-v3.md` and `opus-prd2-v3.md` provide technical specifications

### 4.5 Agentic Systems
The concept of agentic processing appears in several files:
- `chatgpt5.2-prd.md` discusses using different models for different chunking tasks
- `gemini-prd.md` describes the Autodidactic Omni-Loop with agentic components
- `opus-prd3-v3.md` emphasizes agentic chunking for semantic and recursive hierarchical methods

## 5. Conflict and Divergence Analysis

### 5.1 Model Selection
While most files converge on Qwen3-VL-Embedding-8B as the primary model, there are some differences:
- `chatgpt5.2-prd.md` also considers Gemini Text-Embedding-001 as an alternative
- Different files provide varying technical specifications and benchmarks

### 5.2 Chunking Approach
There are differing perspectives on chunking:
- `chatgpt5.2-prd.md` treats chunking as a requirement to be implemented
- `opus-prd3-v3.md` presents it as a foundational theory with historical context
- `opus-prd2-v3.md` provides concrete configuration parameters

### 5.3 System Architecture
Different architectural visions emerge:
- `gemini-prd.md` focuses on the Autodidactic Omni-Loop with specific memory components
- `opus-prd1-v3.md` emphasizes multimodal agentic swarms
- `chatgpt5.2-prd.md` is more focused on practical implementation concerns like cost

### 5.4 Implementation Priority
Files differ in their emphasis:
- `chatgpt5.2-prd.md` prioritizes immediate implementation concerns like cost and asynchronous processing
- `gemini-prd.md` focuses on long-term autonomous improvement capabilities
- `opus-prd1-v3.md` and `opus-prd3-v3.md` emphasize cutting-edge technical capabilities

## 6. Handling Overlaps and Conflicts

### 6.1 Overlap Handling Strategies

1. **Complementary Detail Levels**: 
   - Use high-level conceptual descriptions from `opus-prd1-v3.md` and `gemini-prd.md`
   - Supplement with technical specifications from `opus-prd2-v3.md` and `docs/SCHEMA_REFERENCE.md`
   - Include implementation considerations from `chatgpt5.2-prd.md`

2. **Progressive Enhancement**:
   - Present core concepts first
   - Add technical specifications as refinements
   - Include implementation considerations as practical notes

3. **Cross-Referencing**:
   - Maintain connections between related concepts across different files
   - Highlight how theoretical concepts translate to practical implementations

### 6.2 Conflict Resolution Approaches

1. **Technical Depth Hierarchy**:
   - Prioritize technically detailed and recent specifications (favor `opus-prd` series)
   - Use implementation-focused concerns (`chatgpt5.2-prd.md`) as practical considerations rather than primary specifications
   - Treat architectural visions (`gemini-prd.md`) as complementary enhancements

2. **Consensus Building**:
   - Where multiple files agree (e.g., Qwen3-VL-Embedding-8B), present as established consensus
   - Where they differ, present alternatives with clear attribution to source files
   - Provide decision frameworks for choosing between alternatives

3. **Contextual Application**:
   - Present different approaches as suitable for different contexts
   - Emphasize that architectural choices depend on specific implementation goals
   - Maintain flexibility in the unified document while providing clear guidance

## 7. Implementation Recommendations

To create the unified PRD effectively:

1. Start with the structural outline provided above
2. Integrate content from each source file according to the overlap handling strategies
3. Clearly mark where conflicts exist and provide resolution approaches
4. Maintain the technical depth progression from conceptual to implementation
5. Preserve the innovative elements from each source while ensuring coherence
6. Include all checklist items to ensure comprehensive coverage
7. Provide clear navigation and cross-referencing between related concepts