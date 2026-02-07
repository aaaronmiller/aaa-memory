

# =============================================================================
# CUTTING-EDGE RAG CONFIGURATION v3.0
# January 2026 SOTA: Qwen3-VL-Embedding Multimodal + Knowledge Graph Metadata
# =============================================================================

version: "3.0.0"
release_date: "2026-01-09"

# =============================================================================
# EMBEDDING MODELS (January 2026 SOTA)
# =============================================================================
embedding:
  # PRIMARY: Qwen3-VL-Embedding-8B (Released Jan 7-8, 2026)
  # MMEB-V2 Score: 77.8 (Rank #1)
  # Multimodal: text + images + screenshots + video + mixed
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

  # Lightweight model for boundary detection in semantic chunking
  boundary_detection:
    model_id: "Qwen/Qwen3-Embedding-0.6B"
    model_type: "text_only"
    parameters: 595800000
    dimensions:
      native: 1024
    purpose: "cheap_fast_similarity_detection"

  # RERANKER: Qwen3-VL-Reranker-8B
  # Cross-attention precision reranking for multimodal pairs
  reranker:
    model_id: "Qwen/Qwen3-VL-Reranker-8B"
    model_type: "multimodal_reranker"
    parameters: 8140000000
    layers: 36
    architecture: "single_tower_cross_attention"
    
    input_format: "query_document_pair"
    output: "relevance_score"
    
    supported_modalities:
      - text
      - image
      - video
      - mixed
    
    inference:
      torch_dtype: "bfloat16"
      attn_implementation: "flash_attention_2"

  # Fallback: Text-only model (higher MTEB but no multimodal)
  fallback:
    model_id: "Qwen/Qwen3-Embedding-8B"
    model_type: "text_only"
    parameters: 7570000000
    dimensions:
      native: 4096
    benchmarks:
      mteb: 70.58
      mteb_rank: 1

# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================
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

  # Method 4: Fixed-Size (Data/Config)
  fixed_size:
    enabled: true
    window_tokens: 512
    overlap_tokens: 50
    applies_to:
      content_types: ["configuration", "data"]
      modalities: ["text"]

  # Method 5: Multimodal Boundary (NEW - Mixed Content)
  multimodal_boundary:
    enabled: true
    visual_context_window: 1  # paragraphs before/after
    caption_detection: true
    figure_reference_detection: true
    preserve_figure_caption_pairs: true
    applies_to:
      content_types: ["documentation", "research_paper"]
      modalities: ["mixed_text_image", "mixed_all"]

  # Method 6: Screenshot-Code Fusion (NEW)
  screenshot_code_fusion:
    enabled: true
    matching_strategies:
      - filename_similarity
      - ocr_text_matching
      - reference_comment_detection
    applies_to:
      content_types: ["code"]
      modalities: ["mixed_text_image"]

# =============================================================================
# METADATA SCHEMA CONFIGURATION
# =============================================================================
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

# =============================================================================
# ORCHESTRATION
# =============================================================================
orchestration:
  headless:
    enabled: true
    logic_file: "orchestration_logic_v3.md"
    checkpoint_interval_minutes: 5
  
  concurrency:
    max_files: 50
    modality_detector_workers: 2
    content_router_workers: 2
    code_specialist_workers: 8
    text_specialist_workers: 8
    multimodal_specialist_workers: 4
    graph_builder_workers: 2
    integration_workers: 2
  
  batching:
    embedding_batch_size: 16  # Smaller for multimodal
    integration_buffer_size: 50
    integration_flush_timeout_ms: 5000
  
  mcp_servers:
    - name: "filesystem-mcp"
      description: "Sandboxed file system access"
    - name: "git-mcp"
      description: "Git history for provenance"
    - name: "embedding-mcp"
      description: "Unified multimodal embedding API"
      config:
        default_model: "Qwen/Qwen3-VL-Embedding-8B"
    - name: "memvid-mcp"
      description: "Video-encoded vector storage"
    - name: "entity-mcp"
      description: "Named entity extraction"

# =============================================================================
# MEMVID STORAGE
# =============================================================================
memvid:
  encoder:
    codec: "hevc"
    crf: 18
    gop: 30
    preset: "medium"
  
  vector_config:
    input_dimensions: 4096
    storage_dimensions: 1024
    similarity_sort: true
  
  features:
    parallel_segments: true
    smart_recall: true
    text_search: true
    hnsw_index: true
  
  files:
    codebase: "codebase.mp4"
    research: "research.mp4"
    prompts: "prompts.mp4"

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
retrieval:
  # Stage 1: Broad recall
  recall:
    model: "primary"  # Qwen3-VL-Embedding-8B
    top_k: 100
    similarity_threshold: 0.5
    multimodal_query_support: true
  
  # Hybrid search
  hybrid:
    enabled: true
    vector_weight: 0.7
    keyword_weight: 0.3
    keyword_method: "bm25"
  
  # Stage 2: Precision reranking
  reranking:
    enabled: true
    model: "reranker"  # Qwen3-VL-Reranker-8B
    top_k_input: 100
    top_k_output: 10
    multimodal_rerank: true
  
  # Cross-modal retrieval
  cross_modal:
    enabled: true
    query_modalities: ["text", "image", "mixed"]
    result_modalities: ["text", "image", "code", "mixed"]
  
  targets:
    max_latency_ms: 550
    min_relevance_score: 0.6

# =============================================================================
# DOMAIN CONFIGURATION
# =============================================================================
domains:
  - name: "prompts"
    description: "User inputs and prompts to LLMs"
    storage: "prompts.mp4"
    chunking_methods:
      - "semantic"
      - "fixed_size"
    retention: "30_days_rolling"
    multimodal: false
    
  - name: "codebase"
    description: "Multi-repository source code and configs"
    storage: "codebase.mp4"
    chunking_methods:
      - "ast_structural"
      - "fixed_size"
      - "screenshot_code_fusion"
    retention: "version_controlled"
    multimodal: true
    cross_reference: true
    
  - name: "research"
    description: "Research papers, documentation, diagrams"
    storage: "research.mp4"
    chunking_methods:
      - "recursive_hierarchical"
      - "semantic"
      - "multimodal_boundary"
    retention: "permanent"
    multimodal: true

# =============================================================================
# QUALITY ASSURANCE
# =============================================================================
quality:
  validation:
    validate_all_chunks: true
    min_coherence_score: 0.6
    min_completeness_score: 0.5
    flag_outlier_embeddings: true
  
  verification_queries:
    - query: "database connection setup"
      query_type: "text"
      expected_domain: "codebase"
      max_latency_ms: 550
      
    - query: "system architecture diagram"
      query_type: "text"
      expected_modalities: ["image", "mixed"]
      max_latency_ms: 600
      
    - query:
        text: "Find code that implements this UI"
        image: "test_screenshot.png"
      query_type: "mixed"
      expected_domain: "codebase"
      max_latency_ms: 700

# =============================================================================
# ERROR HANDLING
# =============================================================================
error_handling:
  api_rate_limit:
    initial_backoff_seconds: 5
    max_backoff_seconds: 60
    max_retries: 5
  
  embedding_failure:
    retry_count: 3
    fallback_to_text_only: true
  
  parse_failure:
    log_file: "ingestion_errors.log"
    continue_on_error: true
    quarantine_failed: true
  
  multimodal_failure:
    fallback_to_text_only: true
    log_visual_errors: true

