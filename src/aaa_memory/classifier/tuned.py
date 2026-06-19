
"""Tuned combined classifier with filename heuristic fallback."""
from pathlib import Path
from .rules import classify_file, ClassificationResult

def classify(path, content=None, llm_api_key=None, llm_fallback=True):
    """Classify with filename heuristic for short files, then rule-based + LLM."""
    if isinstance(path, str):
        path = Path(path)
    name = path.name.lower()
    text = content if content is not None else (path.read_text(errors="replace") if path.exists() else "")
    
    # Filename heuristic for short files
    if len(text.strip()) < 100:
        if "prd" in name:
            return ClassificationResult(category="prd", confidence=0.75, rule_match="filename_heuristic")
        if "transcript" in name or "session" in name:
            return ClassificationResult(category="transcript", confidence=0.75, rule_match="filename_heuristic")
        if "paper" in name or "research" in name:
            return ClassificationResult(category="research_paper", confidence=0.75, rule_match="filename_heuristic")
        if "knowledge" in name or "concept" in name or "extract" in name:
            return ClassificationResult(category="knowledge_extract", confidence=0.75, rule_match="filename_heuristic")
    
    from .combined import classify as combined_classify
    return combined_classify(path, content=content, llm_api_key=llm_api_key, llm_fallback=llm_fallback)
