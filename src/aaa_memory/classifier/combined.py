"""Combined classifier interface."""

from .rules import ClassificationResult
from typing import Optional


def classify(
    path,
    content: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_fallback: bool = True,
) -> ClassificationResult:
    from .rules import classify_file as rule_classify

    result = rule_classify(path, content, llm_fallback=None)
    if result.confidence >= 0.7:
        return result
    if llm_fallback and result.category not in ("unknown",):
        try:
            from .llm_classifier import classify as llm_classify_func

            text = (
                content
                if content is not None
                else path.read_text(errors="replace")[:8000]
            )
            llm_cat = llm_classify_func(text, api_key=llm_api_key)
            from . import ClassificationResult as CR

            return CR(category=llm_cat, confidence=0.75, llm_used=True)
        except Exception:
            pass
    return result
