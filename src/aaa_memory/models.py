"""Shared models for aaa-memory."""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# ── Core Entities ──────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """Atomic unit of interaction — one user prompt or model response."""

    turn_id: str
    agent: str  # e.g., 'claude-code', 'openclaw', 'gemini-web'
    session_id: str
    turn_index: int
    turn_type: str  # 'user' | 'model' | 'system'
    raw_text: str
    created_at: str  # ISO timestamp
    metadata: str  # JSON blob string


@dataclass
class Element:
    """Extracted knowledge element."""

    element_id: Optional[str] = None
    type: str = ""  # decision, pattern, code, prompt, fact, concept
    title: str = ""
    content: str = ""
    confidence: float = 1.0
    tags: List[str] = None
    source_turn_id: Optional[str] = None
    source_file: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding: Optional[bytes] = None  # raw float32 bytes
    extracted_at: Optional[str] = None
    project: Optional[str] = None
    agent: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.extracted_at is None:
            self.extracted_at = datetime.now(timezone.utc).isoformat()


@dataclass
class WikiPage:
    """Compiled wiki article with [[wikilinks]]."""

    page_id: str
    title: str
    slug: str
    content: str  # Markdown body
    frontmatter: Dict[str, Any]
    backlinks: List[str] = None  # pages linking here
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.backlinks is None:
            self.backlinks = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class GraphEpisode:
    """Graphiti episode — temporal knowledge edge."""

    episode_id: str
    entity_name: str
    entity_type: str  # person, project, concept, tool
    predicate: str  # 'implemented', 'decided', 'used', 'related_to'
    object_entity: Optional[str] = None
    value: Optional[str] = None  # e.g., "SQLite" for used predicate
    timestamp: Optional[str] = None
    turn_ids: List[str] = None

    def __post_init__(self):
        if self.turn_ids is None:
            self.turn_ids = []


# ── Schema versions ────────────────────────────────────────────────────────────

SCHEMA_VERSION = "0.1.0"

# ── Utilities ──────────────────────────────────────────────────────────────────


def turn_to_dict(turn: Turn) -> dict:
    return {
        "turn_id": turn.turn_id,
        "agent": turn.agent,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "turn_type": turn.turn_type,
        "raw_text": turn.raw_text,
        "created_at": turn.created_at,
        "metadata": turn.metadata,
    }


def element_to_dict(el: Element) -> dict:
    d = asdict(el)
    if el.embedding is not None:
        d["embedding_b64"] = base64.b64encode(el.embedding).decode("ascii")
        del d["embedding"]
    return d
