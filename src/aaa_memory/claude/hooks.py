"""Claude Code lifecycle hooks for session capture.
Uses the actual vault schema: id, turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata."""
import json, os, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone

VAULT = Path(config.VAULT)

def _ensure_schema():
    """Create the turns table matching the actual vault schema."""
    VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VAULT))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE,
            agent TEXT,
            session_id TEXT,
            turn_index INTEGER DEFAULT 0,
            turn_type TEXT DEFAULT 'conversation',
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    conn.close()

def store_turn(agent: str, prompt: str, response: str, session_id: Optional[str] = None, project: Optional[str] = None) -> str:
    """Store a conversation turn into the vault."""
    _ensure_schema()
    conn = sqlite3.connect(str(VAULT))
    tid = str(uuid.uuid4())
    sid = session_id or tid
    raw = f"Human: {prompt}\n\nAssistant: {response}"
    meta = json.dumps({"project": project or os.getcwd().split("/")[-1]})
    conn.execute(
        "INSERT OR IGNORE INTO turns (turn_id, agent, session_id, raw_text, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tid, agent, sid, raw, meta, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return tid


def auto_compile(agent: str, raw_text: str, session_id: str) -> None:
    """After storing a turn, auto-extract elements and compile to wiki."""
    try:
        from aaa_memory.extractor.llm_extractor import extract_fallback
        from aaa_memory.metadata.injector import inject_metadata
        from aaa_memory.wiki.compiler import compile_to_wiki
        from aaa_memory import config

        elements = extract_fallback(raw_text)
        for elem in elements:
            inject_metadata(elem, source_file=f"session:{session_id}", classification="transcript")
        files = compile_to_wiki(elements, wiki_base=config.WIKI_BASE)
        if files:
            # Re-index wiki into vault
            import sqlite3
            conn = sqlite3.connect(str(config.VAULT))
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages USING fts5(title, content, category, path)")
            for md_file in files:
                title = Path(md_file).stem
                content = Path(md_file).read_text(errors="replace")
                try:
                    conn.execute("INSERT OR REPLACE INTO wiki_pages (title, content, category, path) VALUES (?, ?, ?, ?)",
                                (title, content, "transcript", md_file))
                except sqlite3.OperationalError:
                    continue
            conn.commit()
            conn.close()
    except Exception:
        pass  # Best-effort — don't break the turn storage
