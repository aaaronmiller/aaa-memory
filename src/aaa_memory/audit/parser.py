"""
Session parser — converts agent-specific formats into normalized Turn objects.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Optional
from aaa_memory.models import Turn

# ── Format loaders ────────────────────────────────────────────────────────────


def parse_claude_jsonl(path: Path) -> Iterator[Turn]:
    """Claude Code JSONL sessions — each line is a message dict."""
    with open(path, "r") as f:
        for line in f:
            try:
                msg = json.loads(line)
                turn_id = f"claude:{path.name}:{msg.get('index', '?')}"
                yield Turn(
                    turn_id=turn_id,
                    agent="claude-code",
                    session_id=path.stem,
                    turn_index=msg.get("index", 0),
                    turn_type=msg.get("type", "user"),
                    raw_text=msg.get("text", ""),
                    created_at=msg.get("timestamp", datetime.now().isoformat()),
                    metadata=json.dumps(
                        {"tool_calls": msg.get("tool_use"), "path": str(path)}
                    ),
                )
            except Exception:
                continue


def parse_openclaw_json(path: Path) -> Iterator[Turn]:
    """OpenClaw JSON session log."""
    data = json.loads(path.read_text())
    for i, turn in enumerate(data.get("turns", [])):
        yield Turn(
            turn_id=f"openclaw:{path.name}:{i}",
            agent="openclaw",
            session_id=data.get("session_id", path.stem),
            turn_index=i,
            turn_type=turn.get("role", "user"),
            raw_text=turn.get("content", ""),
            created_at=turn.get("ts", ""),
            metadata=json.dumps({"channel": data.get("channel")}),
        )


def parse_web_jsonl(path: Path) -> Iterator[Turn]:
    """Tampermonkey web captures — {'platform':..., 'messages':[...]}"""
    with open(path, "r") as f:
        for line in f:
            try:
                blob = json.loads(line)
                platform = blob.get("platform", "unknown")
                messages = blob.get("messages", [])
                for i, msg in enumerate(messages):
                    yield Turn(
                        turn_id=f"web:{platform}:{path.name}:{i}",
                        agent=f"web-{platform}",
                        session_id=path.stem,
                        turn_index=i,
                        turn_type=msg.get("role", "user"),
                        raw_text=msg.get("content", ""),
                        created_at=blob.get("timestamp", ""),
                        metadata=json.dumps({"url": blob.get("url")}),
                    )
            except Exception:
                continue


def parse_hermes_db(path: Path) -> Iterator[Turn]:
    """Hermes stores conversation history in a SQLite state.db with 'messages' table."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        # Check if messages table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cur.fetchone():
            # No messages table — nothing to parse
            conn.close()
            return
        # Query all messages ordered by id (which reflects conversation order)
        cur.execute("SELECT id, session_id, role, content, timestamp FROM messages ORDER BY id")
        for i, (msg_id, session_id, role, content, ts) in enumerate(cur.fetchall()):
            yield Turn(
                turn_id=f"hermes:{path.name}:msg_{msg_id}",
                agent="hermes",
                session_id=session_id or path.stem,
                turn_index=i,
                turn_type=role if role in ('user', 'model', 'system', 'assistant') else 'user',
                raw_text=content or "",
                created_at=datetime.fromtimestamp(ts).isoformat() if ts else datetime.now().isoformat(),
                metadata=json.dumps({"source": "hermes-state-db", "message_id": msg_id}),
            )
        conn.close()
    except Exception as e:
        print(f"[parser] Hermes parse error: {e}")
        return


def parse_file(path: Path, hint: Optional[str] = None) -> Iterator[Turn]:
    """
    Parse a session file, auto-detecting format.

    hint: 'claude', 'openclaw', 'web' to force parser
    """
    if hint:
        parser = PARSERS.get(hint)
        if parser:
            yield from parser(path)
            return

    # Auto-detect by parent directory name
    parent = path.parent.name
    if "web" in parent:
        yield from parse_web_jsonl(path)
    elif path.name.endswith(".jsonl"):
        yield from parse_claude_jsonl(path)
    elif path.suffix == ".json":
        # Could be OpenClaw — peek inside
        try:
            data = json.loads(path.read_text())
            if "turns" in data:
                yield from parse_openclaw_json(path)
            else:
                # treat as generic turn list
                for i, msg in enumerate(data):
                    yield Turn(
                        turn_id=f"unknown:{path.name}:{i}",
                        agent="unknown",
                        session_id=path.stem,
                        turn_index=i,
                        turn_type=msg.get("role", "user"),
                        raw_text=msg.get("content", ""),
                        created_at=msg.get("ts", ""),
                        metadata="{}",
                    )
        except Exception:
            pass


# Registry
PARSERS = {
    ".jsonl": parse_claude_jsonl,
    ".json": parse_openclaw_json,
    "claude": parse_claude_jsonl,
    "openclaw": parse_openclaw_json,
    "web": parse_web_jsonl,
    "hermes": parse_hermes_db,
}


def parse_file(path: Path, hint: Optional[str] = None) -> Iterator[Turn]:
    """
    Parse a session file, auto-detecting format.

    hint: 'claude', 'openclaw', 'web', 'hermes' to force parser
    """
    if hint:
        parser = PARSERS.get(hint)
        if parser:
            yield from parser(path)
            return

    # Auto-detect by extension and parent directory
    parent = path.parent.name
    if "web" in parent:
        yield from parse_web_jsonl(path)
    elif path.suffix == ".db":
        yield from parse_hermes_db(path)
    elif path.name.endswith(".jsonl"):
        yield from parse_claude_jsonl(path)
    elif path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
            if "turns" in data:
                yield from parse_openclaw_json(path)
            else:
                for i, msg in enumerate(data):
                    yield Turn(
                        turn_id=f"unknown:{path.name}:{i}",
                        agent="unknown",
                        session_id=path.stem,
                        turn_index=i,
                        turn_type=msg.get("role", "user"),
                        raw_text=msg.get("content", ""),
                        created_at=msg.get("ts", ""),
                        metadata="{}",
                    )
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    for turn in parse_file(p):
        print(turn.turn_id, turn.turn_type, turn.raw_text[:80])
