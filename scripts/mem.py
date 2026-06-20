#!/usr/bin/env python3
"""
Unified memory CLI — replaces wiki-memory/memory/mem.py.

Uses aaa-memory vault as the backend. Same CLI interface, same behavior.
Drop-in replacement for the hooks.

Usage:
    mem.py save "fact" [--tags a,b] [--project p] [--source cli] [--pin]
    mem.py recall "query" [--limit N] [--project p] [--json]
    mem.py inject [--limit N] [--project p]
    mem.py list [--limit N] [--project p] [--json]
    mem.py forget <id|substring>
    mem.py capture <transcript-path> [--source cli] [--project p]
    mem.py stats
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add aaa-memory to path
AAA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AAA_ROOT / "src"))

from aaa_memory.hot.mem_store import VaultMemoryStore

# Capture markers (from wiki-memory)
_CAPTURE_MARKERS = (
    "remember that", "remember to", "remember this", "note that", "keep in mind",
    "for future reference", "don't forget", "do not forget", "important:",
    "decided to", "we decided", "the decision is", "going forward",
    "from now on", "always ", "never ", "prefers ", "i prefer", "make sure to",
)


def render_injection(memories, header="Relevant memories"):
    if not memories:
        return ""
    lines = [f"<memory source=\"aaa-memory\" hint=\"{header}\">"]
    for m in memories:
        pin = "📌 " if m.get("pinned") else ""
        tags = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
        lines.append(f"- {pin}{m['content']}{tags}")
    lines.append("</memory>")
    return "\n".join(lines)


def detect_save_directive(text):
    patterns = [
        r"(?:please\s+)?remember(?:\s+that|\s+this|\s+to)?\s*:?\s+(.+)",
        r"(?:make a |add a |save (?:a |this )?)?(?:note|memory)\s*:?\s+(.+)",
        r"don'?t forget(?:\s+to)?\s*:?\s+(.+)",
        r"keep in mind\s*:?\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def capture_from_transcript(path, source="unknown", project="default", store=None):
    store = store or VaultMemoryStore()
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    existing = {r["content"].lower() for r in store.all(project)}
    captured = []
    seen = set()
    for line in lines:
        clean = line.strip()
        low = clean.lower()
        if len(clean) < 12 or len(clean) > 500:
            continue
        if not any(m in low for m in _CAPTURE_MARKERS):
            continue
        key = low[:120]
        if key in seen or low in existing:
            continue
        seen.add(key)
        rec = store.add(clean, tags=["captured"], project=project, source=source)
        if rec:
            captured.append(rec)
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified memory engine (vault-backed)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save")
    p_save.add_argument("content")
    p_save.add_argument("--tags", default="")
    p_save.add_argument("--project", default="default")
    p_save.add_argument("--source", default="cli")
    p_save.add_argument("--pin", action="store_true")

    p_recall = sub.add_parser("recall")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=8)
    p_recall.add_argument("--project", default=None)
    p_recall.add_argument("--json", action="store_true")

    p_inject = sub.add_parser("inject")
    p_inject.add_argument("--query", default=None)
    p_inject.add_argument("--limit", type=int, default=6)
    p_inject.add_argument("--project", default=None)

    p_list = sub.add_parser("list")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--project", default=None)
    p_list.add_argument("--json", action="store_true")

    p_forget = sub.add_parser("forget")
    p_forget.add_argument("needle")

    p_capture = sub.add_parser("capture")
    p_capture.add_argument("transcript")
    p_capture.add_argument("--source", default="unknown")
    p_capture.add_argument("--project", default="default")

    sub.add_parser("stats")

    args = parser.parse_args(argv)
    store = VaultMemoryStore()

    if args.cmd == "save":
        rec = store.add(args.content, tags=args.tags, project=args.project,
                        source=args.source, pinned=args.pin)
        print(json.dumps(rec, indent=2) if rec else "{}")

    elif args.cmd == "recall":
        results = store.recall(args.query, limit=args.limit, project=args.project)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(render_injection(results) or "(no relevant memories)")

    elif args.cmd == "inject":
        if args.query:
            results = store.recall(args.query, limit=args.limit, project=args.project)
        else:
            results = store.recent(limit=args.limit, project=args.project)
        block = render_injection(results)
        if block:
            print(block)

    elif args.cmd == "list":
        results = store.recent(limit=args.limit, project=args.project)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                pin = "📌" if r.get("pinned") else "  "
                print(f"{pin} {r['id']}  [{r['project']}]  {r['content'][:80]}")

    elif args.cmd == "forget":
        n = store.forget(args.needle)
        print(f"Removed {n} mem-{'y' if n == 1 else 'ies'}.")

    elif args.cmd == "capture":
        caught = capture_from_transcript(args.transcript, source=args.source,
                                         project=args.project, store=store)
        print(f"Captured {len(caught)} memories from {args.transcript}")

    elif args.cmd == "stats":
        print(json.dumps(store.stats(), indent=2))


if __name__ == "__main__":
    main()
