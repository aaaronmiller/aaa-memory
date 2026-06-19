#!/usr/bin/env python3
"""
CLI search tool for aaa-memory.

Usage:
    python3 scripts/search.py "how does auth work"
    python3 scripts/search.py --intent recent "what did we do yesterday"
    python3 scripts/search.py --project crash-guard "memory system"
    python3 scripts/search.py --limit 5 "websocket debug"
    python3 scripts/search.py --timeline crash-guard --days 14
    python3 scripts/search.py --sessions
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aaa_memory.retrieval.pipeline import search
from aaa_memory.router.intent import classify_intent, Intent
from aaa_memory.audit.timeline import assemble_timeline


def main():
    parser = argparse.ArgumentParser(description="Search aaa-memory")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--intent", choices=[e.value for e in Intent], default=None)
    parser.add_argument("--project", help="Project filter")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeline", metavar="PROJECT", help="Show timeline for project")
    parser.add_argument("--sessions", action="store_true", help="List sessions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.timeline:
        md = assemble_timeline(args.timeline, args.days)
        print(md)
        return

    if args.sessions:
        from aaa_memory.retrieval.hot import VAULT
        if not VAULT.exists():
            print("No vault found — no sessions yet.")
            return
        import sqlite3
        conn = sqlite3.connect(str(VAULT))
        cur = conn.cursor()
        try:
            cur.execute("SELECT DISTINCT session_id, agent, project, MIN(timestamp) FROM turns GROUP BY session_id ORDER BY MIN(timestamp) DESC LIMIT 20")
            for row in cur.fetchall():
                print(f"  {row[0][:12]}..  {row[1]:12s}  {row[2] or '?':20s}  {row[3] or '?'}")
        except sqlite3.OperationalError:
            print("No sessions found (empty vault).")
        conn.close()
        return

    if not args.query:
        parser.print_help()
        return

    # Classify intent if not provided
    intent = None
    if args.intent:
        intent = Intent(args.intent)

    results = search(args.query, limit=args.limit, intent=intent)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"  Results for: {args.query}")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        tier_icon = {"hot": "⚡", "warm": "🔗", "cold": "❄️"}.get(r.get("tier", ""), "📄")
        score = r.get("score", 0)
        print(f"{i:2d}. {tier_icon} [{r.get('tier', '?')}] score={score:.3f}")
        print(f"    Agent: {r.get('agent', '?')}  |  Project: {r.get('project', '?')}")
        text = r.get("raw_text", r.get("text", ""))[:200]
        print(f"    {text}")
        print()


if __name__ == "__main__":
    main()
