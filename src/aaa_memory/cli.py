"""
CLI entry point for aaa-memory.

Commands:
  aaa-memory sessions          → list recent sessions across agents
  aaa-memory timeline <proj>   → render project timeline markdown
  aaa-memory audit --update    → run full discovery scan
"""

import sys
import argparse
from pathlib import Path
from aaa_memory.audit.discover import discover_sessions
from aaa_memory.audit.timeline import assemble_timeline
from aaa_memory.audit.classify import classify_session
from aaa_memory.audit.parser import parse_file
from aaa_memory import config


def cmd_sessions(args):
    """List all discovered sessions."""
    sessions = discover_sessions()
    total = 0
    for agent, files in sessions.items():
        print(f"\n{agent}: {len(files)} files")
        for f in files[:5]:  # show first 5
            print(f"  - {f.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        total += len(files)
    print(f"\nTotal: {total} session files discovered")


def cmd_timeline(args):
    """Build timeline for project."""
    project = args.project
    days = args.days
    assemble_timeline(project, days)
    print(
        f"Timeline for {project} (last {days}d) written to ~/knowledge/projects/{project}/timeline.md"
    )


def cmd_audit(args):
    """Full audit update — discovery + classification + timeline refresh."""
    print("=== Session audit update started ===")
    # 1. Discover — already done by parser, just collect
    sessions = discover_sessions()
    total = sum(len(v) for v in sessions.values())
    print(f"Discovered {total} session files across {len(sessions)} agents")

    # 2. Classify all sessions → write cache
    cache_path = config.CACHE_FILE
    cache = {}
    for agent, files in sessions.items():
        for f in files:
            try:
                turns = list(parse_file(f))
                if turns:
                    meta = classify_session(f.stem, turns, filepath=str(f))
                    cache[str(f)] = meta
            except Exception:
                pass

    cache_path.write_text(json.dumps(cache, indent=2))
    print(f"Classified {len(cache)} sessions → {cache_path}")

    # 3. Refresh timelines for all active projects
    projects = set(
        m["project_id"]
        for m in cache.values()
        if m.get("project_id") not in ("unknown", "")
    )
    for proj in projects:
        assemble_timeline(proj, days=7)
    print(f"Timeline refreshed for {len(projects)} projects")
    print("=== Audit update complete ===")


def main():
    parser = argparse.ArgumentParser(description="aaa-memory CLI")
    sub = parser.add_subparsers(dest="command")

    sub_sessions = sub.add_parser("sessions", help="List discovered sessions")
    sub_timeline = sub.add_parser("timeline", help="Generate project timeline")
    sub_timeline.add_argument("project", help="Project ID")
    sub_timeline.add_argument("--days", type=int, default=7, help="Lookback window")
    sub_audit = sub.add_parser("audit", help="Run full audit")
    sub_audit.add_argument("--update", action="store_true", help="Refresh all caches")

    args = parser.parse_args()

    if args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "timeline":
        cmd_timeline(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
