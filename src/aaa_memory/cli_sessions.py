"""CLI commands for session audit."""
import argparse, sqlite3, json, os, sys
from pathlib import Path

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def cmd_sessions(args):
    if not VAULT.exists():
        print("No vault found")
        return
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    project_filter = ""
    params = []
    if args.project:
        project_filter = "WHERE project = ?"
        params.append(args.project)
    cur.execute(f"SELECT session_id, agent, project, MIN(timestamp) as first_seen, MAX(timestamp) as last_seen, COUNT(*) as turns FROM turns {project_filter} GROUP BY session_id ORDER BY last_seen DESC LIMIT {args.limit or 50}")
    for row in cur.fetchall():
        print(f"  {dict(row)['session_id'][:16]}  {dict(row)['agent']:12s}  {dict(row)['project'] or '?':20s}  turns={dict(row)['turns']:4d}  {dict(row)['first_seen'] or '?'}")
    conn.close()

def cmd_timeline(args):
    from aaa_memory.audit.timeline import assemble_timeline
    print(assemble_timeline(args.project, args.days or 7))

def cmd_audit(args):
    from aaa_memory.audit.discover import discover_all
    from aaa_memory.audit.parser import parse_all
    sessions = discover_all()
    parsed = parse_all(sessions)
    print(f"Discovered {len(sessions)} sessions, parsed {len(parsed)}")

def cmd_report(args):
    '''Generate comprehensive status report.'''
    from aaa_memory.reporting.transition_report import generate_report
    print(generate_report())

def cmd_search(args):
    '''Search across all tiers.'''
    from aaa_memory.retrieval.pipeline import search
    results = search(args.query, limit=args.limit or 10)
    for r in results:
        print(f"  [{r.get('tier','?')}] {r.get('raw_text','')[:200]}")

def main():
    parser = argparse.ArgumentParser(description="aaa-memory CLI")
    sub = parser.add_subparsers(dest="command")
    
    p_sessions = sub.add_parser("sessions", help="List sessions")
    p_sessions.add_argument("--project")
    p_sessions.add_argument("--limit", type=int, default=50)
    p_sessions.set_defaults(func=cmd_sessions)
    
    p_timeline = sub.add_parser("timeline", help="Show project timeline")
    p_timeline.add_argument("project")
    p_timeline.add_argument("--days", type=int, default=7)
    p_timeline.set_defaults(func=cmd_timeline)
    
    p_report = sub.add_parser("report", help="System status report")
    p_report.set_defaults(func=cmd_report)
    
    p_search = sub.add_parser("search", help="Search across all tiers")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_search)
    
    p_audit = sub.add_parser("audit", help="Run session discovery")
    p_audit.set_defaults(func=cmd_audit)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
