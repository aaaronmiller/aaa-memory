#!/usr/bin/env python3
"""
Test session classification on Hermes data.
"""

from pathlib import Path
from collections import defaultdict
from aaa_memory.audit.parser import parse_hermes_db
from aaa_memory.audit.classify import classify_session

hermes_db = Path('/home/misscheta/.hermes/state.db')
print("=== Loading Hermes turns ===")
turns = list(parse_hermes_db(hermes_db))
print(f"Loaded {len(turns)} turns")

# Group by session_id
sessions = defaultdict(list)
for t in turns:
    sessions[t.session_id].append(t)

print(f"Found {len(sessions)} unique sessions")

# Classify first 5 sessions
print("\n=== Session Classifications ===")
for i, (session_id, sturns) in enumerate(list(sessions.items())[:5]):
    result = classify_session(session_id, sturns)
    print(f"\nSession {i+1}: {session_id}")
    print(f"  Project: {result['project_id']}")
    print(f"  Type: {result['session_type']}")
    print(f"  Turns: {result['turn_count']}")
    print(f"  Decisions: {len(result['key_decisions'])}")
    if result['key_decisions']:
        for d in result['key_decisions'][:2]:
            print(f"    • {d[:80]}")
