#!/usr/bin/env python3
from pathlib import Path
from aaa_memory.audit.parser import parse_hermes_db, parse_file

hermes_db = Path('/home/misscheta/.hermes/state.db')
print("=== Testing Hermes Parser ===")
turns = list(parse_hermes_db(hermes_db))
print(f"Hermes turns: {len(turns)}")
for t in turns[:5]:
    print(f"  {t.turn_id} [{t.turn_type}]: {t.raw_text[:60]}...")

print("\n=== Testing parse_file auto-detect ===")
all_turns = list(parse_file(hermes_db))
print(f"Total turns via parse_file: {len(all_turns)}")
