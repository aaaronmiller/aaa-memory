"""Post-transition status reporting — generates markdown reports."""
import sqlite3, os
from pathlib import Path
from datetime import datetime

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def generate_report() -> str:
    lines = ["# Memory System Report", f"**Generated**: {datetime.now().isoformat()}", ""]
    if not VAULT.exists():
        lines.append("No vault found.")
        return "\n".join(lines)
    conn = sqlite3.connect(str(VAULT))
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM turns")
        lines.append(f"**Hot tier turns**: {cur.fetchone()[0]}")
    except sqlite3.OperationalError:
        lines.append("**Hot tier turns**: 0")
    try:
        cur.execute("SELECT COUNT(DISTINCT session_id) FROM turns")
        lines.append(f"**Sessions**: {cur.fetchone()[0]}")
    except sqlite3.OperationalError:
        lines.append("**Sessions**: 0")
    conn.close()
    cold_vault = Path(os.getenv("AAA_MEMORY_COLD_VAULT", str(Path.home() / ".cache/aaa-memory/cold.sqlite")))
    if cold_vault.exists():
        conn = sqlite3.connect(str(cold_vault))
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM turns_archive")
            lines.append(f"**Cold tier turns**: {cur.fetchone()[0]}")
        except sqlite3.OperationalError:
            lines.append("**Cold tier turns**: 0")
        conn.close()
    else:
        lines.append("**Cold tier turns**: 0 (no cold vault)")
    return "\n".join(lines)

if __name__ == "__main__":
    print(generate_report())
