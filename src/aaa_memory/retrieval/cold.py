"""
Cold tier retrieval — MemVid V2 .mv2 archives or compressed SQLite FTS5.
Placeholder — MemVid adapter TBD.
"""

from typing import List, Dict


def search_archive(query: str, limit: int = 10) -> List[Dict]:
    """Search long-term compressed archive."""
    return [
        {
            "archive": "placeholder",
            "note": "MemVid V2 not installed — using uncompressed fallback",
        }
    ]


if __name__ == "__main__":
    print(search_archive("websocket debug"))
