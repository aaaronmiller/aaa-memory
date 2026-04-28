"""
Warm tier retrieval — Graphiti (Kuzu) graph traversal.
Placeholder — Graphiti Python SDK integration.
"""

from typing import List, Dict


def search_relationship(query: str, limit: int = 10) -> List[Dict]:
    """
    Traverse knowledge graph for entity/relationship queries.
    Placeholder — real implementation uses Graphiti client.
    """
    return [
        {
            "entity": "placeholder",
            "relation": "Graphiti not yet installed",
            "snippet": "Install graphiti package and load episodes",
        }
    ]


if __name__ == "__main__":
    print(search_relationship("how does auth relate to tokens?"))
