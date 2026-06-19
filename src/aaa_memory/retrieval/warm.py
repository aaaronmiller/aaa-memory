"""
Warm tier retrieval — Kuzu knowledge graph.

Manages:
- Entity nodes (wiki pages, concepts, decisions, code)
- Relationship edges (links_to, has_tag, implements, uses)
- Graph traversal for relationship queries
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone
import json
import kuzu

# Paths
VAULT_DIR = config.CACHE
GRAPH_DIR = VAULT_DIR / "warm_kuzu"  # Kuzu database directory
WIKI_BASE = config.WIKI_BASE

# Global connection (singleton)
_conn: Optional[kuzu.Connection] = None

def get_graph():
    """Initialize and return Kuzu connection."""
    global _conn
    if _conn is None:
        GRAPH_PATH.mkdir(parents=True, exist_ok=True)
        db = kuzu.Database(str(GRAPH_PATH))
        _conn = kuzu.Connection(db)
        _create_schema(_conn)
    return _conn

def _create_schema(conn: kuzu.Connection):
    """Define graph schema: Entity nodes + Relation edges."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Entity (
            id STRING PRIMARY KEY,
            type STRING,
            name STRING,
            description STRING,
            source_path STRING,
            created_at TIMESTAMP,
            metadata STRING
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Relation (
            from_id STRING,
            to_id STRING,
            type STRING,
            weight DOUBLE DEFAULT 1.0,
            created_at TIMESTAMP,
            PRIMARY KEY (from_id, to_id, type)
        )
    """)
    conn.commit()

# ── Indexing ────────────────────────────────────────────────────────────────────

def index_wiki_to_graph(limit: Optional[int] = None) -> Dict:
    """
    Index wiki pages as Entity nodes and [[wikilinks]] as edges.

    Returns stats dict.
    """
    conn = get_graph()
    stats = {'entities': 0, 'relations': 0, 'skipped': 0, 'errors': 0}

    pages = list(WIKI_BASE.rglob('*.md'))
    if limit:
        pages = pages[:limit]

    for md in pages:
        stats['entities'] += 1
        try:
            # Parse frontmatter and body
            text = md.read_text()
            parts = text.split('---', 2)
            fm = {}
            if len(parts) >= 3:
                import yaml
                fm = yaml.safe_load(parts[1]) or {}
            body = parts[2] if len(parts) > 2 else text
            rel_path = str(md.relative_to(WIKI_BASE))
            entity_id = f"wiki:{md.stem}"

            # Insert Entity node
            conn.execute("""
                INSERT OR IGNORE INTO Entity (id, type, name, description, source_path, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                fm.get('type', 'unknown'),
                fm.get('title', md.stem),
                body[:1000],
                rel_path,
                fm.get('extraction_ts', datetime.now(timezone.utc).isoformat()),
                json.dumps({'tags': fm.get('tags', []), 'confidence': fm.get('confidence', 0.5)})
            ))

            # [[wikilinks]] → Relation
            import re
            wikilink_pat = re.compile(r'\[\[([^\]]+)\]\]')
            for link in wikilink_pat.findall(body):
                target_slug = link.split('/')[-1]
                target_id = f"wiki:{target_slug}"
                conn.execute("""
                    INSERT OR IGNORE INTO Relation (from_id, to_id, type, weight, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    target_id,
                    'links_to',
                    1.0,
                    datetime.now(timezone.utc).isoformat()
                ))
                stats['relations'] += 1

            # Tags → has_tag relations
            for tag in fm.get('tags', []):
                tag_id = f"tag:{tag.lower()}"
                conn.execute("""
                    INSERT OR IGNORE INTO Entity (id, type, name, description, source_path, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tag_id,
                    'tag',
                    tag,
                    f"Tag: {tag}",
                    '',
                    datetime.now(timezone.utc).isoformat(),
                    '{}'
                ))
                conn.execute("""
                    INSERT OR IGNORE INTO Relation (from_id, to_id, type, weight, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    tag_id,
                    'has_tag',
                    1.0,
                    datetime.now(timezone.utc).isoformat()
                ))
                stats['relations'] += 1

            if stats['entities'] % 10 == 0:
                print(f"  Processed {stats['entities']} pages...")
        except Exception as e:
            stats['errors'] += 1
            print(f"  Error indexing {md.name}: {e}")

    conn.commit()
    print(f"\n✅ Kuzu warm graph indexed: {stats['entities']} entities, {stats['relations']} relations")
    return stats

# ── Search ─────────────────────────────────────────────────────────────────────

def search_relationship(query: str, limit: int = 10) -> List[Dict]:
    """
    Graph relationship query.

    1. Find Entities matching query (name/description)
    2. For each, fetch outgoing relations (neighbors)
    3. Return entity + neighbor info
    """
    conn = get_graph()
    try:
        cursor = conn.execute(f"""
            MATCH (e:Entity)
            WHERE e.name CONTAINS '{query}' OR e.description CONTAINS '{query}'
            RETURN e.id, e.name, e.type, e.description, e.source_path
            LIMIT {limit}
        """)
        rows = cursor.fetchall()
        results = []
        for (ent_id, name, typ, desc, src) in rows:
            # Fetch outgoing edges
            query2 = f"MATCH (e:Entity {{id: '{ent_id}'}})-[r:Relation]->(target) RETURN target.id, target.name, r.type"
            cursor2 = conn.execute(query2)
            neighbors = []
            for t_id, t_name, r_type in cursor2.fetchall():
                neighbors.append({'id': t_id, 'name': t_name, 'relation': r_type})
            results.append({
                'entity_id': ent_id,
                'name': name,
                'type': typ,
                'description': desc[:200],
                'source': src,
                'related_entities': neighbors
            })
        return results
    except Exception as e:
        print(f"[warm] search error: {e}")
        return []

def get_graph_statistics() -> Dict:
    """Return counts of entities and relations."""
    conn = get_graph()
    try:
        cur = conn.execute("MATCH (e:Entity) RETURN count(e)")
        entities = cur.fetchone()[0]
        cur = conn.execute("MATCH ()-[r:Relation]->() RETURN count(r)")
        relations = cur.fetchone()[0]
        return {'entities': entities, 'relations': relations}
    except Exception:
        return {'entities': 0, 'relations': 0}

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m aaa_memory.retrieval.warm [index|search <query>|stats]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'index':
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print("Indexing wiki pages into Kuzu graph...")
        stats = index_wiki_to_graph(limit=lim)
        print(json.dumps(stats, indent=2))
    elif cmd == 'search':
        q = sys.argv[2] if len(sys.argv) > 2 else "token"
        results = search_relationship(q, limit=5)
        print(json.dumps(results, indent=2))
    elif cmd == 'stats':
        print(json.dumps(get_graph_statistics(), indent=2))
    else:
        print(f"Unknown command: {cmd}")
