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
from aaa_memory import config
GRAPH_PATH = config.CACHE / "graphiti_db_data"

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
        # kuzu creates its own directory
    # kuzu.Database creates its own directory
        db = kuzu.Database(str(GRAPH_PATH))
        _conn = kuzu.Connection(db)
        _create_schema(_conn)
    return _conn

def _create_schema(conn: kuzu.Connection):
    """Define graph schema: Entity nodes + Relation edges."""
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Entity (
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
        CREATE REL TABLE IF NOT EXISTS Relation (
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
    """Search entities in the graph by name/description. Returns related entities."""
    conn = get_graph()
    if conn is None:
        return stats
        return []
    try:
        cursor = conn.execute(f"""
            MATCH (e:Entity)
            WHERE e.name CONTAINS '{query}' OR e.description CONTAINS '{query}'
            RETURN e.id, e.name, e.type, e.description, e.source_path
            LIMIT {limit}
        """)
        rows = []
        while cursor.has_next():
            rows.append(cursor.get_next())
        results = []
        for row in rows:
            ent_id, name, typ, desc, src = row[0], row[1], row[2], row[3], row[4]
            query2 = f"MATCH (e:Entity {{id: '{ent_id}'}})-[r:Relation]->(target) RETURN target.id, target.name, r.type"
            cursor2 = conn.execute(query2)
            neighbors = []
            while cursor2.has_next():
                rrow = cursor2.get_next()
                t_id, t_name, r_type = rrow[0], rrow[1], rrow[2]
                neighbors.append({'id': t_id, 'name': t_name, 'relation': r_type})
            results.append({
                'entity_id': ent_id,
                'name': name,
                'type': typ,
                'description': desc,
                'source_path': src,
                'neighbors': neighbors[:5],
            })
        return results
    except Exception as e:
        print(f"[warm] search error: {e}", flush=True)
        return []