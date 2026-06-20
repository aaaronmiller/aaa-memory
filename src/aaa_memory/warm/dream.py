#!/usr/bin/env python3
"""
Dream Agent — ported from wiki-memory to aaa-memory.

Sleep-time compute that compiles raw knowledge into wiki pages.
Reads from the aaa-memory vault (turns + hot_memories) and raw/ directory.
Writes wiki pages to the vault's wiki_pages table and ~/ai-wiki/pages/.

Phase 0: Budget allocation
Phase 1: Extract — read new content from vault + raw/
Phase 2: Refine — confidence scoring
Phase 3: Compile — write wiki pages
Phase 4: Pattern Detect — repeated tasks → SKILL.md
Phase 5: Re-index
Phase 6: Improve — vault quality
"""

import os
import sys
import json
import time
import re
import subprocess
import sqlite3
from datetime import datetime, date, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

# ─── Configuration ───────────────────────────────────────────────
AAA_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AAA_ROOT / "src"))

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
AI_WIKI = Path.home() / "ai-wiki"
RAW_DIR = AI_WIKI / "raw"
WIKI_DIR = AI_WIKI / "pages"
CONCEPTS_DIR = WIKI_DIR / "concepts"
ENTITIES_DIR = WIKI_DIR / "entities"
SOURCES_DIR = WIKI_DIR / "sources"
QUERIES_DIR = WIKI_DIR / "queries"
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / "log.md"
META_DIR = AI_WIKI / ".meta"
SKILL_PATTERNS_FILE = META_DIR / "skill_patterns.json"
INTAKE_LOG_FILE = META_DIR / "intake_log.jsonl"
PI_SKILLS_DIR = Path(os.environ.get("PI_SKILLS_DIR", Path.home() / ".pi" / "agent" / "skills"))
WIKI_SKILLS_DIR = META_DIR / "skills"

ALL_DIRS = [RAW_DIR, CONCEPTS_DIR, ENTITIES_DIR, SOURCES_DIR, QUERIES_DIR,
            META_DIR, PI_SKILLS_DIR, WIKI_SKILLS_DIR]

SKILL_CREATION_THRESHOLD = int(os.environ.get("SKILL_CREATION_THRESHOLD", "3"))
CONFIDENCE_AUTO = float(os.environ.get("CONFIDENCE_AUTO", "0.3"))
CONFIDENCE_FLAG = float(os.environ.get("CONFIDENCE_FLAG", "0.5"))


@dataclass
class Budget:
    total_seconds: float
    intake: float = 0.0
    compile: float = 0.0
    improve: float = 0.0
    lint: float = 0.0

@dataclass
class Claim:
    text: str
    source: str
    category: str = "note"
    confidence: float = 0.0
    entities: list = field(default_factory=list)
    concepts: list = field(default_factory=list)

@dataclass
class DreamReport:
    timestamp: str = ""
    duration: float = 0.0
    sources_scanned: int = 0
    sources_processed: int = 0
    claims_extracted: int = 0
    claims_accepted: int = 0
    claims_rejected: int = 0
    pages_created: int = 0
    pages_updated: int = 0
    patterns_detected: int = 0
    skills_created: int = 0
    improvements_made: int = 0
    errors: list = field(default_factory=list)


# ─── Vault access ────────────────────────────────────────────────

def _vault_conn():
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    return conn


def extract_from_vault(report: DreamReport) -> list[Claim]:
    """Extract recent content from the aaa-memory vault."""
    claims = []
    conn = _vault_conn()

    # Recent turns
    rows = conn.execute("""
        SELECT * FROM turns
        WHERE created_at > datetime('now', '-7 days')
        ORDER BY created_at DESC LIMIT 50
    """).fetchall()
    for r in rows:
        text = r["raw_text"]
        if len(text) < 30:
            continue
        claims.append(Claim(
            text=text,
            source=f"turn:{r['turn_id'][:30]}",
            category="conversation",
        ))

    # Hot memories
    rows = conn.execute("SELECT * FROM hot_memories ORDER BY created DESC LIMIT 20").fetchall()
    for r in rows:
        claims.append(Claim(
            text=r["content"],
            source=f"memory:{r['id']}",
            category="memory",
        ))

    conn.close()
    report.sources_scanned = len(claims)
    report.sources_processed = len(claims)
    return claims


def extract_from_raw(report: DreamReport) -> list[Claim]:
    """Extract from raw/ directory (legacy wiki-memory intake)."""
    claims = []
    if not RAW_DIR.exists():
        return claims

    processed = set()
    if INTAKE_LOG_FILE.exists():
        for line in INTAKE_LOG_FILE.read_text().strip().split("\n"):
            if line:
                try:
                    processed.add(json.loads(line).get("filename", ""))
                except (json.JSONDecodeError, Exception):
                    pass

    for f in sorted(RAW_DIR.glob("*")):
        if f.is_file() and f.suffix in (".md", ".txt", ".json", ".jsonl"):
            if f.name not in processed:
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) >= 20:
                    claims.append(Claim(
                        text=content,
                        source=f.name,
                        category="raw",
                    ))
                    with open(INTAKE_LOG_FILE, "a") as log:
                        log.write(json.dumps({"filename": f.name, "size": len(content),
                                               "timestamp": datetime.now().isoformat()}) + "\n")

    report.sources_scanned += len(claims)
    report.sources_processed += len(claims)
    return claims


def score_claim(claim: Claim) -> Claim:
    """Score claim confidence using heuristics."""
    text = claim.text.lower()

    # Extract entities
    claim.entities = list(set(
        w.strip(",.!?;:\"'()[]{}") for w in claim.text.split()
        if len(w.strip(",.!?;:\"'()[]{}")) > 2
        and w.strip(",.!?;:\"'()[]{}")[0].isupper()
        and w.strip(",.!?;:\"'()[]{}") not in ("The", "This", "That", "We", "It", "I")
    ))[:10]

    # Extract concepts
    tech_keywords = {"python", "typescript", "react", "rust", "docker", "postgres",
                     "redis", "kubernetes", "aws", "api", "cli", "git", "mcp",
                     "sqlite", "llm", "rag", "agent", "vector", "model", "memory"}
    claim.concepts = [kw for kw in tech_keywords if kw in text]

    # Confidence
    evidence = min(1.0, len(claim.entities) / 5.0)
    length = min(1.0, len(claim.text) / 500.0)
    claim.confidence = round(0.5 * evidence + 0.3 * length + 0.2, 3)
    return claim


def compile_to_wiki(claims: list[Claim], report: DreamReport):
    """Write accepted claims as wiki pages."""
    conn = _vault_conn()
    for claim in claims:
        title = claim.text.split("\n")[0][:80].strip("# ")
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')

        # Determine category
        if claim.concepts:
            category = "concepts"
            target_dir = CONCEPTS_DIR
        else:
            category = "entities"
            target_dir = ENTITIES_DIR

        target_dir.mkdir(parents=True, exist_ok=True)
        page_path = target_dir / f"{slug}.md"

        # Build content
        today = date.today().isoformat()
        status = "stable" if claim.confidence >= 0.8 else "draft"
        tags = ", ".join(claim.concepts[:5]) if claim.concepts else ""

        content = f"""---
title: "{title}"
created: "{today}"
updated: "{today}"
tags: [{tags}]
confidence: {claim.confidence}
status: {status}
---

# {title}

{claim.text.strip()[:500]}

_Source: {claim.source} (confidence: {claim.confidence})_
"""

        is_new = not page_path.exists()
        page_path.write_text(content)

        # Also store in vault
        conn.execute("""
            INSERT OR REPLACE INTO wiki_pages (title, content, category, path)
            VALUES (?, ?, ?, ?)
        """, (title, content, category, f"{category}/{slug}.md"))

        if is_new:
            report.pages_created += 1
        else:
            report.pages_updated += 1

    conn.commit()
    conn.close()


def detect_patterns(claims: list[Claim], report: DreamReport):
    """Detect repeated task patterns for skill creation."""
    patterns = {}
    if SKILL_PATTERNS_FILE.exists():
        try:
            patterns = json.loads(SKILL_PATTERNS_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            patterns = {"patterns": [], "skills_created": []}

    for claim in claims:
        text = claim.text.lower()
        types = []
        if re.search(r'\b(review|pr |code review)\b', text): types.append("code-review")
        if re.search(r'\b(deploy|release|ship)\b', text): types.append("deployment")
        if re.search(r'\b(test|spec|coverage)\b', text): types.append("testing")
        if re.search(r'\b(debug|bug|error|fix)\b', text): types.append("debugging")

        for t in types:
            found = next((p for p in patterns.get("patterns", []) if p["type"] == t), None)
            if found:
                found["count"] += 1
                found["last_seen"] = date.today().isoformat()
            else:
                patterns.setdefault("patterns", []).append({
                    "type": t, "count": 1,
                    "first_seen": date.today().isoformat(),
                    "last_seen": date.today().isoformat(),
                    "skill_created": False,
                })

    report.patterns_detected = len(patterns.get("patterns", []))

    for p in patterns.get("patterns", []):
        if p["count"] >= SKILL_CREATION_THRESHOLD and not p.get("skill_created"):
            _create_skill(p["type"], p["count"])
            p["skill_created"] = True
            report.skills_created += 1

    SKILL_PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SKILL_PATTERNS_FILE.write_text(json.dumps(patterns, indent=2))


def _create_skill(task_type: str, count: int):
    """Auto-generate a skill from a detected pattern."""
    skill_dir = PI_SKILLS_DIR / f"auto-{task_type}"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: auto-{task_type}
description: "Auto-generated from {count} repeated task patterns"
version: 1.0.0
---

# {task_type.title().replace('-', ' ')}

Auto-generated skill based on {count} detected task patterns.

## When to Use

User task matches the {task_type} pattern.

## Procedure

1. Follow standard workflow for this task type
2. Verify results before declaring completion
""")


def log_action(action: str, detail: str = ""):
    if LOG_FILE.exists() or LOG_FILE.parent.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"## {date.today().isoformat()} | {action} | {detail}\n")


def run_dream_cycle(idle_seconds: float = 600, verbose: bool = True) -> DreamReport:
    """Execute one complete dream cycle."""
    ALL_DIRS[0].parent.mkdir(parents=True, exist_ok=True)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    start = time.time()
    report = DreamReport(timestamp=datetime.now().isoformat())

    if verbose:
        print("🌙 Dream agent waking...")

    # Phase 0: Budget
    budget = Budget(total_seconds=min(idle_seconds * 0.25, 7200))
    budget.intake = budget.total_seconds * 0.4
    budget.compile = budget.total_seconds * 0.3
    budget.improve = budget.total_seconds * 0.2
    budget.lint = budget.total_seconds * 0.1

    if verbose:
        print(f"  ⏱ Budget: {budget.total_seconds:.0f}s")

    # Phase 1: Extract
    claims = extract_from_vault(report)
    claims.extend(extract_from_raw(report))
    report.claims_extracted = len(claims)

    if not claims:
        if verbose:
            print("  ℹ  No new content. Skipping.")
        return report

    # Phase 2: Refine
    accepted = []
    for claim in claims:
        score_claim(claim)
        if claim.confidence >= CONFIDENCE_AUTO:
            accepted.append(claim)
            report.claims_accepted += 1
        else:
            report.claims_rejected += 1

    if verbose:
        print(f"  → {len(accepted)}/{len(claims)} claims accepted")

    # Phase 3: Compile
    if accepted:
        compile_to_wiki(accepted, report)
        if verbose:
            print(f"  → {report.pages_created} created, {report.pages_updated} updated")

    # Phase 4: Patterns
    detect_patterns(claims, report)

    # Phase 6: Improve (simple structural fixes)
    for page_dir in [CONCEPTS_DIR, ENTITIES_DIR]:
        if page_dir.exists():
            for f in page_dir.glob("*.md"):
                content = f.read_text()
                if "confidence:" not in content:
                    content = content.replace("---\n", "---\nconfidence: 0.5\n", 1)
                    f.write_text(content)
                    report.improvements_made += 1

    report.duration = round(time.time() - start, 2)

    if verbose:
        print(f"  ✅ Dream cycle complete ({report.duration}s)")
        print(f"     Claims: {report.claims_extracted} → {report.claims_accepted} accepted")
        print(f"     Pages: {report.pages_created} created, {report.pages_updated} updated")
        if report.improvements_made:
            print(f"     Improvements: {report.improvements_made}")

    log_action("dream-cycle",
               f"claims={report.claims_extracted}/{report.claims_accepted}, "
               f"pages={report.pages_created}+{report.pages_updated}, "
               f"skills={report.skills_created}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dream Agent (aaa-memory)")
    parser.add_argument("--idle", type=int, default=600)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run_dream_cycle(idle_seconds=args.idle, verbose=not args.quiet)
    if not args.quiet:
        print(json.dumps(asdict(report), indent=2))
