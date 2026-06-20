#!/usr/bin/env python3
"""
Dream Agent — sleep-time compute for knowledge refinement.

Architecture (from wiki-memory specs):
  Hot:  aaa-memory vault (SQLite FTS5) — fast writes, session capture
  Warm: ClawMem (REST API on :7438) — main searchable knowledge base
  Cold: MemVid V2 (monthly snapshots) — not yet implemented

The dream agent reads from ClawMem (warm tier), refines content,
writes wiki pages, and re-indexes ClawMem.

Phases:
  0. Budget allocation (idle_seconds × 0.25, capped at 7200s)
  1. Extract — read new/updated docs from ClawMem
  2. Refine — confidence scoring (35% consistency, 25% freshness, 25% cross-ref, 15% evidence)
  3. Compile — write wiki pages with YAML frontmatter
  4. Pattern Detect — repeated tasks → auto-skill creation
  5. Re-index — trigger ClawMem reindex
  6. Improve — vault quality (structural fixes)
"""

import os
import sys
import json
import time
import re
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, date, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

AAA_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AAA_ROOT / "src"))

# ─── Configuration ───────────────────────────────────────────────
CLAWMEM_URL = os.environ.get("CLAWMEM_URL", "http://localhost:7438")
CLAWMEM_COLLECTION = os.environ.get("CLAWMEM_COLLECTION", "wiki")

AI_WIKI = Path(os.environ.get("AI_WIKI", Path.home() / "ai-wiki"))
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

# Thresholds (from spec F-005)
CONFIDENCE_AUTO = float(os.environ.get("CONFIDENCE_AUTO", "0.8"))
CONFIDENCE_FLAG = float(os.environ.get("CONFIDENCE_FLAG", "0.5"))
CONFIDENCE_REJECT = float(os.environ.get("CONFIDENCE_REJECT", "0.5"))
SKILL_CREATION_THRESHOLD = int(os.environ.get("SKILL_CREATION_THRESHOLD", "3"))

# ─── Data Classes ────────────────────────────────────────────────
@dataclass
class Budget:
    total_seconds: float
    intake: float = 0.0
    refine: float = 0.0
    compile: float = 0.0
    improve: float = 0.0
    lint: float = 0.0

@dataclass
class Claim:
    text: str
    source_docid: str
    source_filename: str
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
    claims_flagged: int = 0
    claims_rejected: int = 0
    pages_created: int = 0
    pages_updated: int = 0
    patterns_detected: int = 0
    skills_created: int = 0
    improvements_made: int = 0
    errors: list = field(default_factory=list)

# ─── ClawMem helpers ─────────────────────────────────────────────
def clawmem_available() -> bool:
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/health")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False

def clawmem_list_docs(pattern: str = "*", limit: int = 100) -> list:
    """List documents from ClawMem."""
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/documents?pattern={pattern}&limit={limit}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("documents", [])
    except Exception:
        return []

def clawmem_get_doc(docid: str) -> dict:
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/documents/{docid}", timeout=5)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}

def clawmem_reindex():
    try:
        body = json.dumps({"collection": CLAWMEM_COLLECTION}).encode()
        req = urllib.request.Request(f"{CLAWMEM_URL}/reindex", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass

# ─── Phase 0: Budget ─────────────────────────────────────────────
def allocate_budget(idle_seconds: float) -> Budget:
    total = min(idle_seconds * 0.25, 7200.0)
    b = Budget(total_seconds=total)
    b.intake = total * 0.40
    b.refine = total * 0.30
    b.compile = total * 0.20
    b.improve = total * 0.10
    return b

# ─── Phase 1: Extract from ClawMem ───────────────────────────────
def extract_from_clawmem(report: DreamReport) -> list[Claim]:
    """Extract documents from ClawMem (warm tier)."""
    claims = []
    
    # List all documents from ClawMem
    docs = clawmem_list_docs(pattern="*", limit=100)
    
    for doc in docs:
        docid = doc.get("docid", "")
        if not docid:
            continue
        
        # Get full document body
        content = doc.get("body", "")
        if not content or len(content.strip()) < 50:
            continue
        
        # Extract entities
        entities = list(set(
            w.strip(",.!?;:\"'()[]{}") for w in content.split()
            if len(w.strip(",.!?;:\"'()[]{}")) > 2
            and w.strip(",.!?;:\"'()[]{}")[0].isupper()
            and w.strip(",.!?;:\"'()[]{}") not in ("The", "This", "That", "We", "It", "I")
        ))[:10]
        
        # Extract concepts
        tech_keywords = {"python", "typescript", "react", "rust", "docker", "postgres",
                         "redis", "kubernetes", "aws", "api", "cli", "git", "mcp",
                         "sqlite", "llm", "rag", "agent", "vector", "model", "memory"}
        concepts = [kw for kw in tech_keywords if kw in content.lower()]
        
        claims.append(Claim(
            text=content[:2000],
            source_docid=docid,
            source_filename=doc.get("path", f"clawmem-{docid[:8]}"),
            category=doc.get("collection", "note"),
            entities=entities,
            concepts=concepts,
        ))
    
    report.sources_scanned = len(claims)
    report.sources_processed = len(claims)
    return claims

# ─── Phase 2: Refine ─────────────────────────────────────────────
def refine_claim(claim: Claim) -> tuple[Claim, bool]:
    """Score claim confidence (spec F-005)."""
    text = claim.text.lower()
    
    # Self-consistency (35%)
    contradiction_markers = ["however", "but", "on the other hand", "although", "nevertheless"]
    has_contradiction = any(m in text for m in contradiction_markers)
    consistency = 0.6 if has_contradiction else 0.95
    
    # Source freshness (25%) — estimate from filename
    age_days = 90.0
    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', claim.source_filename)
    if date_match:
        try:
            y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            age_days = (date.today() - date(y, m, d)).days
        except ValueError:
            pass
    freshness = max(0.0, 1.0 - age_days / 365.0)
    
    # Cross-reference agreement (25%) — check if related wiki pages exist
    agreement = 1.0
    for entity in claim.entities[:3]:
        slug = entity.lower().replace(" ", "-")
        for d in [CONCEPTS_DIR, ENTITIES_DIR]:
            if (d / f"{slug}.md").exists():
                agreement = min(agreement, 0.9)
                break
    
    # Evidence count (15%)
    evidence = min(1.0, len(claim.entities) / 5.0)
    
    # Weighted confidence
    confidence = 0.35 * consistency + 0.25 * freshness + 0.25 * agreement + 0.15 * evidence
    claim.confidence = round(confidence, 3)
    
    # Decision
    if confidence >= CONFIDENCE_AUTO:
        return claim, True
    elif confidence >= CONFIDENCE_FLAG:
        return claim, False  # Flagged for review
    else:
        return claim, False  # Rejected

# ─── Phase 3: Compile ────────────────────────────────────────────
def compile_to_wiki(claims: list[Claim], report: DreamReport):
    """Write wiki pages from accepted claims."""
    today = date.today().isoformat()
    
    for claim in claims:
        # Determine page location
        if claim.concepts:
            target_dir = CONCEPTS_DIR
            primary = claim.concepts[0]
        elif claim.entities:
            target_dir = ENTITIES_DIR
            primary = claim.entities[0]
        else:
            target_dir = SOURCES_DIR
            primary = claim.source_filename.split(".")[0]
        
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r'[^a-z0-9]+', '-', primary.lower()).strip('-')[:60]
        page_path = target_dir / f"{slug}.md"
        
        # Build content
        status = "stable" if claim.confidence >= 0.8 else "draft"
        tags = ", ".join(claim.concepts[:5]) if claim.concepts else ""
        
        content = f"""---
title: "{primary}"
created: "{today}"
updated: "{today}"
tags: [{tags}]
confidence: {claim.confidence}
status: {status}
sources:
  - clawmem://{claim.source_docid}
---

# {primary}

{claim.text.strip()[:1000]}

_Source: `{claim.source_filename}` (confidence: {claim.confidence})_
"""
        
        is_new = not page_path.exists()
        page_path.write_text(content)
        
        if is_new:
            report.pages_created += 1
        else:
            report.pages_updated += 1

# ─── Phase 4: Pattern Detect ─────────────────────────────────────
def detect_patterns(claims: list[Claim], report: DreamReport):
    """Detect repeated task patterns for auto-skill creation."""
    patterns = {}
    if SKILL_PATTERNS_FILE.exists():
        try:
            patterns = json.loads(SKILL_PATTERNS_FILE.read_text())
        except:
            patterns = {"patterns": []}
    
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

# ─── Phase 5: Re-index ───────────────────────────────────────────
def reindex_clawmem(report: DreamReport):
    """Trigger ClawMem reindex after wiki writes."""
    if clawmem_available():
        clawmem_reindex()

# ─── Phase 6: Improve ────────────────────────────────────────────
def improve_wiki(report: DreamReport):
    """Fix structural issues in wiki pages."""
    improved = 0
    for page_dir in [CONCEPTS_DIR, ENTITIES_DIR]:
        if not page_dir.exists():
            continue
        for f in page_dir.glob("*.md"):
            content = f.read_text()
            if "confidence:" not in content:
                content = content.replace("---\n", "---\nconfidence: 0.5\n", 1)
                f.write_text(content)
                improved += 1
    report.improvements_made = improved

# ─── Helpers ─────────────────────────────────────────────────────
def log_action(action: str, detail: str = ""):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"## {date.today().isoformat()} | {action} | {detail}\n")

# ─── Main ────────────────────────────────────────────────────────
def run_dream_cycle(idle_seconds: float = 600, verbose: bool = True) -> DreamReport:
    ALL_DIRS[0].parent.mkdir(parents=True, exist_ok=True)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    
    start = time.time()
    report = DreamReport(timestamp=datetime.now().isoformat())
    
    if verbose:
        print("🌙 Dream agent waking...")
    
    # Phase 0: Budget
    budget = allocate_budget(idle_seconds)
    if verbose:
        print(f"  ⏱ Budget: {budget.total_seconds:.0f}s")
    
    # Phase 1: Extract from ClawMem
    if not clawmem_available():
        if verbose:
            print("  ⚠ ClawMem offline — extracting from raw/ only")
        # Fallback to raw files
        claims = _extract_from_raw(report)
    else:
        claims = extract_from_clawmem(report)
        # Also check raw/ for new files
        claims.extend(_extract_from_raw(report))
    
    report.claims_extracted = len(claims)
    
    if not claims:
        if verbose:
            print("  ℹ  No content to process")
        return report
    
    # Phase 2: Refine
    accepted = []
    for claim in claims:
        refined, is_accepted = refine_claim(claim)
        if is_accepted:
            accepted.append(refined)
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
    
    # Phase 5: Re-index ClawMem
    reindex_clawmem(report)
    
    # Phase 6: Improve
    improve_wiki(report)
    
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


def _extract_from_raw(report: DreamReport) -> list[Claim]:
    """Fallback: extract from raw/ directory."""
    claims = []
    if not RAW_DIR.exists():
        return claims
    
    processed = set()
    if INTAKE_LOG_FILE.exists():
        for line in INTAKE_LOG_FILE.read_text().strip().split("\n"):
            if line:
                try:
                    processed.add(json.loads(line).get("filename", ""))
                except:
                    pass
    
    for f in sorted(RAW_DIR.glob("*")):
        if f.is_file() and f.suffix in (".md", ".txt", ".json", ".jsonl"):
            if f.name not in processed:
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) >= 20:
                    claims.append(Claim(
                        text=content[:2000],
                        source_docid=f"raw-{f.stem}",
                        source_filename=f.name,
                        category="raw",
                    ))
                    with open(INTAKE_LOG_FILE, "a") as log:
                        log.write(json.dumps({"filename": f.name, "size": len(content),
                                               "timestamp": datetime.now().isoformat()}) + "\n")
    
    report.sources_scanned += len(claims)
    report.sources_processed += len(claims)
    return claims


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dream Agent (aaa-memory)")
    parser.add_argument("--idle", type=int, default=600)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    report = run_dream_cycle(idle_seconds=args.idle, verbose=not args.quiet)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
