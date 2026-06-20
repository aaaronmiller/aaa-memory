#!/usr/bin/env python3
"""
aaa-memory setup wizard.

Interactive setup for new installations. Detects installed agents,
configures hooks, initializes storage, and verifies everything works.

Usage:
    python3 -m aaa_memory.setup          # interactive wizard
    python3 -m aaa_memory.setup --auto   # non-interactive (defaults)
"""

import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path

AAA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AAA_ROOT / "src"))

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
AI_WIKI = Path.home() / "ai-wiki"
CLAWMEM_CONFIG = Path.home() / ".config" / "clawmem" / "index.yml"


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg): print(f"  {Colors.GREEN}✓{Colors.END} {msg}")
def warn(msg): print(f"  {Colors.YELLOW}⚠{Colors.END} {msg}")
def fail(msg): print(f"  {Colors.RED}✗{Colors.END} {msg}")
def info(msg): print(f"  {Colors.CYAN}→{Colors.END} {msg}")


def detect_agents():
    """Detect which AI agents are installed."""
    agents = {}

    # Hermes
    hermes_config = Path.home() / ".hermes" / "config.yaml"
    if hermes_config.exists():
        agents["hermes"] = {
            "config": str(hermes_config),
            "plugin_dir": str(Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory"),
        }

    # Claude Code
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        agents["claude"] = {
            "config": str(claude_md),
            "settings": str(Path.home() / ".claude" / "settings.json"),
        }

    # Pi
    pi_skills = Path.home() / ".pi" / "agent" / "skills"
    if pi_skills.exists():
        agents["pi"] = {
            "skills_dir": str(pi_skills),
        }

    # Codex
    codex_config = Path.home() / ".codex" / "config.toml"
    if codex_config.exists():
        agents["codex"] = {
            "config": str(codex_config),
        }

    # OpenCode
    opencode_config = Path.home() / ".config" / "opencode" / "config.json"
    if opencode_config.exists():
        agents["opencode"] = {
            "config": str(opencode_config),
        }

    return agents


def check_prerequisites():
    """Check system prerequisites."""
    print(f"\n{Colors.BOLD}Prerequisites{Colors.END}")

    # Python
    version = sys.version_info
    if version >= (3, 10):
        ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        fail(f"Python {version.major}.{version.minor} (need 3.10+)")
        return False

    # pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       capture_output=True, check=True)
        ok("pip installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        fail("pip not found")
        return False

    # ClawMem
    try:
        result = subprocess.run(["clawmem", "collection", "list"],
                                capture_output=True, text=True, timeout=5)
        ok("ClawMem installed")
        return True
    except FileNotFoundError:
        warn("ClawMem not installed (optional — enables FTS cold tier)")
        info("Install: npm install -g clawmem")
        return True


def create_directories():
    """Create required directories."""
    print(f"\n{Colors.BOLD}Creating directories{Colors.END}")

    dirs = [
        AI_WIKI / "raw",
        AI_WIKI / "pages" / "concepts",
        AI_WIKI / "pages" / "entities",
        AI_WIKI / "pages" / "sources",
        AI_WIKI / "pages" / "queries",
        AI_WIKI / ".meta",
        Path.home() / ".cache" / "aaa-memory",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"{d.relative_to(Path.home())}")


def init_vault():
    """Initialize the aaa-memory vault."""
    print(f"\n{Colors.BOLD}Initializing vault{Colors.END}")

    if VAULT.exists():
        conn = sqlite3.connect(str(VAULT))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        required = ["turns", "hot_memories", "wiki_pages"]
        missing = [t for t in required if t not in tables]
        if not missing:
            ok(f"Vault exists with all tables ({len(tables)} total)")
            return
        warn(f"Vault exists but missing tables: {', '.join(missing)}")

    # Create vault with all tables
    conn = sqlite3.connect(str(VAULT))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE NOT NULL,
            agent TEXT NOT NULL,
            session_id TEXT,
            turn_index INTEGER,
            turn_type TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS hot_memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            project TEXT DEFAULT 'default',
            source TEXT DEFAULT 'unknown',
            pinned INTEGER DEFAULT 0,
            created TEXT NOT NULL,
            accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages USING fts5(
            title, content, category, path,
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS wiki_meta (
            path TEXT PRIMARY KEY,
            indexed_at TEXT DEFAULT (datetime('now')),
            word_count INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_turns_agent ON turns(agent);
        CREATE INDEX IF NOT EXISTS idx_hot_project ON hot_memories(project);
    """)
    conn.commit()
    conn.close()
    ok("Vault initialized")


def setup_clawmem():
    """Set up ClawMem collections."""
    print(f"\n{Colors.BOLD}ClawMem{Colors.END}")

    try:
        result = subprocess.run(["clawmem", "collection", "list"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            warn("ClawMem not available — skipping")
            return
    except FileNotFoundError:
        warn("ClawMem not installed — skipping")
        return

    # Check if wiki collection exists
    if "wiki" in result.stdout:
        ok("Wiki collection already configured")
    else:
        info("Adding wiki collection...")
        subprocess.run(["clawmem", "collection", "add",
                        str(AI_WIKI / "pages"), "--name", "wiki"],
                       capture_output=True, timeout=10)
        ok("Wiki collection added")

    # Check ClawMem health
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:7438/health", timeout=2)
        urllib.request.urlopen(req, timeout=2)
        ok("ClawMem server running")
    except Exception:
        warn("ClawMem server not running")
        info("Start with: systemctl --user start clawmem-serve")


def configure_agent(agent_name, agent_info):
    """Configure an agent to use aaa-memory."""
    print(f"\n{Colors.BOLD}{agent_name}{Colors.END}")

    if agent_name == "hermes":
        config_path = Path(agent_info["config"])
        content = config_path.read_text()
        if "provider: aaa-memory" in content:
            ok("Already configured")
        elif "provider:" in content:
            warn("Memory provider set to something else")
            info(f"Edit {config_path} — set memory.provider to aaa-memory")
        else:
            info("Add to config.yaml under memory:")
            info("  provider: aaa-memory")

    elif agent_name == "claude":
        config_path = Path(agent_info["config"])
        content = config_path.read_text()
        if "aaa-memory" in content:
            ok("CLAUDE.md references aaa-memory")
        else:
            info("Add to CLAUDE.md:")
            info("  python3 ~/code/aaa-memory/scripts/mem.py recall \"<query>\"")

    elif agent_name == "codex":
        config_path = Path(agent_info["config"])
        content = config_path.read_text()
        if "aaa-memory" in content:
            ok("config.toml references aaa-memory")
        else:
            info("Add to config.toml:")
            info('  notify = ["python3", "~/code/aaa-memory/scripts/mem.py", "inject"]')

    elif agent_name == "pi":
        ok("Skills directory exists")
        info("Add to AGENTS.md:")
        info("  Use `python3 ~/code/aaa-memory/scripts/mem.py recall \"<query>\"`")

    else:
        info(f"Add to {agent_name}'s system prompt:")
        info("  Search: python3 ~/code/aaa-memory/scripts/mem.py recall \"<query>\"")
        info("  Store:  python3 ~/code/aaa-memory/scripts/mem.py save \"<fact>\"")


def verify():
    """Verify the installation works."""
    print(f"\n{Colors.BOLD}Verification{Colors.END}")

    # Test vault
    try:
        from aaa_memory.hot.mem_store import VaultMemoryStore
        store = VaultMemoryStore()
        stats = store.stats()
        ok(f"Vault accessible — {stats['total']} memories, {stats['pinned']} pinned")
    except Exception as e:
        fail(f"Vault test failed: {e}")

    # Test dream agent
    try:
        from aaa_memory.warm.dream import DreamReport
        ok("Dream agent importable")
    except Exception as e:
        fail(f"Dream agent import failed: {e}")

    # Test MCP
    try:
        from aaa_memory.mcp import handle_search
        ok("MCP server importable")
    except Exception as e:
        fail(f"MCP import failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="aaa-memory setup wizard")
    parser.add_argument("--auto", action="store_true", help="Non-interactive mode")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'='*50}")
    print(f"  aaa-memory Setup Wizard")
    print(f"{'='*50}{Colors.END}")

    # Step 1: Prerequisites
    if not check_prerequisites():
        fail("Fix prerequisites and re-run")
        sys.exit(1)

    # Step 2: Create directories
    create_directories()

    # Step 3: Initialize vault
    init_vault()

    # Step 4: Set up ClawMem
    setup_clawmem()

    # Step 5: Detect and configure agents
    print(f"\n{Colors.BOLD}Agent Detection{Colors.END}")
    agents = detect_agents()
    if agents:
        for name, info in agents.items():
            configure_agent(name, info)
    else:
        warn("No agents detected")
        info("See README.md for manual setup instructions")

    # Step 6: Verify
    verify()

    print(f"\n{Colors.BOLD}{'='*50}")
    print(f"  Setup complete!")
    print(f"{'='*50}{Colors.END}")
    print(f"\nNext steps:")
    print(f"  1. Test: python3 ~/code/aaa-memory/scripts/mem.py stats")
    print(f"  2. Store: python3 ~/code/aaa-memory/scripts/mem.py save \"test memory\"")
    print(f"  3. Search: python3 ~/code/aaa-memory/scripts/mem.py recall \"test\"")
    print(f"  4. Dream: python3 ~/code/aaa-memory/src/aaa_memory/warm/dream.py --idle 60")
    print()


if __name__ == "__main__":
    main()
