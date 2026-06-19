"""Single source of truth for all aaa-memory paths and settings."""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
HOME = Path.home()
CACHE = Path(os.getenv("AAA_MEMORY_CACHE", str(HOME / ".cache/aaa-memory")))
VAULT = CACHE / "vault.sqlite"
COLD_VAULT = CACHE / "cold.sqlite"
LOG_DIR = Path(os.getenv("AAA_MEMORY_LOG_DIR", str(CACHE / "logs")))
WIKI_BASE = Path(os.getenv("AAA_MEMORY_WIKI", str(HOME / "knowledge/wiki")))
PROJECTS_DIR = Path(os.getenv("AAA_MEMORY_PROJECTS", str(HOME / "knowledge/projects")))
RAW_DIR = Path(os.getenv("AAA_MEMORY_RAW", str(HOME / "knowledge/raw")))
CACHE_FILE = CACHE / "session_cache.json"

# ── Agent discovery paths ────────────────────────────────────────────────────
AGENT_PATHS = {
    "claude-code": HOME / ".claude/sessions",
    "openclaw": HOME / ".openclaw/sessions",
    "hermes": HOME / ".hermes/state.db",
    "qwen": HOME / ".qwen/context",
    "opencode": HOME / ".opencode/sessions",
    "codex": HOME / ".codex/rollouts",
    "web": HOME / "knowledge/raw/web",
}

# ── Search defaults ──────────────────────────────────────────────────────────
TOP_K = 20
MAX_TOKENS = 2000

# ── Secret patterns (shared) ─────────────────────────────────────────────────
SECRET_PATTERNS = [
    ".env", ".env.*", "*key*", "*secret*", "*token*",
    "*credential*", "*password*", "*.pem", "*.key",
    "*auth*", "*access*", "*api-key*",
    "*passwd*", "*private*", "*ssh*", "*cert*",
    "*.p12", "*.jks", "*keystore*", "*truststore*",
]

# ── Ensure cache directories exist ──────────────────────────────────────────
def ensure_dirs():
    for d in [CACHE, LOG_DIR, WIKI_BASE.parent, PROJECTS_DIR.parent]:
        d.mkdir(parents=True, exist_ok=True)


def validate():
    """Validate all paths and config on startup."""
    issues = []
    for name, path in AGENT_PATHS.items():
        if not path.exists():
            issues.append(f"Agent path not found: {name} -> {path}")
    if not VAULT.parent.exists():
        issues.append(f"Vault parent dir doesn't exist: {VAULT.parent}")
    if issues:
        import logging
        log = logging.getLogger("aaa-memory.config")
        for issue in issues:
            log.warning(issue)
    return len(issues) == 0

# ── Lazy getters (for testing with env var overrides) ─────────────────────────
def get_vault():
    return Path(os.getenv("AAA_MEMORY_VAULT", str(CACHE / "vault.sqlite")))

def get_cold_vault():
    return Path(os.getenv("AAA_MEMORY_COLD_VAULT", str(CACHE / "cold.sqlite")))

def get_wiki():
    return Path(os.getenv("AAA_MEMORY_WIKI", str(HOME / "knowledge/wiki")))
