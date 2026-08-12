# Topic: Proxy / Shim (The Gatekeeper)

## Summary
The Claude-Proxy wrapper that intercepts user prompts and model outputs, injects context from memory, and sanitizes data for storage. This is the central hub of the system.

---

## Role

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

| Component | Role | Responsibility |
|-----------|------|---------------|
| The Proxy (Shim) | The Gatekeeper | Wraps `claude` command. Intercepts user prompts and model outputs. Injects context from memory. Sanitizes data via Pydantic schemas before storage. |

---

## Proxy Logic Flow

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 3.2)

1. **Intercept:** Capture `stdin` (User Prompt)
2. **Enrich:**
   - Run Classification (Intent Detection)
   - Query Graphiti (Warm) + ByteRover (Hot)
   - Inject: Prepend relevant context as a "System Note"
3. **Execute:** Pass modified payload to the real `claude` binary
4. **Capture:** Read the resulting `stdout` and log files
5. **Sanitize:** Pass output to Local LLM (Structure Gate) to strip noise
6. **Ingest:** Write structured JSON to `~/.byterover/inbox/`

---

## Architecture

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 3.1)

The Claude-Proxy (Python) sits at the center with spokes:

- **North:** StdIO Interface (User Terminal)
- **South:** Anthropic API (Claude Code Execution)
- **East (Storage):**
  - ByteRover Interface (File I/O)
  - Graphiti Interface (Bolt Protocol to FalkorDB)
  - MemVid Interface (FFmpeg + FAISS)
- **West (Compute):**
  - OpenRouter API (Sleep-Time Models)
  - Local LLM (Ollama — Pydantic Guardrails)

---

## User Experience

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 1.6)

- **Transparent Operation:** User types `claude` as normal. Proxy handles all complexity invisibly.
- **Context Injection:** "God Mode" automatically prepends relevant Hot/Warm memory based on intent classification.
- **Feedback Loop:** If user explicitly praises/scolds the agent, the Proxy tags that interaction for high-priority processing by the Tribunal during sleep.

---

## Installation

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 3.5)

A single `install.sh` script that:
1. Sets up Python `venv`
2. Installs `ffmpeg`, `ghostscript` (for QR)
3. Aliases `claude` to `python ~/.bin/claude_proxy.py`
4. Registers the `sleep_daemon` with `launchd`

---

## Data Sanitization

> **Source:** [gemini-prd.md](../gemini-prd.md)

- Uses Pydantic schemas for strict data validation
- Local LLM (Ollama) acts as Structure Gate to strip conversational noise
- Output format: Structured JSON with fields: type, summary, content, tags, timestamp

---

## Implementation Requirements

1. Create `claude_proxy.py` wrapper script
2. Implement stdin/stdout interception
3. Build intent classification module
4. Implement context retrieval from ByteRover (Hot) and Graphiti (Warm)
5. Build context injection (System Note prepending)
6. Implement output capture and sanitization via Pydantic
7. Build Local LLM integration (Ollama) for structure gating
8. Create JSONL writer for ByteRover inbox
9. Implement feedback detection (praise/scold tagging)
10. Create `install.sh` for turnkey setup

---

## Conflicts / Ambiguities

- **⚠️ Local LLM dependency:** The proxy requires a local LLM (Ollama) for sanitization. This adds a dependency that may not be available on all systems. Could be made optional with a simpler regex-based fallback.
- **⚠️ macOS-specific:** `launchd` registration is macOS-only. Linux would need systemd, Windows would need a service. Should be abstracted.
- **⚠️ Claude binary wrapping:** Assumes the `claude` CLI binary exists and can be wrapped. The exact interception mechanism depends on Claude Code's CLI interface.
