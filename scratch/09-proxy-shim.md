# Topic 09: Proxy Shim (The Gatekeeper)

## Summary
Implement the Claude Code proxy wrapper that intercepts user prompts, injects context from memory tiers, captures outputs, sanitizes data, and ingests structured results into ByteRover.

---

## Overview
> Sources: `gemini-prd.md` (lines 28, 115–227), `docs/UNIFIED_PRD.md` (line 54)

The Proxy (Shim) is the central hub of the system. It wraps the `claude` command, making the entire memory and self-improvement system transparent to the user.

---

## Component Role
> Source: `gemini-prd.md` (lines 28, 115–122)

| Aspect | Detail |
|--------|--------|
| **Role** | The Gatekeeper |
| **Wraps** | `claude` CLI command |
| **Functions** | Intercept prompts, inject context, capture outputs, sanitize data |
| **User Experience** | Transparent — user types `claude` as normal |

---

## Proxy Logic Flow
> Source: `gemini-prd.md` (lines 206–227)

### Step-by-step:

1. **Intercept:** Capture `stdin` (User Prompt)

2. **Enrich:**
   - Run **Classification** (Intent Detection) on the user's prompt
   - Query **Graphiti** (Warm Memory) for relevant knowledge graph nodes
   - Query **ByteRover** (Hot Memory) for recent context
   - **Inject:** Prepend retrieved context to the prompt as a "System Note"

3. **Execute:** Pass modified payload to the real `claude` binary

4. **Capture:** Read the resulting `stdout` and log files

5. **Sanitize:** Pass output to Local LLM (Structure Gate) to strip noise
   - Uses Pydantic schemas for validation
   - Local LLM (Ollama) for guardrails

6. **Ingest:** Write structured JSON to `~/.byterover/inbox/`

---

## Context Injection ("God Mode")
> Source: `gemini-prd.md` (lines 117–119)

- Automatically prepends relevant Hot/Warm memory to the user's prompt
- Based on intent classification of the user's query
- Transparent to the user — they see normal Claude behavior

## Feedback Loop
> Source: `gemini-prd.md` (lines 121–122)

- If user explicitly praises/scolds the agent, the Proxy tags that interaction
- Tagged interactions get high-priority processing by the Tribunal during sleep time

---

## Installation & Setup
> Source: `gemini-prd.md` (lines 267–282)

**Turnkey Installation:**
1. `docker-compose.yml` for FalkorDB (Graphiti backend) and Qdrant/FAISS (MemVid index)
2. `.env` file for OpenRouter keys and path configurations
3. Single `install.sh` script that:
   - Sets up Python `venv`
   - Installs `ffmpeg`, `ghostscript` (for QR)
   - Aliases `claude` to `python ~/.bin/claude_proxy.py`
   - Registers the `sleep_daemon` with `launchd`

---

## Architecture
> Source: `gemini-prd.md` (lines 181–204)

```
                    ┌─────────────┐
                    │   User      │
                    │  Terminal   │
                    └──────┬──────┘
                           │ StdIO
                    ┌──────▼──────┐
          ┌─────── │ Claude-Proxy │ ───────┐
          │        │   (Python)   │        │
          │        └──────┬──────┘        │
          │               │               │
    ┌─────▼─────┐   ┌────▼────┐   ┌──────▼──────┐
    │ Storage   │   │Anthropic│   │  Compute    │
    │           │   │  API    │   │             │
    │ ByteRover │   │ (Claude │   │ OpenRouter  │
    │ Graphiti  │   │  Code)  │   │ Local LLM   │
    │ MemVid    │   └─────────┘   │ (Ollama)    │
    └───────────┘                 └─────────────┘
```

---

## Implementation Tasks

1. Create `src/proxy/claude_proxy.py` — Main proxy script (intercept, enrich, execute, capture, sanitize, ingest)
2. Create `src/proxy/intent_classifier.py` — Classify user prompt intent for context selection
3. Create `src/proxy/context_injector.py` — Query Hot+Warm memory and format context injection
4. Create `src/proxy/output_sanitizer.py` — Pydantic-based output validation and noise stripping
5. Create `src/proxy/feedback_tagger.py` — Detect praise/criticism and tag for Tribunal
6. Create `install.sh` — Turnkey installation script
7. Create `docker-compose.yml` — FalkorDB + FAISS/Qdrant containers

---

## Conflicts & Ambiguities

1. **Local LLM for sanitization:** `gemini-prd.md` specifies Ollama for the "Structure Gate" sanitization step. This requires a local LLM running alongside the system. The specific model for sanitization is not specified — could be a small model like Phi-3 or Llama-3.

2. **Proxy vs orchestration:** The proxy handles real-time user interactions. The orchestration pipeline (Topic 08) handles batch ingestion. These are separate entry points into the system but share the same storage backends.

3. **Claude binary path:** The proxy aliases `claude` to the proxy script. The real `claude` binary needs to be accessible at a different path. Installation script must handle this carefully to avoid circular aliasing.
