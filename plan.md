╭─ ⚕ Hermes ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮

    date: 2026-06-22 20:18:10 PDT
    ver: 1.0.0
    author: Sliither
    model: gpt-5.5
    tags: [aaa-memory, plugins, hooks, claude-code, codex, pi, ante, memory, performance]


    Ice-ninja, conclusion first: yes, this is possible. The fastest path is not MCP. It is native lifecycle integration
per agent, backed by a shared fast aaa-memory core.

    The current setup is slow mostly because some Python entrypoints import the whole aaa_memory package, and aaa_memory.init eagerly imports modules that drag in openai/httpx/classifier/extractor/embedding code. Direct SQLite access is sub-millisecond. The hook CLI path is about 2.1 seconds.

    Evidence from local checks:
      Repo resolved to: /home/cheta/code/aaa-memory
      Hermes plugin exists at: /home/cheta/.hermes/plugins/aaa-memory/init.py
      Current generic hook exists at: /home/cheta/.agent-hooks/aaa-memory-hook.py
      Codex already has aaa-memory hook entries in: /home/cheta/.codex/hooks.json
      Ante already has aaa-memory hook entries in: /home/cheta/.ante/settings.json
      Pi has a TS extension pattern at: /home/cheta/.pi/agent/extensions/wiki-memory-hooks.ts
      aaa-memory CLI benchmark:
        python3 scripts/mem.py stats: elapsed 2.19s, maxrss 71680 KB
        python3 scripts/mem.py inject: elapsed 2.06s, maxrss 72200 KB
        direct sqlite count query: 0.599 ms

    My read of the task:
      Evaluate whether aaa-memory can become a native plugin or extension for Claude Code, Codex, Pi, and Ante, similar
to the Hermes plugin, without relying on MCP, with native memory intake and recall, and identify the hard parts plus an
implementation plan. That is what the careful Sliither investigated.

    Feasibility by agent

    1. Hermes
      Status: already basically solved.
      Current integration: true Hermes memory provider plugin.
      Local evidence:
        /home/cheta/.hermes/plugins/aaa-memory/init.py defines AaaMemoryProvider.
        It implements:
          system_prompt_block
          prefetch
          sync_turn
          get_tool_schemas
          handle_tool_call
          on_memory_write
          on_session_end
          backup_paths
      Difficulty: low.
      Main work:
        Move the plugin into the aaa-memory repo as a distributable adapter.
        Replace hardcoded /home/cheta/code/aaa-memory assumptions.
        Use a fast core API rather than ad hoc importlib loading.
        Make prefetch query-aware and project-aware.
        Keep tool exposure optional, because native recall/intake should work without user-triggered MCP/tool calls.

    2. Claude Code
      Status: feasible through native Claude Code plugin plus hooks.
      Web grounding:
        Claude Code plugins can package skills, agents, hooks, and MCP servers.
        Source: https://docs.claude.com/en/docs/claude-code/plugins
        Claude Code hooks include SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop, PreCompact, PostCompact, SessionEnd, InstructionsLoaded, and more.
        Source: https://code.claude.com/docs/en/hooks
      Difficulty: medium.
      Best shape:
        A .claude-plugin/plugin.json package.
        Hooks for:
          SessionStart: load stable user/profile/project memory.
          UserPromptSubmit: recall relevant memory for the prompt and inject as context.
          Stop or SessionEnd: capture explicit memory directives and durable outcomes.
          PreCompact: store compacted summary or snapshot before context loss.
        Optional skill file only tells Claude what memory means, not how to manually call it.
      Main difficulty:
        Hook output contract and context injection must be exact. Claude hooks are powerful, but the adapter must be deterministic, fast, and never block the session.
        Claude has official plugin distribution, so packaging is clean.

    3. Codex
      Status: feasible through native Codex plugin plus hooks.
      Web grounding:
        Codex supports AGENTS.md custom instructions.
        Source: https://developers.openai.com/codex/guides/agents-md
        Codex has hooks documentation and plugin build documentation.
        Source: https://developers.openai.com/codex/hooks
        Source: https://developers.openai.com/codex/plugins/build
      Local evidence:
        /home/cheta/.codex/hooks.json already runs /home/cheta/.agent-hooks/aaa-memory-hook.py on UserPromptSubmit and Stop.
        /home/cheta/.codex/config.toml has [features] memories = true and mcp_servers.aaa-memory, but that is separate from native lifecycle memory.
      Difficulty: medium-high.
      Best shape:
        A .codex-plugin plugin with bundled lifecycle hooks.
        Hook events:
          SessionStart: inject project/user context.
          UserPromptSubmit: prompt-specific recall.
          Stop: capture explicit memory and durable decisions.
          PreCompact if Codex exposes it reliably in the installed version.
        Keep AGENTS.md limited to policy/context, not memory retrieval mechanics.
      Main difficulty:
        Codex plugin + hooks are newer and still have edge-case churn. I found a current public issue saying plugin-bundled hooks are supported by docs/runtime but the bundled validator may lag in places.
        Source: https://github.com/openai/codex/issues/27141
        So the fallback should remain raw hooks.json installation until plugin validation is proven on Ice-ninja's installed Codex.

    4. Pi
      Status: very feasible, likely the best non-Hermes target.
      Web grounding:
        Pi extensions are TypeScript modules. They can subscribe to lifecycle events, register tools, add commands, inject context, customize compaction, and persist state.
        Source: https://pi.dev/docs/latest/extensions
        Pi memory precedent exists. Honcho's Pi memory extension syncs user/assistant messages after each response and injects cached user profile/project context with zero network latency.
        Source: https://honcho.dev/docs/v3/guides/community/pi-honcho-memory
      Local evidence:
        /home/cheta/.pi/agent/extensions/wiki-memory-hooks.ts already demonstrates:
          session_start
          resources_discover
          before_agent_start
          session_before_compact
          session_shutdown
          context message injection
          fire-and-forget background compute
      Difficulty: medium.
      Best shape:
        Native TypeScript extension, not Python hooks.
        Direct SQLite read for recall.
        Async background write for capture.
        Event map:
          before_agent_start: inject stable memory block.
          per-turn event if available: inject prompt-specific memory.
          session_before_compact: store summary and durable outcomes.
          session_shutdown: fire-and-forget session capture.
          registerTool: optional aaa_memory_search and aaa_memory_store tools for active recall, but not required for native recall.
      Main difficulty:
        Need a Pi-packaged SQLite dependency or a small local helper binary. Shelling to Python would keep the slow path unless the Python core is fixed.

    5. Ante
      Status: possible locally, but less externally grounded.
      Local evidence:
        /home/cheta/.ante/settings.json has:
          hooks.rules for pre_tool_use, pre_compact, session_end, user_prompt_submit
          memory.dbPath = /home/cheta/.cache/aaa-memory/vault.sqlite
          extensibilityEnabled = true
        /home/cheta/.ante/hooks/wiki_memory_wrapper.py already exists as a hook wrapper.
      Difficulty: medium-high.
      Best shape:
        Treat Ante as a local plugin/settings package rather than a public marketplace plugin unless better docs exist.
        Hook events:
          user_prompt_submit: recall and inject context if Ante supports hook output context.
          pre_compact: store snapshot.
          session_end: capture durable memory.
        Keep a settings fragment plus hook adapter under aaa-memory, installable into ~/.ante.
      Main difficulty:
        Public documentation was not discoverable in this pass. The local settings contract is clear enough to build against, but verification must happen against the installed Ante runtime.

    Core blocker: aaa-memory hot path is not plugin-ready yet

    The important problem is not whether the agents support plugins. They do. The problem is that aaa-memory's current Python import graph and schema are not yet right for fast native lifecycle hooks.

    Specific difficulties found:

    1. Eager imports make simple hooks slow.
      File:
        /home/cheta/code/aaa-memory/src/aaa_memory/init.py
      Problem:
        Importing aaa_memory.hot.mem_store causes aaa_memory.init to import classifier, extractor, embedding, router, audit, cli.
      Evidence:
        importtime showed aaa_memory.hot.mem_store cumulative import at 1727.5 ms.
        openai import path consumed about 1514 ms through classifier.llm_classifier.
      Fix class:
        Make init.py lazy and minimal.
        Or make all hook adapters import by file path or import aaa_memory.fast only.

    2. The SQLite backend is fast, but the CLI wrapper is slow.
      Evidence:
        Direct sqlite count: 0.599 ms.
        scripts/mem.py stats: 2.19s.
      Meaning:
        A fast native plugin is realistic. The bottleneck is packaging/import design, not storage.

    3. Hot search is weak and partially broken.
      File:
        /home/cheta/code/aaa-memory/src/aaa_memory/retrieval/pipeline.py
      Problem:
        _hot_search tries:
          FROM turns WHERE turns MATCH ?
        But turns is a normal SQLite table, not an FTS virtual table.
      Result:
        That path catches sqlite OperationalError and silently falls back.
      Current recall mostly depends on:
        hot_memories keyword loop
        wiki_pages FTS
        ClawMem if running
      Fix class:
        Add turns_fts and hot_memories_fts virtual tables.
        Or make hot_memories itself FTS-backed.
        Add triggers to keep FTS synced.

    4. The vault is currently empty.
      Evidence:
        scripts/mem.py stats returned total 0 and projects {}.
      Meaning:
        Existing hooks are installed, but either they have not captured data, are writing elsewhere, or capture criteria are too narrow.
      Fix class:
        Add deterministic smoke tests:
          feed synthetic UserPromptSubmit JSON
          verify hot_memories row appears
          feed synthetic Stop JSON with transcript_path
          verify extracted memory appears
          run recall
          verify injected context string appears

    5. Hardcoded paths are scattered.
      Examples:
        Hermes plugin hardcodes ~/code/aaa-memory.
        config.py uses ~/.cache/aaa-memory and ~/knowledge/wiki.
        README says ~/ai-wiki/pages and ~/ai-wiki/raw.
        capture-bridge.py hardcodes /home/misscheta/knowledge/raw/web.
      Fix class:
        One config resolver:
          AAA_MEMORY_ROOT
          AAA_MEMORY_VAULT
          AAA_MEMORY_WIKI
          current user home
          per-agent project cwd
        No /home/misscheta, no hardcoded /home/cheta inside distributable code.

    6. Ingestion policy is too crude.
      Current hook captures only explicit markers like remember, note, for future reference, prefers.
      That avoids spam but misses many durable decisions.
      A native memory layer needs:
        explicit user preference capture
        durable decision capture
        project fact capture
        session summary capture
        noise rejection
        secret redaction
      Hard part:
        Doing that without sending every turn through a slow LLM classifier.

    7. Multi-agent concurrency needs SQLite discipline.
      Claude, Codex, Pi, Ante, Hermes can all run at once.
      Required:
        WAL mode
        busy_timeout
        atomic inserts
        idempotency keys per event
        no long write transactions inside hooks

    8. Native injection semantics differ per agent.
      Hermes has a MemoryProvider API.
      Claude has hook JSON output and plugin packaging.
      Codex has hooks and plugin packaging, but docs/runtime/validator may lag.
      Pi has TypeScript extensions and richer lifecycle control.
      Ante has local hook rules, but public docs are sparse.
      One adapter cannot be copied blindly. The core can be shared, but each native wrapper must respect its host's lifecycle contract.

    Recommended architecture

    Build aaa-memory as two layers:

    Layer 1: fast memory core
      Purpose: one tiny stable interface used by every agent adapter.
      Must not import openai, torch, sentence-transformers, kuzu, rich, or requests.
      Operations:
        recall(query, agent, cwd, session_id, limit, token_budget) -> context block + raw rows
        store_fact(content, source, project, tags)
        store_turn_summary(session_id, agent, text, metadata)
        capture_hook_event(event_json, agent) -> write count
        health() -> schema and timing info

      Implementation choices:
        Option A: Python aaa_memory.fast module with only stdlib sqlite3/json/pathlib/re.
        Option B: Node/TypeScript package for Pi/Codex/Claude plus Python core for Hermes.
        Option C: small compiled binary for all hooks.
      Best first move:
        Option A. It will be fast enough if imports are fixed. Direct SQLite is already sub-ms.

    Layer 2: native adapters
      Hermes:
        MemoryProvider plugin using aaa_memory.fast.
      Claude Code:
        .claude-plugin package with hooks and a minimal skill.
      Codex:
        .codex-plugin package with bundled hooks, plus hooks.json fallback.
      Pi:
        TypeScript extension using direct SQLite or the fast helper.
      Ante:
        settings fragment + hook scripts, verified against local runtime.

    Execution plan

    Phase 1: define the adapter contract
      Deliverable:
        docs/native-plugin-contract.md
      Contents:
        normalized event schema
        context block format
        memory row schema
        failure policy
        timing budget
      Target:
        Hook must never block an agent if aaa-memory fails.
        Recall path target under 50 ms warm, under 150 ms cold start.
        Capture path target under 100 ms.

    Phase 2: create aaa_memory.fast
      Deliverable:
        src/aaa_memory/fast.py
      Work:
        Move VaultMemoryStore equivalent into a dependency-light module.
        Avoid importing aaa_memory.init.
        Add WAL and busy_timeout.
        Add FTS tables:
          hot_memories_fts
          turns_fts
        Add migrations.
        Add context renderer with token/char budget.
      Verification:
        python3 -X importtime -c 'from aaa_memory.fast import recall'
        benchmark recall/store/inject with synthetic data.
      Expected improvement:
        From about 2.1s per hook to tens of ms.

    Phase 3: replace generic hook bridge with fast core
      Deliverable:
        scripts/aaa-memory-hook.py or installed aaa-memory-hook command.
      Work:
        Parse Claude/Codex/Ante hook JSON variants.
        Normalize event names.
        Store explicit directives on UserPromptSubmit.
        Store summaries or transcript-derived facts on Stop/session_end/pre_compact.
        Print host-specific allow/context output.
      Verification:
        Synthetic event fixtures for Claude, Codex, Ante.

    Phase 4: build Hermes plugin package
      Deliverable:
        integrations/hermes/aaa-memory/
      Work:
        Move current ~/.hermes/plugins/aaa-memory implementation into repo.
        Replace hardcoded paths.
        Use aaa_memory.fast.
        Keep optional tools:
          aaa_memory_search
          aaa_memory_store
        Keep native memory:
          prefetch
          sync_turn
          on_memory_write
          on_session_end
      Verification:
        hermes memory status
        start fresh Hermes session with known memory
        verify memory block is injected without MCP call

    Phase 5: build Claude Code plugin
      Deliverable:
        integrations/claude-code/aaa-memory/.claude-plugin/plugin.json
        integrations/claude-code/aaa-memory/hooks/.py or hooks/.js
        integrations/claude-code/aaa-memory/skills/aaa-memory.md
      Work:
        Package hooks:
          SessionStart
          UserPromptSubmit
          Stop
          PreCompact
          SessionEnd if available
        Use plugin manifest and local plugin-dir install path first.
      Verification:
        claude --plugin-dir integrations/claude-code/aaa-memory -p '...'
        Hook event fixture tests.
        Real prompt recall test.
      Risk:
        Hook output injection contract must be verified carefully.

    Phase 6: build Codex plugin
      Deliverable:
        integrations/codex/aaa-memory/.codex-plugin/plugin.json
        integrations/codex/aaa-memory/hooks/hooks.json
        integrations/codex/aaa-memory/hooks/aaa-memory-hook.py
      Work:
        Package UserPromptSubmit and Stop hooks first.
        Add SessionStart after prompt injection behavior is verified.
        Maintain direct ~/.codex/hooks.json installer fallback.
      Verification:
        codex plugin marketplace local install if supported.
        hooks.json fallback installation test.
        codex exec with synthetic memory recall.
      Risk:
        Plugin hook packaging may be ahead of validator behavior in some Codex versions.

    Phase 7: build Pi extension
      Deliverable:
        integrations/pi/aaa-memory-extension/index.ts
      Work:
        Native TS extension:
          before_agent_start context injection
          session_before_compact capture
          session_shutdown capture
          optional commands /aaa-memory-status and /aaa-memory-recall
          optional tools aaa_memory_search and aaa_memory_store
        Prefer direct SQLite or fast helper with persistent process.
      Verification:
        pi -e integrations/pi/aaa-memory-extension/index.ts -p '...'
        Confirm context injection and capture.
      Risk:
        SQLite dependency packaging. If this is annoying, use fast helper subprocess only after Python import issue is fixed.

    Phase 8: build Ante adapter
      Deliverable:
        integrations/ante/settings.fragment.json
        integrations/ante/hooks/aaa-memory-hook.py
      Work:
        Mirror existing local ~/.ante/settings.json contract.
        Do not depend on MCP.
        Keep output {"type":"allow"} for observational events.
        Add context output only after confirming Ante accepts injected context from hooks.
      Verification:
        Synthetic event fixtures.
        Real Ante run with debug logs.
      Risk:
        Public docs are sparse, so runtime probing is mandatory.

    Phase 9: smoke-test matrix
      Deliverable:
        tests/integrations/test_native_plugins.py
      Matrix:
        Hermes: prefetch + sync_turn
        Claude: UserPromptSubmit + Stop event fixture
        Codex: UserPromptSubmit + Stop event fixture
        Pi: extension event harness if available, otherwise fixture runner
        Ante: settings hook fixture
      Success criteria:
        Store explicit fact.
        Recall fact from another agent.
        Inject fact into prompt context.
        Agent continues if aaa-memory is missing or vault locked.
        Hook path stays under timing budget.

    Hardest problems to overcome

    1. Reliable context injection per host.
      Storage is easy. Making the memory appear naturally before the model thinks is the hard part.
      Hermes and Pi are cleanest.
      Claude and Codex require precise hook output behavior.
      Ante needs local contract verification.

    2. Fast hot path.
      Current Python CLI is too slow for every prompt. Fixing aaa_memory.init and creating aaa_memory.fast is mandatory.

    3. Avoiding memory spam.
      Native intake should not store every noisy turn as a durable fact.
      Need rule-based capture plus optional delayed summarization, not LLM classification in every hook.

    4. Schema migration.
      Current turns table is not FTS-searchable despite retrieval code expecting MATCH.
      Without FTS fixes, recall quality will stay weak.

    5. Packaging drift across agents.
      Claude plugins, Codex plugins, Pi extensions, and Ante settings are four different formats.
      The shared core prevents logic drift, but each adapter still needs its own installer and verifier.

    6. Safety and privacy.
      Hooks see prompts, transcripts, tool inputs, and file paths.
      The adapter must redact secrets before storage and skip secret-looking paths.
      This matters because memory is cross-agent and persistent.

    7. Cross-machine consistency.
      If Worker A/B/Pi machines all write to the same logical memory, there needs to be a sync story:
        local per-machine vault plus merge
        shared NFS/Syncthing vault with SQLite constraints
        or small local HTTP daemon
      Direct shared SQLite over flaky sync is risky.

    Best answer to the original idea

    Yes, build it. Do not build it as MCP. MCP can remain optional for manual recall, but native memory should use lifecycle hooks/plugins/extensions.

    The correct architecture is:
      fast local SQLite memory core
      per-agent native adapter
      no per-prompt heavy Python package imports
      no LLM calls in hook hot path
      optional background summarizer/dream agent for higher-quality consolidation

    The first technical target should be speed:
      Fix aaa_memory.init eager imports.
      Add aaa_memory.fast.
      Add FTS for hot_memories and turns.
      Replace current hook script with fast core.
      Then package adapters.


    Grounding sources used:
      Claude Code hooks reference:
        https://code.claude.com/docs/en/hooks
      Claude Code plugins:
        https://docs.claude.com/en/docs/claude-code/plugins
      Codex AGENTS.md:
        https://developers.openai.com/codex/guides/agents-md
      Codex hooks:
        https://developers.openai.com/codex/hooks
      Codex plugin build docs:
        https://developers.openai.com/codex/plugins/build
      Pi extensions:
        https://pi.dev/docs/latest/extensions
      Pi Honcho memory precedent:
        https://honcho.dev/docs/v3/guides/community/pi-honcho-memory
      Cross-agent memory precedent:
        https://github.com/rohitg00/agentmemory
      Codex plugin hook validator drift:
        https://github.com/openai/codex/issues/27141

    No files were modified in this investigation.