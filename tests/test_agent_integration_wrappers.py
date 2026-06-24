"""Tests for CLI-backed agent integration wrappers."""

import json
import subprocess


def test_opencode_search_uses_clawmem_json_flag(monkeypatch):
    import aaa_memory.agent_integration.opencode as opencode

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([{"title": "real result"}]))

    monkeypatch.setattr(opencode.subprocess, "run", fake_run)

    assert opencode.search("memory query", limit=3) == [{"title": "real result"}]
    assert calls[0][0] == [
        opencode.CLAWMEM_BIN,
        "search",
        "memory query",
        "-n",
        "3",
        "--json",
    ]


def test_opencode_parser_remains_exported():
    from aaa_memory.agent_integration import parse_opencode_sessions

    assert parse_opencode_sessions() == []


def test_store_turn_uses_diary_write(monkeypatch):
    import aaa_memory.agent_integration.codex as codex
    import aaa_memory.agent_integration.opencode as opencode
    import aaa_memory.agent_integration.pi as pi

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="saved")

    for module, agent in [(codex, "codex"), (opencode, "opencode"), (pi, "pi")]:
        monkeypatch.setattr(module.subprocess, "run", fake_run)
        assert module.store_turn("prompt text", "response text", "session-1") is True
        cmd, kwargs = calls.pop()
        assert cmd[:3] == [module.CLAWMEM_BIN, "diary", "write"]
        assert "session_id: session-1" in cmd[3]
        assert "prompt:\nprompt text" in cmd[3]
        assert "response:\nresponse text" in cmd[3]
        assert cmd[-4:] == ["-t", "agent-turn", "-a", agent]
        assert kwargs["timeout"] == 15
