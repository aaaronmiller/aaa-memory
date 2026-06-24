import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_hook():
    spec = importlib.util.spec_from_file_location(
        "cass_context_hook",
        ROOT / "scripts" / "cass_context_hook.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cass_hook_builds_bounded_context(monkeypatch):
    hook = _load_hook()

    def fake_run_json(args, timeout=8):
        if args[:2] == ["cass", "search"]:
            return {
                "hits": [{
                    "score": 0.42,
                    "title": "memory request",
                    "content": "implement the memory system",
                    "source_path": "/tmp/session.jsonl",
                    "line_number": 1,
                }]
            }
        return [
            {"role": "user", "content": "implement the memory system for hooks"},
            {"role": "assistant", "content": "[Tool: Bash - ignored]"},
            {"role": "assistant", "content": "I found the docs."},
        ]

    monkeypatch.setattr(hook, "_run_json", fake_run_json)

    context = hook.build_context("audit memory system hooks")

    assert "<cass-prompt-history>" in context
    assert "implement the memory system" in context
    assert "[Tool:" not in context


def test_cass_hook_main_returns_empty_on_bad_input(capsys, monkeypatch):
    hook = _load_hook()

    monkeypatch.setattr("sys.stdin.read", lambda: "{bad json")

    assert hook.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hookSpecificOutput"]["additionalContext"] == ""
