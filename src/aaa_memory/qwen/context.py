"""Qwen Code integration — context file injection + MCP."""
import os, json
from pathlib import Path

def write_project_summary(project: str, summary: str):
    path = Path(f"PROJECT_SUMMARY_{project}.md")
    path.write_text(summary)

def read_context(project: str) -> str:
    path = Path(f"PROJECT_SUMMARY_{project}.md")
    return path.read_text() if path.exists() else ""
