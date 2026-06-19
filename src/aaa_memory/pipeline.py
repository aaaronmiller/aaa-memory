"""Pipeline orchestrator — chains classify → extract → embed → compile → index."""
import json, logging
from pathlib import Path
from typing import List, Optional
from aaa_memory import config

logger = logging.getLogger("aaa-memory.pipeline")


class Pipeline:
    """Orchestrates the full aaa-memory ingestion pipeline."""

    def __init__(self, vault_path=None, wiki_base=None):
        self.vault = Path(vault_path or config.VAULT)
        self.wiki = Path(wiki_base or config.WIKI_BASE)
        config.ensure_dirs()

    def process_file(self, filepath: Path, llm_api_key: Optional[str] = None) -> dict:
        """Process a single file through the full pipeline."""
        from aaa_memory.classifier.tuned import classify
        from aaa_memory.extractor.llm_extractor import extract, extract_fallback
        from aaa_memory.metadata.injector import inject_metadata
        from aaa_memory.wiki.compiler import compile_to_wiki

        logger.info(f"Processing {filepath.name}")

        # Step 1: Classify
        result = classify(filepath, llm_fallback=llm_api_key is not None)
        logger.info(f"  Classified as: {result.category} (conf={result.confidence:.2f})")

        # Step 2: Extract
        text = filepath.read_text(errors="replace")
        if llm_api_key:
            elements = extract(text, api_key=llm_api_key)
        else:
            elements = extract_fallback(text)
        logger.info(f"  Extracted {len(elements)} elements")

        # Step 3: Inject metadata
        for elem in elements:
            inject_metadata(elem, source_file=str(filepath), classification=result.category)

        # Step 4: Compile to wiki
        wiki_files = compile_to_wiki(elements, wiki_base=self.wiki)
        logger.info(f"  Compiled {len(wiki_files)} wiki files")

        # Step 5: Index into vault
        self._index_wiki()

        return {
            "file": str(filepath),
            "classification": result.category,
            "elements": len(elements),
            "wiki_files": wiki_files,
        }

    def process_batch(self, files: List[Path], batch_size: int = 20, checkpoint: bool = True) -> List[dict]:
        """Process multiple files with optional checkpointing."""
        results = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            for f in batch:
                try:
                    r = self.process_file(f)
                    results.append(r)
                except Exception as e:
                    logger.error(f"Failed to process {f}: {e}")
                    results.append({"file": str(f), "error": str(e)})
            if checkpoint:
                self._save_checkpoint(i // batch_size + 1, results)
        return results

    def _index_wiki(self):
        """Index wiki files into vault FTS5."""
        import sqlite3
        self.vault.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.vault))
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages USING fts5(title, content, category, path)")
        count = 0
        for md_file in sorted(self.wiki.rglob("*.md")):
            title = md_file.stem
            cat = md_file.parent.name
            content = md_file.read_text(errors="replace")
            try:
                conn.execute("INSERT OR REPLACE INTO wiki_pages (title, content, category, path) VALUES (?, ?, ?, ?)",
                            (title, content, cat, str(md_file.relative_to(self.wiki))))
                count += 1
            except sqlite3.OperationalError:
                continue
        conn.commit()
        conn.close()
        logger.info(f"  Indexed {count} wiki pages into vault")

    @staticmethod
    def _save_checkpoint(batch_num: int, results: List[dict]):
        path = config.CACHE / f"checkpoint_{batch_num:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"batch": batch_num, "results": results}, f, indent=2, default=str)
        logger.info(f"  Checkpoint {batch_num} saved to {path.name}")

    def search(self, query: str, limit: int = 20):
        """End-to-end search across all tiers."""
        from aaa_memory.retrieval.pipeline import search as _search
        return _search(query, limit=limit)
