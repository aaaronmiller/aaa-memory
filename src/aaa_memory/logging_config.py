"""Structured JSON logging with rotation."""
import json, logging, os
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(os.getenv("AAA_MEMORY_LOG_DIR", str(Path.home() / ".cache/aaa-memory/logs")))

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "extra": getattr(record, "extra", {}),
        })

def setup_logging(name: str = "aaa-memory") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = RotatingFileHandler(LOG_DIR / f"{name}.log", maxBytes=50 * 1024 * 1024, backupCount=5)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger
