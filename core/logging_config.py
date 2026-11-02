import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON without exposing sensitive data."""

    SENSITIVE_KEYS = {"api_key", "x-api-key", "gemini_api_key", "password", "token", "secret"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt) if self.datefmt else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Extra metadata provided via extra={...}
        if hasattr(record, "props") and isinstance(record.props, dict):
            payload.update(self._sanitize(record.props))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

    def _sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for k, v in data.items():
            if any(s in k.lower() for s in self.SENSITIVE_KEYS):
                clean[k] = "[REDACTED]"
            elif isinstance(v, dict):
                clean[k] = self._sanitize(v)
            else:
                clean[k] = v
        return clean


def setup_logger(name: str = "semantic_cache", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger

