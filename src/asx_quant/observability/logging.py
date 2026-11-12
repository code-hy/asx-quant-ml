"""Structured logging configuration.

Console output is human-readable; file output (when configured) is JSON-lines
so pipeline runs can be parsed by tooling later.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonLineHandler(logging.Handler):
    """Append-only JSON-lines log handler (one object per record)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.format(record)
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except OSError:  # logging must never crash the pipeline
            pass


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the root ``asx_quant`` logger.

    Idempotent: calling twice will not duplicate handlers.

    Windows consoles default to cp1252, which cannot encode common Unicode
    punctuation (em dashes, arrows). Reconfigure stdout to UTF-8 with
    replacement so logging and CLI output can never crash the pipeline.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover
            pass
    logger = logging.getLogger("asx_quant")
    logger.setLevel(logging.DEBUG)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, JsonLineHandler)
        for h in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(console)

    if log_file is not None:
        has_file = any(
            isinstance(h, JsonLineHandler) and h._path == log_file
            for h in logger.handlers
        )
        if not has_file:
            file_handler = JsonLineHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter())
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced child logger of the ``asx_quant`` root logger."""
    if not name.startswith("asx_quant"):
        name = f"asx_quant.{name}"
    return logging.getLogger(name)
