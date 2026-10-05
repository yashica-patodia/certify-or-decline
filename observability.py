"""Structured operational logging with safe contextual fields."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Iterator


_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "reasoning-eval_log_context", default={}
)
_SENSITIVE_KEY = re.compile(
    r"(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|"
    r"password|secret|credential|cookie)",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
        r"(['\"]?\s*[:=]\s*['\"]?)([^\s,'\"}]{8,})['\"]?"
    ),
)


def redact_text(value: str) -> str:
    """Redact common credential forms from console-safe text previews."""
    value = _SENSITIVE_TEXT[0].sub(r"\1[REDACTED]", value)
    return _SENSITIVE_TEXT[1].sub(r"\1\2[REDACTED]", value)


def _redact(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line for ingestion by standard log systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            **_context.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = {**_context.get(), **getattr(record, "fields", {})}
        suffix = " ".join(f"{k}={v}" for k, v in sorted(_redact(context).items()))
        base = f"{record.levelname.lower()} {record.name}: {record.getMessage()}"
        if suffix:
            base += f" [{suffix}]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(
    *,
    level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure the package root logger once, replacing stale handlers."""
    resolved_level = (level or os.getenv("OWRE_LOG_LEVEL") or "INFO").upper()
    if getattr(logging, resolved_level, None) is None:
        raise ValueError(
            "log level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL; "
            f"got {resolved_level!r}"
        )
    resolved_format = (
        log_format or os.getenv("OWRE_LOG_FORMAT") or "text"
    ).lower()
    if resolved_format not in {"text", "json"}:
        raise ValueError("log format must be 'text' or 'json'")

    logger = logging.getLogger("reasoning-eval")
    logger.setLevel(resolved_level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if resolved_format == "json" else TextFormatter())
    logger.handlers[:] = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"reasoning-eval.{name}")


# ── Structured event stream ──────────────────────────────────────
#
# Console logging (above) is for humans. The events file is for
# machines: one JSON object per line, containing ONLY records emitted
# through the `event()` helper (run.started, problem.started,
# llm.call.*, run.completed, ...). Console noise — streamed [text]/
# [tool] lines, HF messages, tracebacks printed to stdout — never
# reaches it, so the file is always 100% parseable JSONL.

class _StructuredEventFilter(logging.Filter):
    """Pass only records emitted via `event()` (which sets `record.event`)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "event")


_events_handler: logging.Handler | None = None
# Snapshot of (logger.level, [(handler, handler.level), ...]) taken before
# attach mutates levels; restored by detach so console verbosity is unchanged
# for the rest of the process.
_events_level_snapshot: tuple[int, list[tuple[logging.Handler, int]]] | None = None


def attach_events_file(run_id: str, logs_dir: str = "runs/logs") -> str:
    """Route structured events to `<logs_dir>/<run_id>.events.jsonl`.

    Returns the file path. Replaces any previously attached events file
    (one run at a time per process). Console handlers are unaffected.

    ``logs_dir`` defaults to the CWD-relative ``runs/logs`` path used by
    the rest of the harness today; if run-dir relocation ever lands,
    callers should pass an absolute path derived from ``save_path``.
    """
    global _events_handler, _events_level_snapshot
    detach_events_file()
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"{run_id}.events.jsonl")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_StructuredEventFilter())
    logger = logging.getLogger("reasoning-eval")
    # Snapshot before mutating so detach can restore console verbosity.
    _events_level_snapshot = (
        logger.level,
        [(existing, existing.level) for existing in logger.handlers],
    )
    # Events must reach the file even when console verbosity is higher
    # than INFO; the console handler keeps its own threshold.
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        for existing in logger.handlers:
            if existing.level == logging.NOTSET:
                existing.setLevel(logger.getEffectiveLevel())
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    _events_handler = handler
    return path


def detach_events_file() -> None:
    """Detach and close the structured events file handler, if any.

    Restores the ``reasoning-eval`` logger level and any pre-attach handler
    levels that ``attach_events_file`` may have pinned.
    """
    global _events_handler, _events_level_snapshot
    logger = logging.getLogger("reasoning-eval")
    if _events_handler is not None:
        logger.removeHandler(_events_handler)
        _events_handler.close()
        _events_handler = None
    if _events_level_snapshot is not None:
        logger_level, handler_levels = _events_level_snapshot
        logger.setLevel(logger_level)
        for existing, level in handler_levels:
            if existing in logger.handlers:
                existing.setLevel(level)
        _events_level_snapshot = None


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind correlation fields across nested async tasks."""
    merged = {**_context.get(), **{k: v for k, v in fields.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def event(
    logger: logging.Logger,
    level: int,
    event_name: str,
    message: str,
    **fields: Any,
) -> None:
    logger.log(level, message, extra={"event": event_name, "fields": fields})
