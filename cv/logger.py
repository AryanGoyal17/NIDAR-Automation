"""
cv/logger.py - Reusable Logging Helper for NIDAR CV Pipeline

Provides a pre-configured logger that writes to both the console and a
log file. All log entries use ISO 8601 UTC timestamps for consistency
with the project conventions (see docs/conventions.md).

Usage:
    from cv.logger import get_logger

    log = get_logger("detector")
    log.info("Model loaded: yolov8n.pt")
    log.warning("Low confidence detection: 0.31")
    log.debug("Frame 42 processed in 12.3 ms")

Log format:
    2026-10-03T15:04:00Z | INFO     | detector | Model loaded: yolov8n.pt
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class UTCFormatter(logging.Formatter):
    """
    Custom formatter that outputs timestamps in ISO 8601 UTC format.

    Why UTC instead of local time?
    During the competition, your drone's companion computer (Linux) and
    your ground station laptop (Windows) might have different timezone
    settings. UTC removes all ambiguity. When the autonomy lead's SLAM
    log says an event happened at 14:32:01Z and your detection log says
    a survivor was found at 14:32:01Z, you know they are the same instant.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z"


# Module-level registry to avoid duplicate handlers when get_logger is
# called multiple times with the same name.
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: str | Path | None = "outputs/nidar_cv.log",
) -> logging.Logger:
    """
    Create or retrieve a named logger with console and optional file output.

    Args:
        name:     Logger name (e.g. "detector", "gcs", "config").
                  Use short, lowercase names. Each module should have its own.
        level:    Minimum log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_file: Path for the log file. Set to None for console-only output.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(f"nidar.{name}")

    # Avoid adding duplicate handlers if this logger was already configured
    if name in _configured_loggers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Log format: timestamp | level | module name | message
    fmt = UTCFormatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Console handler (writes to stdout so it is visible in the terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (appends to a persistent log file for post-run analysis)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger


# ---------------------------------------------------------------------------
# Quick self-test: run this file directly to see the logger in action.
# Usage: .venv/Scripts/python.exe -m cv.logger
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("test", level="DEBUG")

    log.debug("This is a DEBUG message (detailed tracing)")
    log.info("This is an INFO message (normal operation)")
    log.warning("This is a WARNING message (something unexpected)")
    log.error("This is an ERROR message (something failed)")
    log.critical("This is a CRITICAL message (system cannot continue)")

    print()
    print("Logger self-test complete. Check outputs/nidar_cv.log for file output.")
