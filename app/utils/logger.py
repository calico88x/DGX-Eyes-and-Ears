"""
app/utils/logger.py

Configures the Python logging system for the whole app.

Call setup_logging(config) once at startup in main.py, before
anything else runs. After that, every module just does:

    import logging
    logger = logging.getLogger(__name__)

and logs appear in both the console and the log file.

Log levels map to pipeline verbosity:
    INFO    — normal operation, state transitions, key events
    DEBUG   — classifier output, memory retrieval, prompt content
    WARNING — recoverable errors, fallbacks used
    ERROR   — service failures, unhandled exceptions
"""

import logging
import logging.handlers
from pathlib import Path

from app.utils.config_loader import Config


def setup_logging(config: Config) -> None:
    """
    Configure root logger from config.

    Sets up:
      - Console handler (StreamHandler) at configured level
      - Rotating file handler (10 MB max, 5 backups) at DEBUG level
        so the file always captures everything for troubleshooting

    Parameters
    ----------
    config : Config
        Loaded app config. Uses config.log_cfg.level and config.log_cfg.path.
    """
    log_cfg  = config.log_cfg
    log_path = Path(log_cfg.path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level     = getattr(logging, log_cfg.level, logging.INFO)
    fmt       = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt  = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root captures everything; handlers filter

    # ---- console handler ----
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    # ---- rotating file handler ----
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes    = 10 * 1024 * 1024,   # 10 MB
        backupCount = 5,
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Clear any existing handlers (safe to call setup_logging multiple times)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging configured | console_level=%s log_file=%s",
        log_cfg.level, log_cfg.path
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — same as logging.getLogger(name)."""
    return logging.getLogger(name)
