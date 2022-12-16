import os
from logging import FileHandler, Logger, getLogger
from pathlib import Path

from rich.logging import RichHandler

_LOG_FILE = "log.txt"
"""The name of the log file."""


def init_happyrec_logger(log_dir: str | Path | None = None, mode: str = "a") -> Logger:
    log_path: str | None = (
        os.path.abspath(os.fspath(Path(log_dir) / _LOG_FILE)) if log_dir else None
    )
    has_rich_handler = False
    has_file_handler = False

    happyrec_logger = getLogger("happyrec")
    for handler in happyrec_logger.handlers:
        if isinstance(handler, RichHandler):
            has_rich_handler = True
        elif isinstance(handler, FileHandler) and handler.baseFilename == log_path:
            has_file_handler = True
        else:
            happyrec_logger.removeHandler(handler)

    if not has_rich_handler:
        rich_handler = RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)
        happyrec_logger.addHandler(rich_handler)

    if log_path is not None and not has_file_handler:
        file_handler = FileHandler(log_path, mode=mode)
        happyrec_logger.addHandler(file_handler)

    return happyrec_logger


logger = init_happyrec_logger()
"""The default logger used in the HappyRec framework."""
