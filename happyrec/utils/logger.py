"""
This module defines the logger used in the HappyRec framework.
"""
import os
from logging import FileHandler, Logger, getLogger
from pathlib import Path

import pytorch_lightning  # noqa: F401
from rich.logging import RichHandler

_LOG_FILE = "log.txt"
"""The name of the log file."""


def init_happyrec_logger(log_dir: str | Path | None = None, mode: str = "a") -> Logger:
    """Initialize the logger used in PyTorch Lightning and the HappyRec framework.
    The logger outputs to the console and optionally to a log file.

    :param log_dir: The directory to save the log file. If ``None``, the logger
        will not save the log to a file. Default: ``None``.
    :param mode: The mode to open the log file. Default: ``"a"``.
    :return: The initialized logger.
    """
    log_path: str | None = (
        os.path.abspath(os.fspath(Path(log_dir) / _LOG_FILE)) if log_dir else None
    )
    has_rich_handler = False
    has_file_handler = False

    pl_logger = getLogger("pytorch_lightning")
    for handler in pl_logger.handlers:
        if isinstance(handler, RichHandler):
            has_rich_handler = True
        elif isinstance(handler, FileHandler) and handler.baseFilename == log_path:
            has_file_handler = True
        else:
            pl_logger.removeHandler(handler)

    if not has_rich_handler:
        rich_handler = RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)
        pl_logger.addHandler(rich_handler)

    if log_path is not None and not has_file_handler:
        file_handler = FileHandler(log_path, mode=mode)
        pl_logger.addHandler(file_handler)

    return pl_logger


logger = init_happyrec_logger()
"""The default logger used in the HappyRec framework."""
