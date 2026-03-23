import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Sets up structured logging with timestamp, logger name, level,
    and message. Outputs to stdout for Docker-friendly log collection.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level. Default is logging.INFO.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger