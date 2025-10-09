"""Logging utilities for RAGvix."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a Rich-formatted logger.
    
    Args:
        name: Logger name, defaults to None
        
    Returns:
        Configured logger with Rich formatting
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        console = Console()
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(
            logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger