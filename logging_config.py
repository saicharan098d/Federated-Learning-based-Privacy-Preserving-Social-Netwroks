"""
logging_config.py
─────────────────
Centralized logging configuration for FL-PPSN project.

Usage:
    from logging_config import setup_logging
    
    logger = setup_logging(__name__)
    logger.info("Starting federated training...")
"""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: str = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logging for a module with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional log file path
        log_dir: Directory for log files (default: "logs")
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create logs directory if it doesn't exist
    if log_file or log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler (optional)
    if log_file is None:
        log_file = os.path.join(log_dir, f"{name.split('.')[-1]}.log")
    
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    if log_file:
        file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# Default logger for main module
logger = setup_logging(
    __name__,
    level=logging.INFO,
    log_dir="logs"
)


if __name__ == "__main__":
    # Test logging configuration
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print("\nLog file created at: logs/logging_config.log")
