import logging
import sys

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Keep track of configured loggers to avoid adding multiple handlers
_configured_loggers = {}


def get_logger(
    name: str, level: int = DEFAULT_LOG_LEVEL, log_format: str = DEFAULT_LOG_FORMAT
) -> logging.Logger:
    """
    Configures and returns a logger.
    Avoids adding multiple handlers if logger is already configured.

    Args:
        name (str): The name of the logger (typically __name__).
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): The format string for log messages.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = (
        False  # Prevent log messages from being passed to the root logger's handlers
    )

    # Add a stream handler if no handlers are present for this specific logger
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)  # Log to stdout
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _configured_loggers[name] = logger
    return logger


if __name__ == "__main__":
    # Example usage:
    logger1 = get_logger("my_module_1", level=logging.DEBUG)
    logger2 = get_logger("my_module_2", level=logging.INFO)
    logger_default = get_logger("my_module_default")  # Uses default level

    logger1.debug("This is a debug message from module 1.")
    logger1.info("This is an info message from module 1.")
    logger2.info("This is an info message from module 2.")
    logger2.warning("This is a warning from module 2.")
    logger_default.info("Info from default logger.")
    logger_default.error("Error from default logger.")
