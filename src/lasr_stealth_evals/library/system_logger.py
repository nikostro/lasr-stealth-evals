import os
import logging


LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def get_logger(name: str) -> logging.Logger:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")

    # Add formatter to console_handler
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.ERROR,  # Set default log level to debug
        format="%(message)s",
        datefmt="[%X]",
        handlers=[console_handler],
    )

    return logging.getLogger(name)
