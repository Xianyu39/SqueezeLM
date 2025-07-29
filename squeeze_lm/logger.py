import logging
import sys

def init_logger(
    name="squeeze_lm",
    level=logging.INFO,
    stream: str = "stdout",   # "stdout", "stderr", or a file path
    fmt: str = "[%(asctime)s] %(levelname)s - %(message)s"
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    if stream == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif stream == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(stream, encoding="utf-8")

    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
