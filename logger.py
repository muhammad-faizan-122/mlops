import logging
import os


def setup_logger(
    name: str = "cifar_api",
    log_file: str = "inference.log",
    logs_dir: str = "logs",
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(os.path.join(logs_dir, log_file))
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
