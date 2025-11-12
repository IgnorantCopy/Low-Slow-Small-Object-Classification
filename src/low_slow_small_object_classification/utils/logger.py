from loguru import logger


def get_logger(filename: str):
    logger.add(filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", rotation="10 MB")
    return logger


default_logger = logger