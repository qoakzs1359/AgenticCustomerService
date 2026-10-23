"""日志统一配置 —— 用 loguru"""
import sys
from loguru import logger


def setup_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
    )
    return logger


log = setup_logger()
