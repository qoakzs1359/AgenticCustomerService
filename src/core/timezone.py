"""统一时间工具 —— 显式北京时间 (UTC+8)，不依赖系统时区"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

BEIJING_TZ = timezone(timedelta(hours=8))


def now_beijing() -> datetime:
    """返回带时区的当前北京时间 (aware datetime)"""
    return datetime.now(BEIJING_TZ)


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回北京时间字符串，默认格式 '2026-10-27 15:13:02'"""
    return datetime.now(BEIJING_TZ).strftime(fmt)
