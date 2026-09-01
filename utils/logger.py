"""
日志配置工具。

设计（issue #11 第 9 类 debug 产物）：
- console handler：每个 logger 实例独立（模块名标识）
- file handler：所有 logger 共享 root logger 上的一个全局 file handler
  - 首次调 setup_logger 时自动初始化，默认指向 logs/{timestamp}.log（release fallback）
  - debugger.attach_log_handler() 调 set_log_file() 切换到 debug/{视频名}/09_运行日志/pipeline.log
  - 切换后所有 logger 自动跟随，写入单一文件便于归档
- propagate=True：子 logger 的 log 传到 root，由 root 的 file handler 统一写文件
"""

import logging
import os
from datetime import datetime
from typing import Optional


# 全局共享 file handler（所有 logger 通过 root logger 共用）
_global_file_handler: Optional[logging.FileHandler] = None
_default_log_dir = "./logs"

_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def _create_file_handler(path: str) -> Optional[logging.FileHandler]:
    """创建 file handler，失败返回 None（不影响 console 输出）"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_FORMATTER)
        return handler
    except (OSError, PermissionError):
        return None


def _ensure_global_file_handler(log_dir: str = _default_log_dir) -> Optional[logging.FileHandler]:
    """首次调用时创建全局 file handler，指向 logs/{timestamp}.log"""
    global _global_file_handler
    if _global_file_handler is not None:
        return _global_file_handler
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _global_file_handler = _create_file_handler(log_file)
    if _global_file_handler is not None:
        logging.getLogger().addHandler(_global_file_handler)
    return _global_file_handler


def set_log_file(path: str) -> bool:
    """切换全局 file handler 到新路径

    用于 debugger.attach_log_handler()：把所有 logger 的 file 输出
    从 logs/{timestamp}.log 切换到 debug/{视频名}/09_运行日志/pipeline.log。

    Args:
        path: 新日志文件路径

    Returns:
        True 表示切换成功；False 表示文件创建失败（保持原 handler）
    """
    global _global_file_handler
    root = logging.getLogger()
    old_handler = _global_file_handler
    new_handler = _create_file_handler(path)
    if new_handler is None:
        return False
    if old_handler is not None:
        root.removeHandler(old_handler)
        try:
            old_handler.close()
        except Exception:
            pass
    root.addHandler(new_handler)
    _global_file_handler = new_handler
    return True


def setup_logger(name: str = "videocontents", level: int = logging.INFO) -> logging.Logger:
    """配置并返回日志记录器

    Args:
        name: 日志记录器名称（通常为模块名，如 "ContentExtractor"）
        level: console handler 的日志级别

    Returns:
        配置好的 Logger 对象

    说明：
    - console handler 独立创建，便于按模块过滤
    - file handler 共享 root logger 上的全局 file handler，由 set_log_file 切换路径
    - 所有 logger 的 log 通过 propagate=True 传到 root，由 root 的 file handler 统一写文件
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = True  # 关键：log 传到 root，由 root 的全局 file handler 统一写文件

    # 控制台 handler（每个 logger 独立）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    # 全局 file handler（首次调用时初始化，默认指向 logs/）
    _ensure_global_file_handler()

    return logger
