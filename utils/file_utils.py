"""
文件操作工具
"""

import os
import json
import hashlib
from typing import Any


def ensure_dir(dir_path: str) -> None:
    """确保目录存在，不存在则创建

    Args:
        dir_path: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希（用于缓存键）

    Args:
        file_path: 文件路径

    Returns:
        MD5 哈希字符串
    """
    if not os.path.exists(file_path):
        return hashlib.md5(file_path.encode()).hexdigest()

    stat = os.stat(file_path)
    key = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def save_json(data: Any, file_path: str) -> None:
    """保存数据为 JSON 文件

    Args:
        data: 待保存数据
        file_path: 输出文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(file_path: str) -> Any:
    """加载 JSON 文件

    Args:
        file_path: 文件路径

    Returns:
        加载的数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, file_path: str) -> None:
    """保存文本文件

    Args:
        text: 文本内容
        file_path: 输出文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def get_dir_size_gb(dir_path: str) -> float:
    """计算目录大小（GB）

    Args:
        dir_path: 目录路径

    Returns:
        目录大小（GB）
    """
    if not os.path.exists(dir_path):
        return 0.0
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)
