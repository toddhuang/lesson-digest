"""
时间戳格式化工具
秒 ↔ mm:ss / hh:mm:ss
"""


def format_timestamp(seconds: float, fmt: str = "mm:ss") -> str:
    """将秒数格式化为时间戳字符串

    Args:
        seconds: 秒数
        fmt: 格式：
            "mm:ss" → 分:秒
            "mm:ss.cc" → 分:秒.百分秒（debug 字级时间戳用）
            "hh:mm:ss" → 时:分:秒
            "hh:mm:ss.cc" → 时:分:秒.百分秒

    Returns:
        格式化后的时间戳字符串
    """
    if seconds < 0:
        seconds = 0
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    centi = int(round((seconds - total_sec) * 100))
    if centi >= 100:
        centi = 99

    show_centi = ".cc" in fmt
    show_hours = "hh:" in fmt or hours > 0

    if show_hours:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"
    if show_centi:
        base = f"{base}.{centi:02d}"
    return base


def parse_timestamp(ts_str: str) -> float:
    """将时间戳字符串解析为秒数

    Args:
        ts_str: 时间戳字符串，支持 "mm:ss"、"hh:mm:ss"、"ss"

    Returns:
        秒数
    """
    parts = ts_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])
