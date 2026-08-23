"""
Token 计数工具
mock阶段使用简单估算（中文字符≈1.5 token，英文单词≈1.3 token）
真实运行时使用 tiktoken
"""


def count_tokens(text: str) -> int:
    """统计文本的 token 数（mock估算版）

    Args:
        text: 待统计文本

    Returns:
        估算的 token 数
    """
    if not text:
        return 0
    # 简单估算：中文字符约1.5 token，英文/数字约0.4 token
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_count = len(text) - chinese_count
    return int(chinese_count * 1.5 + other_count * 0.4)
