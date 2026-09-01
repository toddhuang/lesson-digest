r"""LLM JSON 解析容错工具

处理 LLM 返回 JSON 时的常见问题：
1. Markdown 代码块包裹（```json...```）
2. 非法反斜杠转义（LaTeX 公式 \sqrt、\frac 等未转义为 \\sqrt、\\frac）

被 core/content_extractor、knowledge_extractor、problem_extractor 的 _parse_* 方法共用。
"""

import json
import re
from typing import Any


def parse_llm_json(content: str) -> Any:
    r"""容错解析 LLM 返回的 JSON

    三级修复策略，逐级激进：
    1. 直接 json.loads
    2. 修复非法 \\escape（\\X 中 X 非 JSON 合法转义字符：\" \\\\ \\/ \\b \\f \\n \\r \\t \\u）
    3. 把所有 \\字母 都转义（处理 \\frac 被解析为 \\f+rac、\\times 被解析为 \\t+imes 等语义错误）

    Args:
        content: LLM 返回的原始文本

    Returns:
        解析后的 Python 对象（dict 或 list）

    Raises:
        json.JSONDecodeError: 三级修复均失败
    """
    clean = _strip_code_blocks(content)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'\\(?=[^"\\/bfnrtu])', r'\\\\', clean)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    fixed2 = re.sub(r'\\(?=[a-zA-Z])', r'\\\\', clean)
    fixed2 = re.sub(r'\\\\u([0-9a-fA-F]{4})', r'\\u\1', fixed2)
    return json.loads(fixed2)


def _strip_code_blocks(content: str) -> str:
    """去除 Markdown 代码块包裹"""
    clean = re.sub(r'```json\s*', '', content)
    clean = re.sub(r'```\s*', '', clean)
    return clean.strip()
