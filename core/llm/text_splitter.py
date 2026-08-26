"""
递归分隔符文本分块器
当 payload 超过模型上下文限制时，按优先级尝试不同分隔符切分文本。

设计依据：R-003/R-004 调研结论
- 不使用 overlap（教学视频文本按段落/句子切分，每块语义完整）
- 不使用 LangChain，自实现约 100 行
- token 计数使用 utils.token_counter（中文估算）
"""

from typing import List

from utils.token_counter import count_tokens


class RecursiveTextSplitter:
    """递归分隔符文本分块器

    按优先级依次尝试分隔符：
    1. \\n\\n  段落
    2. \\n    行
    3. 。！？ 中文句末标点
    4. .!?   英文句末标点（带空格，避免误切小数点）
    5. ；;   分号
    6. ，,   逗号
    7. 空格  单词
    8. 强制逐字符切分
    """

    DEFAULT_SEPARATORS: List[str] = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ". ",
        "! ",
        "? ",
        "；",
        "; ",
        "，",
        ", ",
        " ",
        "",
    ]

    def __init__(self, separators: List[str] = None):
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def split(self, text: str, max_tokens: int) -> List[str]:
        """将文本切分为不超过 max_tokens 的块

        Args:
            text: 待切分文本
            max_tokens: 每块最大 token 数

        Returns:
            切分后的文本块列表
        """
        if count_tokens(text) <= max_tokens:
            return [text]
        return self._split_recursive(text, self.separators, max_tokens)

    def _split_recursive(
        self, text: str, separators: List[str], max_tokens: int
    ) -> List[str]:
        """递归切分文本"""
        if count_tokens(text) <= max_tokens:
            return [text]

        if not separators:
            return self._force_split(text, max_tokens)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._force_split(text, max_tokens)

        # 按分隔符切分，保留分隔符在前一块末尾
        pieces = self._split_with_separator(text, separator)

        chunks: List[str] = []
        current_chunk = ""

        for piece in pieces:
            candidate = current_chunk + piece
            if count_tokens(candidate) <= max_tokens:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if count_tokens(piece) <= max_tokens:
                    current_chunk = piece
                else:
                    # 单个 piece 仍然超限，用下一级分隔符递归切分
                    sub_chunks = self._split_recursive(
                        piece, remaining_separators, max_tokens
                    )
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1]
                    else:
                        current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """按分隔符切分文本，保留分隔符在每块末尾"""
        if separator == " ":
            parts = text.split(" ")
            result = [p + " " for p in parts[:-1]]
            if parts:
                result.append(parts[-1])
            return result if result != [""] else []

        parts = text.split(separator)
        result = [p + separator for p in parts[:-1]]
        if parts:
            result.append(parts[-1])
        return result if result != [""] else []

    def _force_split(self, text: str, max_tokens: int) -> List[str]:
        """强制逐字符切分（最后手段）"""
        chunks: List[str] = []
        current = ""
        for char in text:
            if count_tokens(current + char) > max_tokens:
                if current:
                    chunks.append(current)
                current = char
            else:
                current += char
        if current:
            chunks.append(current)
        return chunks
