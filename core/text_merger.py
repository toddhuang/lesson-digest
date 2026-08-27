"""
M6 ASR 文本整理模块
从 RawTranscript 或 AlignedTranscript 中提取纯文本全文。
OCR 课件文字不混入全文本，仅作为后续题区域判断依据。
对应文档：03_接口设计/M6_文本合并模块接口.md
"""

from typing import Union

from utils.models import RawTranscript, AlignedTranscript
from utils.logger import setup_logger

logger = setup_logger("M6_merge")

# 纠错后文本或原始文本均可，两者都有 .text 属性
TranscriptType = Union[RawTranscript, AlignedTranscript]


class TextMerger:
    """ASR 文本整理器"""

    def merge(self, transcript: TranscriptType) -> str:
        """从 transcript 提取纯文本全文

        Args:
            transcript: ASR 原始结果或纠错后结果

        Returns:
            纯文本字符串（不含时间戳标记，LLM 输入用）
        """
        logger.info(f"[M6] ASR文本整理: {len(transcript.text)}字")
        return transcript.text
