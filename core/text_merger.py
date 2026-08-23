"""
M6 ASR 文本整理模块
将 ASR 语音识别结果整理为带时间戳的全文本。
注意：OCR 课件文字不混入全文本，仅作为 M9 题目截图的区域判断依据。
对应文档：03_接口设计/M6_文本合并模块接口.md
"""

from typing import List

from utils.models import Sentence
from utils.timestamp import format_timestamp
from utils.logger import setup_logger

logger = setup_logger("M6_merge")


class TextMerger:
    """ASR 文本整理器"""

    def merge(self, asr_results: List[Sentence]) -> str:
        """将 ASR 结果整理为带时间戳的全文本

        Args:
            asr_results: ASR 语音识别结果

        Returns:
            带时间戳的全文本字符串
        """
        logger.info(f"[M6] ASR文本整理: {len(asr_results)}句")

        # 按时间戳排序（ASR结果本身应该已经有序，但保险起见）
        sorted_results = sorted(asr_results, key=lambda x: x.start_time)

        # 格式化为全文本
        lines = []
        for sent in sorted_results:
            ts = format_timestamp(sent.start_time)
            lines.append(f"[{ts}] {sent.text}")

        full_text = "\n".join(lines)
        logger.info(f"[M6] ASR文本整理完成: {len(sorted_results)}句, {len(full_text)}字符")
        return full_text
