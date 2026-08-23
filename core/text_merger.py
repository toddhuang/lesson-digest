"""
M6 文本合并模块
合并 ASR 和 OCR 结果，生成带时间戳的全文本。
对应文档：03_接口设计/M6_文本合并模块接口.md
"""

from typing import List

from utils.models import Sentence, OCRFrameResult, MergedText
from utils.timestamp import format_timestamp
from utils.logger import setup_logger

logger = setup_logger("M6_merge")


class TextMerger:
    """文本合并器"""

    def merge(self, asr_results: List[Sentence], ocr_results: List[OCRFrameResult]) -> str:
        """合并 ASR 和 OCR 结果，生成带时间戳的全文本

        Args:
            asr_results: ASR 语音识别结果
            ocr_results: OCR 文字识别结果

        Returns:
            带时间戳的全文本字符串
        """
        logger.info(f"[M6] 文本合并: ASR {len(asr_results)}句, OCR {len(ocr_results)}帧")

        # 构建合并文本片段列表
        merged: List[MergedText] = []

        # ASR 结果
        for sent in asr_results:
            merged.append(MergedText(
                timestamp=sent.start_time,
                text=sent.text,
                source="asr",
                confidence=sent.confidence,
            ))

        # OCR 结果（跳过重复帧）
        for frame in ocr_results:
            if frame.is_duplicate:
                continue
            if frame.full_text.strip():
                merged.append(MergedText(
                    timestamp=frame.timestamp,
                    text=f"[课件文字] {frame.full_text}",
                    source="ocr",
                    confidence=0.9,
                ))

        # 按时间戳排序
        merged.sort(key=lambda x: x.timestamp)

        # 格式化为全文本
        lines = []
        for mt in merged:
            ts = format_timestamp(mt.timestamp)
            source_tag = "[ASR]" if mt.source == "asr" else "[OCR]"
            lines.append(f"[{ts}] {source_tag} {mt.text}")

        full_text = "\n".join(lines)
        logger.info(f"[M6] 文本合并完成: {len(merged)}片段, {len(full_text)}字符")
        return full_text
