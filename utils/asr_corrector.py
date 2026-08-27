"""
ASR 纠错工具
将 ASR 原始全文（无时间戳）交给 LLM 纠错，再用 difflib 序列对齐
将纠错后文本映射回原始字级时间戳，生成 AlignedTranscript。

设计依据：Document/调研报告/R-006 第 5.3-5.4 节
- LLM 只看纯文本，不看时间戳，避免时间戳干扰语义
- 纠错后用 difflib.SequenceMatcher（Myers diff 等价输出）做字符对齐
- 替换的字继承原字时间戳（语音没变，只是文字修正）
- 新增的标点/连词 raw_align=None，取时间时跳过
"""

import difflib
from typing import Optional

from utils.models import RawTranscript, AlignedTranscript
from utils.logger import setup_logger
from core.llm.protocol import LLMGenerator

logger = setup_logger("ASR_Corrector")

SYSTEM_PROMPT = """你是一个教学视频语音识别纠错助手。请根据上下文纠正ASR识别错误。

【纠错范围】
1. 同音词错误（如"地物"→"生物"，"几何"→"集合"等）
2. 数理化专业术语错误（如"派"→"π"）
3. 明显语义不通顺的地方
4. 标点符号错误、重复字、漏字

【严格禁止】
1. 不要改写、总结、删减内容
2. 不要添加原文没有的信息
3. 不要合并或拆分段落
4. 保持原文的原意和语序不变

【输出格式】
直接输出纠错后的完整文本，不要添加任何解释、序号或Markdown格式。
"""


class ASRCorrector:
    """ASR 纠错器（全文输入，difflib 对齐时间戳）"""

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def correct(self, transcript: RawTranscript) -> AlignedTranscript:
        """对 ASR 原始文本进行纠错并对齐时间戳

        Args:
            transcript: ASR 原始输出（含字级时间戳）

        Returns:
            AlignedTranscript，包含纠错后文本、与原始文本的对齐映射、
            以及对原始 RawTranscript 的引用
        """
        if not transcript.text:
            return AlignedTranscript(text="", raw_align=[], raw=transcript)

        logger.info(f"[ASR纠错] 开始纠错: {len(transcript.text)}字")

        corrected_text = self._call_llm(transcript.text)
        aligned = self._align(transcript, corrected_text)

        logger.info(f"[ASR纠错] 完成: {len(aligned.text)}字")
        return aligned

    def _call_llm(self, text: str) -> str:
        """调用 LLM 纠错全文"""
        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=text)
        return response.content.strip()

    def _align(
        self,
        raw: RawTranscript,
        corrected_text: str,
    ) -> AlignedTranscript:
        """用 difflib 将纠错后文本对齐到原始文本，生成 raw_align 映射

        difflib.SequenceMatcher.get_opcodes() 返回 equal/replace/insert/delete：
        - equal:  B[j1:j2] ↔ A[i1:i2]，直接继承索引
        - replace: B[j1:j2] 替换 A[i1:i2]，B 的字逐个继承 A 对应位置的时间戳
        - insert: B[j1:j2] 是新增内容，raw_align=None
        - delete: A[i1:i2] 在 B 中被删除，不映射
        """
        matcher = difflib.SequenceMatcher(None, raw.text, corrected_text)
        raw_align: list[Optional[int]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(j1, j2):
                    raw_align.append(i1 + (k - j1))
            elif tag == "replace":
                # B 的字逐个对齐到 A 的字；若长度不同，多余的 B 字对齐到 A 最后一个位置
                a_len = i2 - i1
                for k in range(j1, j2):
                    offset = min(k - j1, a_len - 1)
                    raw_align.append(i1 + offset)
            elif tag == "insert":
                for _ in range(j1, j2):
                    raw_align.append(None)
            # delete: A 有 B 没有，不需要添加映射

        return AlignedTranscript(
            text=corrected_text,
            raw_align=raw_align,
            raw=raw,
        )
