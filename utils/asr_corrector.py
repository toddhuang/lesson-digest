"""
ASR 纠错工具（v3）
去掉时间戳和序号，纯文本喂给 LLM，避免误导并减少 token。
返回后按行对应回原始句子的时间戳。

M11-M17 重构：
- 依赖 LLMGenerator 协议，不依赖具体 LLMClient
- 调用 generate(prompt, payload)，不感知模型/temperature/分块
- 删除 backend 参数（模型由任务配置决定）
"""

from typing import List

from utils.models import Sentence
from utils.logger import setup_logger
from core.llm.protocol import LLMGenerator

logger = setup_logger("ASR_Corrector")

SYSTEM_PROMPT = """你是一个教学视频语音识别纠错助手。请根据上下文纠正ASR识别错误。

【纠错范围】
1. 同音词错误（如"地物"→"生物"，"几何"→"集合"等）
2. 数理化专业术语错误
3. 明显语义不通顺的地方
4. 标点符号错误、重复字、漏字

【严格禁止】
1. 不要改写、总结、删减内容
2. 不要添加原文没有的信息
3. 不要合并或拆分句子（保持每行一句，行数不变）
4. 保持每句的原意不变
5. 输出时每行一句，不要添加序号、时间戳或其他标记

【输出格式】
直接输出纠错后的文本，每行一句，行数与输入完全相同。不要添加任何解释或Markdown格式。
"""


class ASRCorrector:
    """ASR 纠错器（纯文本输入，按行对应）"""

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def correct(self, sentences: List[Sentence]) -> List[Sentence]:
        """对 ASR 句子进行纠错

        Args:
            sentences: 原始 ASR 句子列表

        Returns:
            纠错后的句子列表（时间戳不变，文本可能被修改）
        """
        if not sentences:
            return []

        logger.info(f"[ASR纠错] 开始纠错: {len(sentences)}句")

        full_text = "\n".join(sent.text for sent in sentences)
        corrected_text = self._correct_single(full_text)
        corrected_lines = corrected_text.split("\n")

        result = self._align_lines(sentences, corrected_lines)

        logger.info(f"[ASR纠错] 完成: {len(result)}句")
        return result

    def _correct_single(self, text: str) -> str:
        """调用 LLM 纠错单段文本"""
        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=text)
        return response.content.strip()

    def _align_lines(self, original: List[Sentence], corrected_lines: List[str]) -> List[Sentence]:
        """把纠错后的行对应回原始句子的时间戳"""
        result = []
        cleaned_lines = [line.strip() for line in corrected_lines if line.strip()]

        if len(cleaned_lines) == len(original):
            for i, sent in enumerate(original):
                new_text = cleaned_lines[i]
                if new_text:
                    result.append(Sentence(
                        start_time=sent.start_time,
                        end_time=sent.end_time,
                        text=new_text,
                        confidence=sent.confidence,
                    ))
                else:
                    result.append(sent)
            logger.info(f"[ASR纠错] 行数匹配: {len(original)}句，一一对应")
        else:
            logger.warning(f"[ASR纠错] 行数不匹配: 原始{len(original)}句, 纠错后{len(cleaned_lines)}行，使用相似度匹配")
            result = self._match_by_similarity(original, cleaned_lines)

        return result

    def _match_by_similarity(self, original: List[Sentence], corrected: List[str]) -> List[Sentence]:
        """用文本相似度把纠错后的行匹配到原始句子"""
        result = []
        used = set()

        for i, sent in enumerate(original):
            best_idx = -1
            best_sim = 0

            for j, line in enumerate(corrected):
                if j in used:
                    continue
                sim = self._text_similarity(sent.text, line)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            if best_idx >= 0 and best_sim > 0.3:
                used.add(best_idx)
                result.append(Sentence(
                    start_time=sent.start_time,
                    end_time=sent.end_time,
                    text=corrected[best_idx],
                    confidence=sent.confidence,
                ))
            else:
                result.append(sent)

        return result

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的 Jaccard 相似度（基于字符）"""
        set1 = set(text1)
        set2 = set(text2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
