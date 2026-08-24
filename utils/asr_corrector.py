"""
ASR 纠错工具（v3）
去掉时间戳和序号，纯文本喂给 LLM，避免误导并减少 token。
返回后按行对应回原始句子的时间戳。
支持 DeepSeek 和豆包（volcengine）两个云端后端对比。
"""

import re
from typing import List

from utils.models import Sentence
from utils.logger import setup_logger
from utils.exceptions import LLMError

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
    """ASR 纠错器（v3：纯文本输入，按行对应，支持多服务商）"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def correct(self, sentences: List[Sentence], backend: str = "cloud") -> List[Sentence]:
        """对 ASR 句子进行纠错

        Args:
            sentences: 原始 ASR 句子列表
            backend: LLM 后端（"cloud"=默认服务商, "deepseek"=DeepSeek, "volcengine"=豆包）

        Returns:
            纠错后的句子列表（时间戳不变，文本可能被修改）
        """
        if not sentences:
            return []

        logger.info(f"[ASR纠错] 开始纠错: {len(sentences)}句, 后端={backend}")

        # 拼接纯文本（每行一句，不带时间戳和序号）
        full_text = "\n".join(sent.text for sent in sentences)

        # 云端服务商（DeepSeek/豆包）上下文大，一次喂完整文本
        corrected_text = self._correct_single(full_text, backend)
        corrected_lines = corrected_text.split("\n")

        # 按行对应回原始句子的时间戳
        result = self._align_lines(sentences, corrected_lines)

        logger.info(f"[ASR纠错] 完成: {len(result)}句")
        return result

    def _correct_single(self, text: str, backend: str) -> str:
        """调用 LLM 纠错单段文本

        Args:
            text: 待纠错文本（每行一句）
            backend: LLM 后端

        Returns:
            纠错后的文本
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            response = self.llm_client.chat(
                messages=messages,
                backend=backend,
                temperature=0.1,
                max_tokens=8000,
            )
            return response.content.strip()
        except LLMError as e:
            logger.error(f"[ASR纠错] LLM调用失败 ({type(e).__name__}): {e}，使用原文")
            return text

    def _align_lines(self, original: List[Sentence], corrected_lines: List[str]) -> List[Sentence]:
        """把纠错后的行对应回原始句子的时间戳

        Args:
            original: 原始句子列表
            corrected_lines: 纠错后的文本行列表

        Returns:
            对应后的句子列表
        """
        result = []
        cleaned_lines = [line.strip() for line in corrected_lines if line.strip()]

        if len(cleaned_lines) == len(original):
            # 行数完全匹配，一一对应
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
            # 行数不匹配，用文本相似度匹配
            logger.warning(f"[ASR纠错] 行数不匹配: 原始{len(original)}句, 纠错后{len(cleaned_lines)}行，使用相似度匹配")
            result = self._match_by_similarity(original, cleaned_lines)

        return result

    def _match_by_similarity(self, original: List[Sentence], corrected: List[str]) -> List[Sentence]:
        """用文本相似度把纠错后的行匹配到原始句子

        简单策略：按顺序贪心匹配，相似度最高的对应。
        """
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
                # 匹配不上，保留原文
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
