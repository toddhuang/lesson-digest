"""
ASR 纠错 + 知识点段 + 题目段 一次性 LLM 调用提取模块。

AGENTS.md 约定：一次 LLM 调用返回三样东西——
  纠错全文 + 知识点文字段（原文截取）+ 题目文字段（原文截取，不区分题目和讲解）。

输出：
  - AlignedTranscript（纠错后文本 + 字级时间戳对齐映射）
  - List[KnowledgePoint]（知识点列表，通过文字段定位时间戳）
  - List[Problem]（题目列表，通过文字段定位时间戳，question_text 暂为文字段）

后续阶段：
  - knowledge_summary（知识点深度整理，另一次 LLM 调用）
  - problem_extraction（题目原题提取，基于题目段+OCR，另一次 LLM 调用）
  - solution_summary（解题过程整理，基于题目段+ASR+OCR，另一次 LLM 调用）
  - mindmap_generation（思维导图，基于知识点列表，另一次 LLM 调用）
"""

import difflib
import json
import re
from typing import List, Optional, Tuple

from utils.models import (
    RawTranscript, AlignedTranscript,
    KnowledgePoint, Problem,
)
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError, EmptyResultError
from core.llm.protocol import LLMGenerator

logger = setup_logger("ContentExtractor")


SYSTEM_PROMPT = """你是一个教学视频语音识别纠错与内容分析助手。请对以下 ASR 原始文本执行三件事，一次完成：

【任务1：纠错全文】
纠正同音词错误、数理化专业术语错误、标点错误、重复字、漏字。
严格禁止改写、总结、删减、添加原文没有的信息、合并或拆分段落。
保持原文原意和语序不变。

【任务2：知识点文字段】
从纠错后全文中，识别老师讲解的所有知识点（定义、定理、公式、方法、概念等）。
为每个知识点截取原文中对应的文字段（原文截取，不要改写）。
文字段应覆盖该知识点从引入到讲完的完整内容。

【任务3：题目文字段】
从纠错后全文中，识别老师讲解的所有题目（例题、习题、考试题等）。
为每道题截取原文中对应的文字段（原文截取，不区分题目和讲解，整段截取）。
文字段应覆盖从老师开始讲到这道题到讲完这道题的完整内容。

【输出格式】
输出紧凑 JSON（无缩进、无换行、无 Markdown 代码块），结构如下：
{"corrected_text":"纠错后的完整文本","knowledge_segments":[{"name":"知识点名称","segment":"原文截取的文字段"}],"problem_segments":[{"segment":"原文截取的文字段"}]}

注意：
1. corrected_text 是纠错后的完整文本，不是片段
2. segment 必须是 corrected_text 中的连续片段（原文截取），不要改写
3. 知识点和题目按时间顺序排列
4. 不遗漏任何知识点或题目
"""


class ContentExtractor:
    """一次性 LLM 调用：纠错 + 知识点段 + 题目段提取

    AGENTS.md 约定：一次 LLM 调用返回三样东西，不做三次独立调用。
    """

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def extract(
        self,
        transcript: RawTranscript,
    ) -> Tuple[AlignedTranscript, List[KnowledgePoint], List[Problem]]:
        """一次 LLM 调用完成纠错 + 知识点段 + 题目段提取

        Args:
            transcript: ASR 原始输出（含字级时间戳）

        Returns:
            (AlignedTranscript, List[KnowledgePoint], List[Problem]) 三元组
        """
        if not transcript.text:
            return AlignedTranscript(text="", raw_align=[], raw=transcript), [], []

        logger.info(f"[ContentExtractor] 开始提取: 原文{len(transcript.text)}字")

        raw_response = self._call_llm(transcript.text)
        corrected_text, knowledge_segments, problem_segments = self._parse_response(raw_response)

        # 纠错对齐：difflib 把 corrected_text 对齐到 RawTranscript
        aligned = self._align_transcript(transcript, corrected_text)
        logger.info(f"[ContentExtractor] 纠错完成: {len(aligned.text)}字")

        # 知识点定位：在 aligned.text 中定位 segment，通过 raw_align 回溯时间戳
        knowledge_points = self._locate_knowledge_points(knowledge_segments, aligned)
        logger.info(f"[ContentExtractor] 知识点段定位: {len(knowledge_points)}个")

        # 题目定位
        problems = self._locate_problems(problem_segments, aligned)
        logger.info(f"[ContentExtractor] 题目段定位: {len(problems)}道")

        return aligned, knowledge_points, problems

    def _call_llm(self, text: str) -> str:
        """调用 LLM 一次性完成纠错 + 知识点段 + 题目段提取"""
        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=text)
        return response.content.strip()

    def _parse_response(self, content: str) -> Tuple[str, List[dict], List[dict]]:
        """解析 LLM 返回的 JSON

        Returns:
            (corrected_text, knowledge_segments, problem_segments)
            knowledge_segments: [{"name": str, "segment": str}, ...]
            problem_segments: [{"segment": str}, ...]
        """
        clean = re.sub(r'```json\s*', '', content)
        clean = re.sub(r'```\s*', '', clean)
        clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                f"ContentExtractor JSON 解析失败: {e}, 原始内容: {content[:200]}"
            )

        corrected_text = data.get("corrected_text", "")
        if not corrected_text:
            raise EmptyResultError("ContentExtractor: corrected_text 为空")

        knowledge_segments = data.get("knowledge_segments", [])
        problem_segments = data.get("problem_segments", [])

        logger.info(
            f"[ContentExtractor] 解析: 纠错{len(corrected_text)}字, "
            f"知识点段{len(knowledge_segments)}个, 题目段{len(problem_segments)}个"
        )
        return corrected_text, knowledge_segments, problem_segments

    def _align_transcript(
        self,
        raw: RawTranscript,
        corrected_text: str,
    ) -> AlignedTranscript:
        """用 difflib 将纠错后文本对齐到原始文本，生成 raw_align 映射

        复用 ASRCorrector 的对齐逻辑：
        - equal: 直接继承索引
        - replace: B 的字逐个继承 A 对应位置的时间戳
        - insert: B 新增字，raw_align=None
        - delete: A 有 B 没有，不映射
        """
        matcher = difflib.SequenceMatcher(None, raw.text, corrected_text)
        raw_align: List[Optional[int]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(j1, j2):
                    raw_align.append(i1 + (k - j1))
            elif tag == "replace":
                a_len = i2 - i1
                for k in range(j1, j2):
                    offset = min(k - j1, a_len - 1)
                    raw_align.append(i1 + offset)
            elif tag == "insert":
                for _ in range(j1, j2):
                    raw_align.append(None)

        return AlignedTranscript(
            text=corrected_text,
            raw_align=raw_align,
            raw=raw,
        )

    def _locate_segment(
        self,
        segment_text: str,
        aligned: AlignedTranscript,
        search_start: int = 0,
    ) -> Tuple[float, float, int, int]:
        """在 aligned.text 中定位 segment_text 的起止位置和时间戳

        基础版定位算法：用 difflib.find_longest_match 在 aligned.text 中查找
        segment_text 的最佳匹配位置。后续可按 08_文字对比定位核心算法.md
        优化为 n-gram 粗定位 + 编辑距离精定位 + 降阈值阶梯策略。

        Args:
            segment_text: 待定位的文字段
            aligned: 纠错后对齐的文本
            search_start: 搜索起始位置（用于顺序约束）

        Returns:
            (start_time, end_time, start_idx, end_idx)
            定位失败返回 (0.0, 0.0, -1, -1)
        """
        if not segment_text or len(segment_text) < 10:
            logger.warning(f"[ContentExtractor] 文字段过短({len(segment_text)}字)，跳过定位")
            return 0.0, 0.0, -1, -1

        # 用 difflib 在 aligned.text 中查找最佳匹配
        matcher = difflib.SequenceMatcher(None, segment_text, aligned.text[search_start:])
        match = matcher.find_longest_match(0, len(segment_text), 0, len(aligned.text) - search_start)

        if match.size == 0:
            logger.warning(f"[ContentExtractor] 文字段无法定位（匹配长度=0）")
            return 0.0, 0.0, -1, -1

        # 匹配置信度
        confidence = match.size / len(segment_text)
        if confidence < 0.3:
            logger.warning(
                f"[ContentExtractor] 文字段定位置信度低({confidence:.2f})，"
                f"匹配{match.size}/{len(segment_text)}字"
            )

        start_idx = search_start + match.b
        end_idx = start_idx + len(segment_text)
        end_idx = min(end_idx, len(aligned.text))

        # 通过 aligned.get_time_range 回溯时间戳
        start_time, end_time = aligned.get_time_range(start_idx, end_idx)

        logger.debug(
            f"[ContentExtractor] 定位: idx[{start_idx}:{end_idx}], "
            f"time[{start_time:.2f}:{end_time:.2f}], confidence={confidence:.2f}"
        )
        return start_time, end_time, start_idx, end_idx

    def _locate_knowledge_points(
        self,
        knowledge_segments: List[dict],
        aligned: AlignedTranscript,
    ) -> List[KnowledgePoint]:
        """定位知识点文字段，生成 KnowledgePoint 列表"""
        results: List[KnowledgePoint] = []
        search_start = 0

        for i, seg in enumerate(knowledge_segments):
            name = seg.get("name", f"知识点{i+1}")
            segment_text = seg.get("segment", "")

            start_time, end_time, start_idx, end_idx = self._locate_segment(
                segment_text, aligned, search_start
            )

            if start_idx < 0:
                logger.warning(f"[ContentExtractor] 知识点{i+1} '{name}' 定位失败，跳过")
                continue

            # 顺序约束：下一个知识点从此知识点结束位置之后开始搜索
            search_start = max(end_idx, start_idx + 1)

            results.append(KnowledgePoint(
                index=len(results) + 1,
                name=name,
                start_time=start_time,
                confidence=0.8,
            ))

        return results

    def _locate_problems(
        self,
        problem_segments: List[dict],
        aligned: AlignedTranscript,
    ) -> List[Problem]:
        """定位题目文字段，生成 Problem 列表

        注意：此处 Problem.question_text 暂存文字段原文，
        后续 problem_extraction 阶段会用 OCR 补充/替换为原题。
        """
        results: List[Problem] = []
        search_start = 0

        for i, seg in enumerate(problem_segments):
            segment_text = seg.get("segment", "")

            start_time, end_time, start_idx, end_idx = self._locate_segment(
                segment_text, aligned, search_start
            )

            if start_idx < 0:
                logger.warning(f"[ContentExtractor] 题目{i+1} 定位失败，跳过")
                continue

            search_start = max(end_idx, start_idx + 1)

            results.append(Problem(
                index=len(results) + 1,
                start_time=start_time,
                end_time=end_time,
                question_text=segment_text,
                asr_question_text=segment_text,
                confidence=0.8,
            ))

        return results
