"""
M7 知识点提取模块
基于全文本，识别视频中的知识点并标注时间戳。
对应文档：03_接口设计/M7_知识点提取模块接口.md

M11-M17 重构：
- 依赖 LLMGenerator 协议，不依赖具体 LLMClient
- 调用 generate(prompt, payload)，不感知模型/temperature/分块
- 删除 use_cache 参数（缓存由 pipeline 层管理）

10 设计（issue #9）：新增 enrich_knowledge 深度整理
- 每知识点独立调 LLM，融合 ASR+OCR
- 输出 content（核心内容，公式 LaTeX 嵌入）+ supplement（豆包补充，高考范围）
"""

import json
import re
from typing import List, Optional

from utils.models import KnowledgePoint
from utils.timestamp import parse_timestamp
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError, EmptyResultError
from utils.llm_json import parse_llm_json
from core.llm.protocol import LLMGenerator

logger = setup_logger("M7_knowledge")


SYSTEM_PROMPT = """你是一个教学视频内容分析助手。请从以下带时间戳的教学视频全文本中，识别老师讲解的所有知识点。

【要求】
1. 识别视频中讲解的所有知识点（定义、定理、公式、方法、概念等）
2. 为每个知识点标注在视频中的起始时间戳
3. 知识点名称要极简，1-2句话概括，不做详细解释
4. 不标注重点、难点、考点
5. 不遗漏任何知识点
6. 按时间戳顺序输出
7. 输出 JSON 格式，必须紧凑（无缩进、无换行、无Markdown代码块），直接输出JSON数组

【输出格式】
[{"index":1,"name":"知识点名称","start_time":"05:23","confidence":0.9}]
"""


KNOWLEDGE_SUMMARY_PROMPT = """你是教学视频知识点深度整理助手。请综合以下 ASR 语音文字和 OCR 板书文字，整理该知识点的深度内容。

【要求】
1. 核心内容：总结老师对该知识点的讲解，重点是老师分析问题和解决问题的过程
2. 公式用 LaTeX 嵌入在讲解过程中（如 $f(x)=ax^2+bx+c$），不单独列公式段
3. 补充内容：补充老师未提及但相关的知识，范围不超出高考，标注"补充"
4. 输出 JSON，紧凑无缩进、无Markdown代码块：
{"content":"核心内容...","supplement":"补充内容..."}
"""


class KnowledgeExtractor:
    """知识点提取器"""

    def __init__(self, llm: LLMGenerator, summary_llm: Optional[LLMGenerator] = None):
        self.llm = llm
        self.summary_llm = summary_llm or llm

    def extract(self, full_text: str, video_duration: float) -> List[KnowledgePoint]:
        """从全文本中提取知识点列表

        Args:
            full_text: 带时间戳的全文本
            video_duration: 视频总时长（秒）

        Returns:
            KnowledgePoint 列表
        """
        logger.info(f"[M7] 知识点提取: 全文本{len(full_text)}字符")

        knowledge_points = self._extract_single(full_text, video_duration)

        if not knowledge_points:
            raise EmptyResultError("知识点提取结果为空")

        logger.info(f"[M7] 知识点提取完成: {len(knowledge_points)}个知识点")
        return knowledge_points

    def _extract_single(self, text: str, video_duration: float) -> List[KnowledgePoint]:
        """调用 LLM 提取知识点"""
        payload = f"【视频时长】{video_duration} 秒\n\n【视频全文本】\n{text}"

        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=payload)
        return self._parse_response(response.content)

    def enrich_knowledge(self, kp: KnowledgePoint, aligned, ocr_results: list = None) -> KnowledgePoint:
        """对单个知识点，融合 ASR+OCR 深度整理（10 设计，issue #9）

        每知识点独立调用 LLM：
        - ASR 片段：知识点时间范围内的纠错后文本切片
        - OCR 片段：该范围内所有帧的文字+公式 LaTeX（不做颜色过滤）
        - LLM 综合两者，输出 content（核心内容，公式 LaTeX 嵌入）+ supplement（豆包补充）

        Args:
            kp: 单个知识点（含 start_time/end_time/name）
            aligned: AlignedTranscript（纠错后全文 + raw 字级时间戳）
            ocr_results: 全部 OCR 帧结果，内部按知识点时间范围过滤

        Returns:
            填充了 content+supplement 的 kp
        """
        if self.summary_llm is None:
            logger.warning(f"[M7] 知识点{kp.index}未配置 LLM，跳过深度整理")
            return kp

        asr_seg = self._slice_asr_text(aligned, kp.start_time, kp.end_time)
        ocr_seg = self._slice_ocr_text(ocr_results, kp.start_time, kp.end_time)

        if not asr_seg and not ocr_seg:
            logger.warning(f"[M7] 知识点{kp.index} ASR/OCR 片段均为空，跳过深度整理")
            return kp

        payload = (
            f"【知识点名称】\n{kp.name}\n\n"
            f"【ASR 片段】（老师口头讲解）\n{asr_seg}\n\n"
            f"【OCR 板书片段】（含公式 LaTeX）\n{ocr_seg}"
        )
        response = self.summary_llm.generate(prompt=KNOWLEDGE_SUMMARY_PROMPT, payload=payload)
        content, supplement = self._parse_summary_response(response.content)
        kp.content = content
        kp.supplement = supplement
        logger.info(f"[M7] 知识点{kp.index} 深度整理: content {len(content)}字, supplement {len(supplement)}字")
        return kp

    def _slice_asr_text(self, aligned, start_time: float, end_time: float) -> str:
        """从 aligned.raw 切 [start_time, end_time] 时间范围的 ASR 文本片段"""
        if aligned is None or aligned.raw is None:
            return ""
        char_ts = aligned.raw.char_timestamps
        if not char_ts:
            return ""
        start_idx = None
        end_idx = None
        for i, ct in enumerate(char_ts):
            if ct is None:
                continue
            sec = ct.start_ms / 1000.0
            if start_idx is None and sec >= start_time:
                start_idx = i
            if sec <= end_time:
                end_idx = i
            elif start_idx is not None:
                break
        if start_idx is None:
            return ""
        if end_idx is None:
            end_idx = len(aligned.raw.text) - 1
        return aligned.raw.text[start_idx:end_idx + 1]

    def _slice_ocr_text(self, ocr_results: list, start_time: float, end_time: float) -> str:
        """从 ocr_results 过滤时间范围内的帧，拼接文字+公式 LaTeX"""
        if not ocr_results:
            return ""
        parts = []
        for frame in ocr_results:
            if frame.timestamp < start_time or frame.timestamp > end_time:
                continue
            if getattr(frame, "is_duplicate", False):
                continue
            if frame.full_text:
                parts.append(f"[{frame.timestamp:.0f}s] {frame.full_text}")
            for r in frame.results:
                if r.block_type == "formula" and r.latex:
                    parts.append(f"[公式] ${r.latex}$")
        return "\n".join(parts)

    def _parse_summary_response(self, content: str) -> tuple:
        """解析知识点深度整理 LLM 返回的 JSON"""
        try:
            data = parse_llm_json(content)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"知识点深度整理JSON解析失败: {e}, 原始内容: {content[:200]}")

        return data.get("content", ""), data.get("supplement", "")

    def _parse_response(self, content: str) -> List[KnowledgePoint]:
        """解析 LLM 返回的 JSON"""
        try:
            data = parse_llm_json(content)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"知识点提取结果JSON解析失败: {e}, 原始内容: {content[:200]}")

        results = []
        for item in data:
            try:
                kp = KnowledgePoint(
                    index=item.get("index", len(results) + 1),
                    name=item.get("name", ""),
                    start_time=parse_timestamp(item.get("start_time", "00:00")),
                    confidence=item.get("confidence", 0.8),
                )
                results.append(kp)
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                logger.warning(f"[M7] 解析知识点失败: {e}, item={item}")

        return results

    def to_markdown(self, knowledge_points: List[KnowledgePoint], timestamp_format: str = "mm:ss") -> str:
        """将知识点列表转换为 Markdown 格式"""
        from utils.timestamp import format_timestamp
        lines = ["# 知识点清单\n"]
        for kp in knowledge_points:
            ts = format_timestamp(kp.start_time, timestamp_format)
            lines.append(f"{kp.index}. {kp.name} [{ts}]")
        return "\n".join(lines)
