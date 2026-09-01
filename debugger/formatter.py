"""
Debug 产物格式化辅助。

将业务数据结构（RawTranscript / AlignedTranscript / KnowledgePoint / Problem）
格式化为 debug 文件所需的可读文本或 JSON 字典。
对应文档：11_debug模块设计.md §三、§四
"""

from typing import Any, List, Optional

from utils.models import (
    RawTranscript, AlignedTranscript, KnowledgePoint, Problem,
)
from utils.timestamp import format_timestamp


class DebugFormatter:
    """debug 产物格式化器（纯函数，无副作用）"""

    @staticmethod
    def asr_raw_to_json(transcript: RawTranscript) -> dict:
        """1.1 ASR 原始逐字稿 json 格式"""
        return transcript.to_dict()

    @staticmethod
    def asr_raw_to_readable(transcript: RawTranscript) -> str:
        """1.2 ASR 原始逐字稿人读 txt 格式

        每字一行：[mm:ss.cc] 字；无时间戳字（标点等）显示 [——:——.——] (None)
        """
        lines = [f"# ASR 原始逐字稿（字数 {len(transcript.text)}）", ""]
        for i, ch in enumerate(transcript.text):
            ct = transcript.char_timestamps[i] if i < len(transcript.char_timestamps) else None
            if ct is None:
                lines.append("[——:——.——] (None) " + ch)
            else:
                start = format_timestamp(ct.start_ms / 1000.0, "mm:ss.cc")
                end = format_timestamp(ct.end_ms / 1000.0, "mm:ss.cc")
                lines.append(f"[{start}-{end}] {ch}")
        return "\n".join(lines)

    @staticmethod
    def corrected_to_json(aligned: AlignedTranscript) -> dict:
        """2.1 纠错后全文 json 格式"""
        return {
            "text": aligned.text,
            "raw_align": aligned.raw_align,
        }

    @staticmethod
    def corrected_to_readable(aligned: AlignedTranscript) -> str:
        """2.2 纠错后全文人读 txt 格式（纯文本）"""
        return f"# 纠错后全文（字数 {len(aligned.text)}）\n\n{aligned.text}"

    @staticmethod
    def knowledge_segment_to_text(kp: KnowledgePoint) -> str:
        """3. 知识点文字段 txt 格式"""
        start = format_timestamp(kp.start_time, "mm:ss.cc")
        end = format_timestamp(kp.end_time, "mm:ss.cc")
        lines = [
            f"# 知识点{kp.index:02d}: {kp.name}",
            f"> 时间: [{start} - {end}]",
            f"> 置信度: {kp.confidence:.2f}",
            "",
        ]
        if kp.content:
            lines.append("## 核心内容")
            lines.append(kp.content)
            lines.append("")
        if kp.supplement:
            lines.append("## 补充内容")
            lines.append(kp.supplement)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def problem_segment_to_text(problem: Problem) -> str:
        """4. 题目文字段 txt 格式（用 asr_question_text，原始文字段）"""
        start = format_timestamp(problem.start_time, "mm:ss.cc")
        end = format_timestamp(problem.end_time, "mm:ss.cc")
        segment = problem.asr_question_text or problem.question_text
        lines = [
            f"# 题目{problem.index:02d}",
            f"> 时间: [{start} - {end}]",
            f"> 置信度: {problem.confidence:.2f}",
            "",
            "## 题目原文段（ASR 截取）",
            segment,
            "",
        ]
        if problem.solution_steps:
            lines.append("## 解题步骤")
            for step in problem.solution_steps:
                ts = format_timestamp(step.start_time, "mm:ss.cc")
                lines.append(f"{step.step_number}. [{ts}] {step.content}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def locate_record_to_dict(
        segment_text: str, strategy: str, confidence: float,
        start_time: float, end_time: float,
        start_idx: int, end_idx: int, keyword: str = "",
    ) -> dict:
        """5. 定位记录 jsonl 单行字典"""
        record = {
            "segment": segment_text[:200],
            "strategy": strategy,
            "confidence": round(confidence, 4),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "start_idx": start_idx,
            "end_idx": end_idx,
        }
        if keyword:
            record["keyword"] = keyword
        return record
