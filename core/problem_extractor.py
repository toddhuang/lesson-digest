"""
M8 题目提取模块
基于全文本，识别所有题目，提取原题和解题步骤。
OCR 补充：用印刷体题目替换/补充 ASR 原题（更准确）。
对应文档：03_接口设计/M8_题目提取模块接口.md

M11-M17 重构：
- 依赖 LLMGenerator 协议，不依赖具体 LLMClient
- 调用 generate(prompt, payload)，不感知模型/temperature/分块
- 删除 use_cache 参数（缓存由 pipeline 层管理）
"""

import json
import re
from typing import List, Optional

from utils.models import Problem, SolutionStep
from utils.timestamp import parse_timestamp, format_timestamp
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError
from core.llm.protocol import LLMGenerator

logger = setup_logger("M8_problem")


SYSTEM_PROMPT = """你是一个教学视频内容分析助手。请从以下带时间戳的教学视频全文本中，识别老师讲解的所有题目，并提取原题和解题步骤。

【要求】
1. 识别视频中老师讲解的所有题目（例题、习题、考试题等）
2. 老师引入题目的方式多样，注意语义判断，不要遗漏
3. 为每道题标注开始时间和结束时间
4. 提取原题文字
5. 提取解题步骤，分步骤列出，每个步骤标注时间戳
6. 标注题目是否含图
7. 按开始时间顺序输出
8. 输出 JSON 格式，必须紧凑（无缩进、无换行、无Markdown代码块），直接输出JSON数组

【输出格式】
[{"index":1,"start_time":"05:23","end_time":"12:45","question_text":"原题文字","solution_steps":[{"step_number":1,"content":"步骤内容","timestamp":"06:10"}],"has_image":false,"image_description":"","source":"","confidence":0.9}]
"""


class ProblemExtractor:
    """题目提取器（OCR补充原题）"""

    def __init__(self, llm: Optional[LLMGenerator] = None):
        self.llm = llm

    def extract(self, full_text: str, video_duration: float,
                ocr_results: list = None) -> List[Problem]:
        """从全文本中提取题目列表

        Args:
            full_text: 带时间戳的全文本（ASR）
            video_duration: 视频总时长（秒）
            ocr_results: OCR帧结果列表，用于补充原题文字（印刷体更准确）

        Returns:
            Problem 列表
        """
        logger.info(f"[M8] 题目提取: 全文本{len(full_text)}字符, OCR帧={len(ocr_results) if ocr_results else 0}")

        problems = self._extract_single(full_text, video_duration)

        if ocr_results:
            problems = self._enrich_with_ocr(problems, ocr_results)

        if len(problems) > 1:
            problems = self._merge_problems(problems)
            logger.info(f"[M8] 最终去重: -> {len(problems)} 道题")

        logger.info(f"[M8] 题目提取完成: {len(problems)}道题")
        return problems

    def _extract_single(self, text: str, video_duration: float) -> List[Problem]:
        """调用 LLM 提取题目"""
        payload = f"【视频时长】{video_duration} 秒\n\n【视频全文本】\n{text}"

        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=payload)
        return self._parse_response(response.content)

    def _enrich_with_ocr(self, problems: List[Problem], ocr_results: list) -> List[Problem]:
        """用OCR结果补充原题文字（印刷体比ASR更准确）"""
        if not problems or not ocr_results:
            return problems

        enriched_count = 0
        for problem in problems:
            start = problem.start_time - 30
            end = problem.end_time + 30

            frames_in_range = []
            for frame in ocr_results:
                if frame.is_duplicate:
                    continue
                if start <= frame.timestamp <= end:
                    frames_in_range.append(frame)

            if not frames_in_range:
                continue

            best_frame = max(frames_in_range, key=lambda f: len(f.full_text))
            ocr_text = best_frame.full_text.strip()

            if not ocr_text or len(ocr_text) < 10:
                continue

            asr_text = problem.question_text.strip()
            should_replace = False

            math_symbols = sum(1 for c in ocr_text if c in '={}^[]()/∈∉≤≥<>')

            ocr_looks_like_question = False
            question_keywords = ['求', '已知', '设', '若', '则', '证明', '的值', '等于', '多少', '计算', '解']
            if any(kw in ocr_text for kw in question_keywords):
                ocr_looks_like_question = True
            if '？' in ocr_text or '?' in ocr_text:
                ocr_looks_like_question = True
            if math_symbols >= 2 and len(ocr_text) > 20:
                ocr_looks_like_question = True
            if re.search(r'[A-D][\.．、]', ocr_text):
                ocr_looks_like_question = True

            if re.search(r'\d+[\.．、]\s*\S{2,6}', ocr_text) and len(ocr_text) < 100:
                if not any(kw in ocr_text for kw in question_keywords) and '？' not in ocr_text and '?' not in ocr_text:
                    ocr_looks_like_question = False

            if not ocr_looks_like_question:
                continue

            ocr_complete = True
            start_keywords = ['已知', '设', '若', '求', '计算', '证明', '以下', '判断', '下列']
            has_start_keyword = any(ocr_text.startswith(kw) or kw in ocr_text[:10] for kw in start_keywords)
            if re.search(r'[a-zA-Z]{1,2}[\.．:：]\s*$', ocr_text) or ocr_text.endswith('::'):
                ocr_complete = False
            if len(ocr_text) < 20:
                ocr_complete = False
            if '=' in ocr_text and not any(kw in ocr_text for kw in ['求', '则', '的值', '等于', '多少']):
                ocr_complete = False

            if not ocr_complete:
                continue

            if len(ocr_text) > len(asr_text) * 1.2 and len(ocr_text) > 20:
                should_replace = True
            if len(asr_text) < 15 and len(ocr_text) > 20:
                should_replace = True
            if math_symbols >= 2 and len(ocr_text) > len(asr_text):
                should_replace = True
            asr_has_keyword = any(kw in asr_text for kw in question_keywords)
            if not asr_has_keyword and has_start_keyword:
                should_replace = True

            if should_replace:
                problem.asr_question_text = asr_text
                problem.question_text = ocr_text
                problem.source = problem.source or "OCR+ASR"
                enriched_count += 1
                logger.info(f"[M8] 题目{problem.index} OCR补充原题: ASR({len(asr_text)}字) -> OCR({len(ocr_text)}字)")

        logger.info(f"[M8] OCR补充完成: {enriched_count}/{len(problems)}道题使用OCR原题")
        return problems

    def _merge_problems(self, problems: List[Problem]) -> List[Problem]:
        """合并去重题目列表"""
        if not problems:
            return []

        sorted_problems = sorted(problems, key=lambda p: p.start_time)

        merged = []
        for p in sorted_problems:
            found_dup = False
            for m in merged:
                if self._is_duplicate(p, m):
                    if len(p.question_text) > len(m.question_text):
                        m.question_text = p.question_text
                    if len(p.solution_steps) > len(m.solution_steps):
                        m.solution_steps = p.solution_steps
                    m.start_time = min(m.start_time, p.start_time)
                    m.end_time = max(m.end_time, p.end_time)
                    found_dup = True
                    break
            if not found_dup:
                merged.append(p)

        for i, p in enumerate(merged):
            p.index = i + 1

        return merged

    def _is_duplicate(self, p1: Problem, p2: Problem) -> bool:
        """判断两道题是否为同一道题（重复）"""
        if abs(p1.start_time - p2.start_time) > 120:
            return False

        text1 = p1.question_text.strip()
        text2 = p2.question_text.strip()
        if not text1 or not text2:
            return False

        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        similarity = intersection / union if union > 0 else 0

        return similarity > 0.3

    def _parse_response(self, content: str) -> List[Problem]:
        """解析 LLM 返回的 JSON"""
        clean = re.sub(r'```json\s*', '', content)
        clean = re.sub(r'```\s*', '', clean)
        clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"题目提取结果JSON解析失败: {e}, 原始内容: {content[:200]}")

        results = []
        for item in data:
            try:
                steps = []
                for step_data in item.get("solution_steps", []):
                    step = SolutionStep(
                        step_number=step_data.get("step_number", len(steps) + 1),
                        content=step_data.get("content", ""),
                        timestamp=parse_timestamp(step_data.get("timestamp", "00:00")),
                    )
                    steps.append(step)

                problem = Problem(
                    index=item.get("index", len(results) + 1),
                    start_time=parse_timestamp(item.get("start_time", "00:00")),
                    end_time=parse_timestamp(item.get("end_time", "00:00")),
                    question_text=item.get("question_text", ""),
                    solution_steps=steps,
                    has_image=item.get("has_image", False),
                    image_description=item.get("image_description", ""),
                    source=item.get("source", ""),
                    confidence=item.get("confidence", 0.8),
                )
                results.append(problem)
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                logger.warning(f"[M8] 解析题目失败: {e}, item={item}")

        return results

    def to_question_markdown(self, problem: Problem, screenshot_rel: str = None,
                              timestamp_format: str = "mm:ss") -> str:
        """将题目转换为原题 Markdown 格式"""
        lines = [f"# 题目 {problem.index:02d} - 原题\n"]
        lines.append(f"> 时间范围：[{format_timestamp(problem.start_time, timestamp_format)} - {format_timestamp(problem.end_time, timestamp_format)}]")
        if problem.source:
            lines.append(f"> 来源：{problem.source}")
        lines.append("")

        if screenshot_rel:
            lines.append(f"![题目截图]({screenshot_rel})")
            lines.append("")

        lines.append("## 原题")
        lines.append(problem.question_text)
        lines.append("")

        return "\n".join(lines)

    def to_solution_markdown(self, problem: Problem, timestamp_format: str = "mm:ss") -> str:
        """将题目转换为解析 Markdown 格式"""
        lines = [f"# 题目 {problem.index:02d} - 解析\n"]
        lines.append(f"> 时间范围：[{format_timestamp(problem.start_time, timestamp_format)} - {format_timestamp(problem.end_time, timestamp_format)}]")
        lines.append("")

        lines.append("## 原题")
        lines.append(problem.question_text)
        lines.append("")

        lines.append("## 解题步骤")
        for step in problem.solution_steps:
            ts = format_timestamp(step.timestamp, timestamp_format)
            lines.append(f"{step.step_number}. [{ts}] {step.content}")
        lines.append("")

        return "\n".join(lines)
