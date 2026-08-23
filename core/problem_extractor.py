"""
M8 题目提取模块
基于全文本，识别所有题目，提取原题和解题步骤。
对应文档：03_接口设计/M8_题目提取模块接口.md
"""

import json
import re
from typing import List

from utils.models import Problem, SolutionStep
from utils.timestamp import parse_timestamp, format_timestamp
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError
from core.llm_client import LLMClient

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
8. 输出 JSON 格式

【输出格式】
[{
  "index": 1,
  "start_time": "05:23",
  "end_time": "12:45",
  "question_text": "原题文字",
  "solution_steps": [{"step_number": 1, "content": "步骤内容", "timestamp": "06:10"}],
  "has_image": false,
  "image_description": "",
  "source": "",
  "confidence": 0.9
}]
"""


class ProblemExtractor:
    """题目提取器"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract(self, full_text: str, video_duration: float, use_cache: bool = True) -> List[Problem]:
        """从全文本中提取题目列表

        Args:
            full_text: 带时间戳的全文本
            video_duration: 视频总时长（秒）
            use_cache: 是否使用缓存

        Returns:
            Problem 列表
        """
        logger.info(f"[M8] 题目提取: 全文本{len(full_text)}字符")

        user_prompt = f"【视频时长】{video_duration} 秒\n\n【视频全文本】\n{full_text}"

        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            backend="cloud",
            temperature=0.1,
            max_tokens=8000,
            use_cache=use_cache,
        )

        problems = self._parse_response(response.content)
        logger.info(f"[M8] 题目提取完成: {len(problems)}道题")
        return problems

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
                for s in item.get("solution_steps", []):
                    steps.append(SolutionStep(
                        step_number=s.get("step_number", len(steps) + 1),
                        content=s.get("content", ""),
                        timestamp=parse_timestamp(s.get("timestamp", "00:00")),
                    ))

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
            except Exception as e:
                logger.warning(f"[M8] 解析题目失败: {e}, item={item}")

        return results

    def to_question_markdown(self, problem: Problem, screenshot_path: str = None,
                              timestamp_format: str = "mm:ss") -> str:
        """将单道题转换为原题 Markdown 格式"""
        lines = [f"# 题目{problem.index:02d}\n"]
        lines.append(f"> 时间范围：[{format_timestamp(problem.start_time, timestamp_format)} - {format_timestamp(problem.end_time, timestamp_format)}]")
        if problem.source:
            lines.append(f"> 来源：{problem.source}")
        lines.append(f"> 含图：{'是' if problem.has_image else '否'}\n")
        lines.append("## 原题\n")
        lines.append(problem.question_text)
        if screenshot_path:
            lines.append(f"\n## 题目截图\n")
            lines.append(f"![题目{problem.index:02d}截图]({screenshot_path})")
        return "\n".join(lines)

    def to_solution_markdown(self, problem: Problem, timestamp_format: str = "mm:ss") -> str:
        """将单道题转换为解析 Markdown 格式"""
        lines = [f"# 题目{problem.index:02d} 解析\n"]
        lines.append(f"> 时间范围：[{format_timestamp(problem.start_time, timestamp_format)} - {format_timestamp(problem.end_time, timestamp_format)}]\n")
        lines.append("## 原题\n")
        lines.append(problem.question_text)
        lines.append("\n## 解题步骤\n")
        for step in problem.solution_steps:
            ts = format_timestamp(step.timestamp, timestamp_format)
            lines.append(f"### 步骤{step.step_number} [{ts}]")
            lines.append(f"{step.content}\n")
        return "\n".join(lines)
