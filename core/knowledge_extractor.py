"""
M7 知识点提取模块
基于全文本，识别视频中的知识点并标注时间戳。
对应文档：03_接口设计/M7_知识点提取模块接口.md

M11-M17 重构：
- 依赖 LLMGenerator 协议，不依赖具体 LLMClient
- 调用 generate(prompt, payload)，不感知模型/temperature/分块
- 删除 use_cache 参数（缓存由 pipeline 层管理）
"""

import json
import re
from typing import List

from utils.models import KnowledgePoint
from utils.timestamp import parse_timestamp
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError, EmptyResultError
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


class KnowledgeExtractor:
    """知识点提取器"""

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

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

    def _parse_response(self, content: str) -> List[KnowledgePoint]:
        """解析 LLM 返回的 JSON"""
        clean = re.sub(r'```json\s*', '', content)
        clean = re.sub(r'```\s*', '', clean)
        clean = clean.strip()

        try:
            data = json.loads(clean)
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
