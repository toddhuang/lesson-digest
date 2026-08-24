"""
M7 知识点提取模块
基于全文本，识别视频中的知识点并标注时间戳。
纯云端方案：直接调用默认 LLM 服务商（DeepSeek/豆包），128K 上下文无需分段。
对应文档：03_接口设计/M7_知识点提取模块接口.md
"""

import json
import re
from typing import List

from utils.models import KnowledgePoint
from utils.timestamp import parse_timestamp
from utils.logger import setup_logger
from utils.exceptions import LLMResponseParseError, EmptyResultError
from utils.token_counter import count_tokens
from core.llm_client import LLMClient

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
    """知识点提取器（纯云端，无需分段）"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract(self, full_text: str, video_duration: float, use_cache: bool = True) -> List[KnowledgePoint]:
        """从全文本中提取知识点列表

        Args:
            full_text: 带时间戳的全文本
            video_duration: 视频总时长（秒）
            use_cache: 是否使用缓存

        Returns:
            KnowledgePoint 列表
        """
        total_tokens = count_tokens(full_text)
        logger.info(f"[M7] 知识点提取: 全文本{len(full_text)}字符, 约{total_tokens}token")

        knowledge_points = self._extract_single(full_text, video_duration, use_cache)

        if not knowledge_points:
            raise EmptyResultError("知识点提取结果为空")

        logger.info(f"[M7] 知识点提取完成: {len(knowledge_points)}个知识点")
        return knowledge_points

    def _extract_single(self, text: str, video_duration: float, use_cache: bool) -> List[KnowledgePoint]:
        """单次调用 LLM 提取知识点

        Args:
            text: 全文本
            video_duration: 视频总时长
            use_cache: 是否使用缓存

        Returns:
            KnowledgePoint 列表
        """
        user_prompt = f"【视频时长】{video_duration} 秒\n\n【视频全文本】\n{text}"

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
