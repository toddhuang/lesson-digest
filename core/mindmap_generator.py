"""
M10 思维导图生成模块
基于知识点列表，生成课程知识结构（OPML格式）。
对应文档：03_接口设计/M10_思维导图生成模块接口.md

M11-M17 重构：
- 依赖 LLMGenerator 协议，不依赖具体 LLMClient
- 调用 generate(prompt, payload)，不感知模型/temperature/分块
- 删除 use_cache 参数（缓存由 pipeline 层管理）
"""

import os
import xml.etree.ElementTree as ET
from typing import List

from utils.models import KnowledgePoint
from utils.timestamp import format_timestamp
from utils.logger import setup_logger
from utils.exceptions import OPMLValidationError
from core.llm.protocol import LLMGenerator

logger = setup_logger("M10_mindmap")


SYSTEM_PROMPT = """你是一个教学内容结构整理助手。请根据以下教学视频的知识点列表，生成课程知识结构的思维导图。

【要求】
1. 根据知识点列表，整理出课程的章节结构
2. 每个章节下包含相关的知识点
3. 知识点名称使用知识点列表中的名称
4. 为每个知识点标注时间戳（在 _note 属性中）
5. 层级不超过3层
6. 输出标准 OPML 2.0 格式
7. 根节点标题使用提供的视频标题
8. 不要遗漏任何知识点
9. 直接输出 XML，不要包含 Markdown 代码块标记
"""


class MindmapGenerator:
    """思维导图生成器"""

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def generate(self, knowledge_points: List[KnowledgePoint], video_title: str = "",
                 video_duration: float = 0) -> str:
        """基于知识点列表生成 OPML 思维导图

        Args:
            knowledge_points: 知识点列表
            video_title: 视频标题
            video_duration: 视频时长（秒）

        Returns:
            OPML 格式字符串
        """
        logger.info(f"[M10] 思维导图生成: {len(knowledge_points)}个知识点")

        kp_text = "\n".join([
            f"{kp.index}. {kp.name} [{format_timestamp(kp.start_time)}]"
            for kp in knowledge_points
        ])

        payload = f"【视频标题】{video_title or '课程笔记'}\n【视频时长】{format_timestamp(video_duration, 'hh:mm:ss')}\n\n【知识点列表】\n{kp_text}"

        response = self.llm.generate(prompt=SYSTEM_PROMPT, payload=payload)

        opml_content = self._clean_opml(response.content)
        self._validate_opml(opml_content)

        logger.info(f"[M10] 思维导图生成完成: {len(opml_content)}字符")
        return opml_content

    def _clean_opml(self, content: str) -> str:
        """清理 OPML 内容，移除 Markdown 代码块标记"""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content.strip()

    def _validate_opml(self, opml_content: str) -> None:
        """校验 OPML 格式合法性"""
        try:
            root = ET.fromstring(opml_content)
        except ET.ParseError as e:
            raise OPMLValidationError(f"OPML XML 格式错误: {e}")

        if root.tag != "opml" or root.get("version") != "2.0":
            raise OPMLValidationError("根节点必须是 <opml version='2.0'>")
        if root.find("head") is None:
            raise OPMLValidationError("缺少 <head> 节点")
        if root.find("body") is None:
            raise OPMLValidationError("缺少 <body> 节点")

    def save_opml(self, opml_content: str, output_path: str) -> None:
        """保存 OPML 文件"""
        from utils.file_utils import ensure_dir
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(opml_content)
