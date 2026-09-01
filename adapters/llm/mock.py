"""
Mock LLM 适配器
根据 prompt 内容返回不同的假数据，用于链路测试。
实现新的 generate(prompt, payload, temperature) 接口。
"""

import json

from config import ModelConfig, ProviderConfig
from utils.models import LLMResponse, TokenUsage
from adapters.llm.base import LLMAdapter


class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器，根据 prompt 内容返回假数据"""

    def __init__(
        self,
        model_config: ModelConfig = None,
        provider_config: ProviderConfig = None,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        if model_config is None:
            model_config = ModelConfig(
                name="mock-model",
                provider="mock",
                capabilities=["text"],
                context_length=131072,
                max_output=8192,
            )
        self.model_config = model_config

    def generate(self, prompt: str, payload: str, temperature: float) -> LLMResponse:
        """根据 prompt 内容判断任务类型，返回对应的假数据"""
        content = self._generate_mock_response(prompt, payload)

        return LLMResponse(
            content=content,
            model=self.model_config.name,
            usage=TokenUsage(
                prompt_tokens=len(payload) // 2,
                completion_tokens=len(content) // 2,
                total_tokens=(len(payload) + len(content)) // 2,
            ),
            finish_reason="stop",
        )

    def _generate_mock_response(self, prompt: str, payload: str) -> str:
        """根据任务类型生成模拟响应"""
        # 一次性纠错+知识点段+题目段提取（AGENTS.md 约定的合并调用）
        if "corrected_text" in prompt or "一次完成" in prompt:
            return json.dumps({
                "corrected_text": payload,
                "knowledge_segments": [
                    {"name": "二次函数的图像性质", "segment": payload[:min(20, len(payload))]},
                ],
                "problem_segments": [
                    {"segment": payload[:min(15, len(payload))]},
                ],
            }, ensure_ascii=False)

        # 解题过程整理（09 设计 issue #13，ASR+OCR 融合）
        if "解题过程整理助手" in prompt:
            return json.dumps([
                {"step_number": 1, "content": "因式分解：$x^2-5x+6=(x-2)(x-3)$", "start_time": 20.0, "end_time": 28.0},
                {"step_number": 2, "content": "令$(x-2)(x-3)=0$，得 $x=2$ 或 $x=3$", "start_time": 28.0, "end_time": 35.0},
            ], ensure_ascii=False)

        if "思维导图" in prompt or "OPML" in prompt:
            return '''<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head>
    <title>一元二次方程</title>
  </head>
  <body>
    <outline text="第一章 一元二次方程">
      <outline text="1.1 定义与一般形式" _note="时间戳: 00:05"/>
      <outline text="1.2 解法" _note="时间戳: 00:20">
        <outline text="1.2.1 因式分解法" _note="时间戳: 00:28"/>
        <outline text="1.2.2 求根公式法" _note="时间戳: 00:42"/>
      </outline>
      <outline text="1.3 判别式" _note="时间戳: 01:28"/>
    </outline>
  </body>
</opml>'''

        if "知识点" in prompt:
            return json.dumps([
                {"index": 1, "name": "一元二次方程的定义", "start_time": "00:05", "confidence": 0.95},
                {"index": 2, "name": "一元二次方程的一般形式", "start_time": "00:12", "confidence": 0.93},
                {"index": 3, "name": "因式分解法解方程", "start_time": "00:28", "confidence": 0.92},
                {"index": 4, "name": "求根公式推导", "start_time": "00:42", "confidence": 0.94},
                {"index": 5, "name": "判别式与根的关系", "start_time": "01:28", "confidence": 0.91},
            ], ensure_ascii=False)

        if "题目" in prompt or "习题" in prompt:
            return json.dumps([
                {
                    "index": 1,
                    "start_time": "00:20",
                    "end_time": "00:35",
                    "question_text": "解方程：x²-5x+6=0",
                    "solution_steps": [
                        {"step_number": 1, "content": "因式分解：x²-5x+6=(x-2)(x-3)", "start_time": 28.0, "end_time": 32.0},
                        {"step_number": 2, "content": "令(x-2)(x-3)=0，得x=2或x=3", "start_time": 32.0, "end_time": 35.0},
                    ],
                    "has_image": False,
                    "image_description": "",
                    "source": "教材例题",
                    "confidence": 0.93,
                },
            ], ensure_ascii=False)

        return "这是一个mock响应。"
