"""
M17 LLM 适配层
定义统一 LLM 接口，封装具体 LLM 服务。
mock阶段使用 MockLLMAdapter 返回假数据。
对应文档：03_接口设计/M17_LLM适配层接口.md
"""

import json
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional

from utils.models import LLMResponse, LLMChunk, TokenUsage


class LLMAdapter(ABC):
    """LLM 适配器抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """非流式对话"""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> Iterator[LLMChunk]:
        """流式对话"""
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """返回模型上下文长度"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """统计文本的 token 数"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名"""
        pass

    @abstractmethod
    def rebuild_client(self) -> None:
        """重建底层 HTTP 客户端（用于断线重连）"""
        pass


class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器，根据消息内容返回不同的假数据"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.base_url = config.get("base_url", "http://mock.local/v1")
        self.model = config.get("model", "mock-model")
        self._context_length = config.get("context_length", 131072)
        self._client = "mock_client"

    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse:
        """根据 system prompt 内容判断任务类型，返回对应的假数据"""
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        content = self._generate_mock_response(system_content, user_content)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=len(user_content) // 2,
                completion_tokens=len(content) // 2,
                total_tokens=len(user_content) // 2 + len(content) // 2,
            ),
            finish_reason="stop",
        )

    def _generate_mock_response(self, system_content: str, user_content: str) -> str:
        """根据任务类型生成模拟响应"""
        # 思维导图生成任务（优先匹配，因为system prompt中也包含"知识点"字样）
        if "思维导图" in system_content or "OPML" in system_content:
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

        # 知识点提取任务
        if "知识点" in system_content:
            return json.dumps([
                {"index": 1, "name": "一元二次方程的定义", "start_time": "00:05", "confidence": 0.95},
                {"index": 2, "name": "一元二次方程的一般形式", "start_time": "00:12", "confidence": 0.93},
                {"index": 3, "name": "因式分解法解方程", "start_time": "00:28", "confidence": 0.92},
                {"index": 4, "name": "求根公式推导", "start_time": "00:42", "confidence": 0.94},
                {"index": 5, "name": "判别式与根的关系", "start_time": "01:28", "confidence": 0.91},
            ], ensure_ascii=False)

        # 题目提取任务
        if "题目" in system_content or "习题" in system_content:
            return json.dumps([
                {
                    "index": 1,
                    "start_time": "00:20",
                    "end_time": "00:35",
                    "question_text": "解方程：x²-5x+6=0",
                    "solution_steps": [
                        {"step_number": 1, "content": "因式分解：x²-5x+6=(x-2)(x-3)", "timestamp": "00:28"},
                        {"step_number": 2, "content": "令(x-2)(x-3)=0，得x=2或x=3", "timestamp": "00:32"},
                    ],
                    "has_image": False,
                    "image_description": "",
                    "source": "教材例题",
                    "confidence": 0.93,
                },
                {
                    "index": 2,
                    "start_time": "00:58",
                    "end_time": "01:20",
                    "question_text": "用求根公式解方程：2x²+3x-2=0",
                    "solution_steps": [
                        {"step_number": 1, "content": "确定a=2, b=3, c=-2", "timestamp": "01:02"},
                        {"step_number": 2, "content": "计算判别式Δ=b²-4ac=9+16=25", "timestamp": "01:08"},
                        {"step_number": 3, "content": "代入求根公式x=(-3±5)/4", "timestamp": "01:12"},
                        {"step_number": 4, "content": "x₁=1/2, x₂=-2", "timestamp": "01:16"},
                    ],
                    "has_image": False,
                    "image_description": "",
                    "source": "课堂练习",
                    "confidence": 0.90,
                },
            ], ensure_ascii=False)

        # 默认响应
        return "这是一个mock响应。"

    def chat_stream(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> Iterator[LLMChunk]:
        """流式对话，将非流式响应拆分为多个chunk"""
        response = self.chat(messages, temperature, top_p, max_tokens, **kwargs)
        # 简单地按字符拆分
        chunk_size = 10
        for i in range(0, len(response.content), chunk_size):
            yield LLMChunk(
                delta_content=response.content[i:i+chunk_size],
                finish_reason=None,
                usage=None,
            )
        # 最后一个chunk带finish_reason和usage
        yield LLMChunk(
            delta_content="",
            finish_reason="stop",
            usage=response.usage,
        )

    def get_context_length(self) -> int:
        return self._context_length

    def count_tokens(self, text: str) -> int:
        from utils.token_counter import count_tokens
        return count_tokens(text)

    def get_model_name(self) -> str:
        return self.model

    def rebuild_client(self) -> None:
        self._client = "mock_client_rebuilt"


class OpenAICompatibleAdapter(LLMAdapter):
    """OpenAI 兼容 API 适配器基类（vLLM 和 DeepSeek 都基于此）"""

    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "")
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "sk-placeholder")
        self._context_length = config.get("context_length", 8192)
        self._timeout = config.get("timeout", 120)
        self._client = None
        self._build_client()

    def _build_client(self) -> None:
        """构建 OpenAI 客户端"""
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self._timeout,
        )

    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse:
        """非流式对话"""
        from utils.logger import setup_logger
        logger = setup_logger("LLM")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "stop",
        )

    def chat_stream(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> Iterator[LLMChunk]:
        """流式对话"""
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as e:
            from utils.logger import setup_logger
            logger = setup_logger("LLM")
            logger.error(f"LLM 流式调用失败: {e}")
            raise

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield LLMChunk(
                    delta_content=chunk.choices[0].delta.content,
                    finish_reason=None,
                    usage=None,
                )
            if chunk.choices and chunk.choices[0].finish_reason:
                yield LLMChunk(
                    delta_content="",
                    finish_reason=chunk.choices[0].finish_reason,
                    usage=TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens if chunk.usage else 0,
                        completion_tokens=chunk.usage.completion_tokens if chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if chunk.usage else 0,
                    ),
                )

    def get_context_length(self) -> int:
        return self._context_length

    def count_tokens(self, text: str) -> int:
        from utils.token_counter import count_tokens
        return count_tokens(text)

    def get_model_name(self) -> str:
        return self.model

    def rebuild_client(self) -> None:
        """重建底层 HTTP 客户端（用于断线重连）"""
        from utils.logger import setup_logger
        logger = setup_logger("LLM")
        logger.info("重建 LLM 客户端")
        del self._client
        self._build_client()


class VLLMAdapter(OpenAICompatibleAdapter):
    """本地 vLLM 适配器（Qwen3.6-27B AWQ，8K 上下文）"""

    def __init__(self, config: dict):
        # vLLM 默认配置
        default_config = {
            "base_url": "http://192.168.x.x:8000/v1",
            "model": "Qwen3.6-27B-AWQ",
            "api_key": "EMPTY",  # vLLM 默认不需要 API key
            "context_length": 8192,
            "timeout": 120,
        }
        default_config.update(config)
        super().__init__(default_config)


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """云端 DeepSeek API 适配器（deepseek-chat，128K 上下文）"""

    def __init__(self, config: dict):
        import os
        # DeepSeek 默认配置
        default_config = {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "context_length": 131072,
            "timeout": 120,
        }
        default_config.update(config)
        super().__init__(default_config)


def create_llm_adapter(backend: str, config: dict) -> LLMAdapter:
    """LLM 适配器工厂函数

    Args:
        backend: 后端类型（"vllm"/"deepseek"/"mock"）
        config: 后端配置

    Returns:
        LLMAdapter 实例
    """
    adapters = {
        "mock": MockLLMAdapter,
        "vllm": VLLMAdapter,
        "deepseek": DeepSeekAdapter,
    }
    if backend not in adapters:
        raise ValueError(f"不支持的LLM后端: {backend}")
    return adapters[backend](config)
