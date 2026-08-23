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
                extra_body={"thinking": {"type": "disabled"}},  # MVP阶段禁用思考过程，避免token被思考占满
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

        choice = response.choices[0]
        usage = response.usage

        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""

        # DeepSeek 推理模型：content 为空但 reasoning 非空，说明 max_tokens 不够
        if not content and reasoning:
            logger.warning(
                f"LLM 返回 content 为空，但 reasoning_content 非空（{len(reasoning)}字符）。"
                f"finish_reason={choice.finish_reason}，可能是 max_tokens 不够，思考过程占满了 token。"
            )

        return LLMResponse(
            content=content,
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


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """云端 DeepSeek API 适配器（deepseek-chat，128K 上下文）"""

    def __init__(self, config: dict):
        import os
        # DeepSeek 默认配置
        default_config = {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "context_length": 131072,
            "timeout": 120,
        }
        default_config.update(config)
        super().__init__(default_config)


class VolcengineAdapter(OpenAICompatibleAdapter):
    """火山引擎豆包 API 适配器（OpenAI 兼容，中文教育场景优化）

    支持模型：doubao-seed-2.0-pro / doubao-1.5-pro / doubao-1.5-lite 等
    API 地址：https://ark.cn-beijing.volces.com/api/v3
    """

    def __init__(self, config: dict):
        import os
        # 火山引擎默认配置
        default_config = {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-seed-2-1-pro-260628",
            "api_key": os.environ.get("VOLCENGINE_API_KEY", ""),
            "context_length": 131072,
            "timeout": 120,
        }
        default_config.update(config)
        # 调用父类的父类（LLMAdapter）的 __init__，跳过 OpenAICompatibleAdapter 的默认 extra_body
        # 实际上直接调用 OpenAICompatibleAdapter.__init__ 即可，extra_body 在 chat 方法中处理
        super(OpenAICompatibleAdapter, self).__init__()
        self.base_url = default_config["base_url"]
        self.model = default_config["model"]
        self.api_key = default_config["api_key"]
        self._context_length = default_config["context_length"]
        self._timeout = default_config["timeout"]
        self._client = None
        self._build_client()

    def _convert_messages_to_responses_input(self, messages: list) -> list:
        """将传统 chat.completions 的 messages 格式转换为 responses.create 的 input 格式

        responses.create 的 input 格式：
        [
            {"role": "system", "content": [{"type": "input_text", "text": "..."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "..."}]},
        ]
        """
        input_list = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                input_list.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                })
            elif isinstance(content, list):
                # 已经是多模态格式，直接转换
                converted_content = []
                for item in content:
                    if item.get("type") == "text":
                        converted_content.append({"type": "input_text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        converted_content.append({"type": "input_image", "image_url": item.get("image_url", "")})
                    else:
                        converted_content.append(item)
                input_list.append({"role": role, "content": converted_content})
        return input_list

    def _extract_responses_output(self, response) -> str:
        """从 responses.create 响应中提取输出文本

        兼容多种返回格式：
        - response.output_text
        - response.output[0].content[0].text
        - response.output[*].content[*].text
        """
        # 方式1：直接有 output_text 属性
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        # 方式2：从 output 数组中提取
        output = getattr(response, "output", None)
        if output and isinstance(output, list):
            texts = []
            for item in output:
                content = getattr(item, "content", None)
                if content and isinstance(content, list):
                    for c in content:
                        text = getattr(c, "text", None)
                        if text:
                            texts.append(text)
            if texts:
                return "".join(texts)

        # 方式3：尝试字典格式
        if isinstance(response, dict):
            if "output_text" in response:
                return response["output_text"]
            if "output" in response and isinstance(response["output"], list):
                texts = []
                for item in response["output"]:
                    if "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if "text" in c:
                                texts.append(c["text"])
                if texts:
                    return "".join(texts)

        return ""

    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse:
        """非流式对话（使用 responses.create 接口，seed 系列新模型专用）

        注意：seed 系列模型默认开启深度思考，会消耗大量 output token。
        通过 reasoning={"effort": "none"} 禁用思考过程，直接输出答案。
        """
        from utils.logger import setup_logger
        logger = setup_logger("LLM")

        # 转换消息格式
        responses_input = self._convert_messages_to_responses_input(messages)

        try:
            response = self._client.responses.create(
                model=self.model,
                input=responses_input,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning={"effort": "none"},  # 禁用深度思考，直接输出
            )
        except Exception as e:
            logger.error(f"火山引擎豆包 LLM 调用失败 (responses.create): {e}")
            raise

        # 提取输出文本（禁用思考后 output_text 直接有值）
        content = getattr(response, "output_text", "")
        if not content:
            content = self._extract_responses_output(response)

        # 提取 token 使用情况
        usage = getattr(response, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
            completion_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        if not content:
            logger.warning(f"火山引擎豆包返回空内容，status={getattr(response, 'status', 'unknown')}")

        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            finish_reason="stop",
        )


def create_llm_adapter(backend: str, config: dict) -> LLMAdapter:
    """LLM 适配器工厂函数

    Args:
        backend: 后端类型（"deepseek"/"volcengine"/"mock"）
        config: 后端配置

    Returns:
        LLMAdapter 实例
    """
    adapters = {
        "mock": MockLLMAdapter,
        "deepseek": DeepSeekAdapter,
        "volcengine": VolcengineAdapter,
    }
    if backend not in adapters:
        raise ValueError(f"不支持的LLM后端: {backend}，支持: {list(adapters.keys())}")
    return adapters[backend](config)
