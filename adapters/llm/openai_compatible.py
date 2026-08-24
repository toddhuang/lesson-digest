"""
OpenAI 兼容 API 适配器基类
vLLM 和 DeepSeek 都基于此接口，Volcengine 也继承此类但重写 chat 方法。
"""

from typing import Iterator

from utils.models import LLMResponse, LLMChunk, TokenUsage
from adapters.llm.base import LLMAdapter


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
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

        choice = response.choices[0]
        usage = response.usage

        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or ""

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
