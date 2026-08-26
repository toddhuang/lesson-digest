"""
LiteLLM 适配器
通过 LiteLLM 统一调用多家 LLM 服务商（豆包、DeepSeek 等）。

职责：
- 将 prompt + payload 组装为 messages
- 内部统一 stream=True 接收完整响应
- payload 超限时自动分块调用并拼接结果
- 分类处理 LiteLLM 异常，映射为项目自定义异常

M11-M17 重构：替代原 OpenAICompatibleAdapter / VolcengineAdapter / DeepSeekAdapter
"""

from typing import List

from config import ModelConfig, ProviderConfig
from utils.models import LLMResponse, TokenUsage
from utils.token_counter import count_tokens
from utils.logger import setup_logger
from utils.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMClientError,
    LLMConnectionError,
    LLMServerError,
    LLMContextOverflowError,
)
from adapters.llm.base import LLMAdapter
from core.llm.text_splitter import RecursiveTextSplitter

logger = setup_logger("LLM_LiteLLM")

# 上下文安全系数：实际可用空间 = context_length * 此系数 - prompt_tokens - max_output
_CONTEXT_SAFETY_FACTOR = 0.9


class LiteLLMAdapter(LLMAdapter):
    """基于 LiteLLM 的 LLM 适配器"""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.model_config = model_config
        self.provider_config = provider_config
        self.max_retries = max_retries
        self.timeout = timeout
        self._splitter = RecursiveTextSplitter()

    def generate(self, prompt: str, payload: str, temperature: float) -> LLMResponse:
        """生成 LLM 响应，自动处理分块"""
        prompt_tokens = count_tokens(prompt)
        available_tokens = int(
            self.model_config.context_length * _CONTEXT_SAFETY_FACTOR
        ) - prompt_tokens - self.model_config.max_output

        if available_tokens <= 0:
            raise LLMContextOverflowError(
                f"Prompt 本身 token 数({prompt_tokens}) + 预留输出({self.model_config.max_output}) "
                f"超过上下文限制({self.model_config.context_length})，无法处理"
            )

        payload_tokens = count_tokens(payload)

        if payload_tokens <= available_tokens:
            return self._call_single(prompt, payload, temperature)

        # payload 超限，分块处理
        chunks = self._splitter.split(payload, available_tokens)
        if len(chunks) <= 1:
            raise LLMContextOverflowError(
                f"Payload 无法在上下文限制内切分（{payload_tokens} tokens，"
                f"单块上限 {available_tokens} tokens）"
            )

        logger.info(
            f"[LLM] Payload 超限({payload_tokens}>{available_tokens})，"
            f"分 {len(chunks)} 块调用"
        )

        all_content: List[str] = []
        total_usage = TokenUsage()
        for i, chunk in enumerate(chunks):
            logger.info(f"[LLM] 分块调用 {i + 1}/{len(chunks)}")
            response = self._call_single(prompt, chunk, temperature)
            all_content.append(response.content)
            total_usage.prompt_tokens += response.usage.prompt_tokens
            total_usage.completion_tokens += response.usage.completion_tokens
            total_usage.total_tokens += response.usage.total_tokens

        return LLMResponse(
            content="\n".join(all_content),
            model=self.model_config.name,
            usage=total_usage,
            finish_reason="stop",
        )

    def _call_single(
        self, prompt: str, payload: str, temperature: float
    ) -> LLMResponse:
        """单次 API 调用（流式接收）"""
        import litellm

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ]

        litellm_model = (
            f"{self.provider_config.litellm_prefix}/{self.model_config.name}"
        )

        try:
            stream = litellm.completion(
                model=litellm_model,
                api_base=self.provider_config.base_url,
                api_key=self.provider_config.api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=self.model_config.max_output,
                stream=True,
                stream_options={"include_usage": True},
                num_retries=self.max_retries,
                timeout=self.timeout,
            )
        except litellm.BadRequestError as e:
            logger.error(f"[LLM] 参数错误(400): {e}")
            raise LLMClientError(f"参数错误: {e}") from e
        except litellm.AuthenticationError as e:
            logger.error(f"[LLM] 认证失败(401): {e}")
            raise LLMClientError(f"认证失败: {e}") from e
        except litellm.RateLimitError as e:
            retry_after = float(getattr(e, "retry_after", 0) or 0)
            logger.error(f"[LLM] 速率限制(429), retry_after={retry_after}s: {e}")
            raise LLMRateLimitError(str(e), retry_after=retry_after) from e
        except litellm.InternalServerError as e:
            logger.error(f"[LLM] 服务端错误(5xx): {e}")
            raise LLMServerError(str(e)) from e
        except litellm.Timeout as e:
            logger.error(f"[LLM] 调用超时: {e}")
            raise LLMTimeoutError(str(e)) from e
        except litellm.APIConnectionError as e:
            logger.error(f"[LLM] 连接失败: {e}")
            raise LLMConnectionError(str(e)) from e
        except litellm.APIError as e:
            status_code = getattr(e, "status_code", None)
            if status_code and status_code >= 500:
                logger.error(f"[LLM] 服务端错误({status_code}): {e}")
                raise LLMServerError(str(e)) from e
            logger.error(f"[LLM] API错误({status_code}): {e}")
            raise LLMError(str(e)) from e

        return self._collect_stream(stream)

    def _collect_stream(self, stream) -> LLMResponse:
        """收集流式响应为完整 LLMResponse"""
        content_parts: List[str] = []
        usage = TokenUsage()
        finish_reason = "stop"

        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    delta_content = getattr(delta, "content", None)
                    if delta_content:
                        content_parts.append(delta_content)
                if choices[0].finish_reason:
                    finish_reason = choices[0].finish_reason

            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                prompt_tokens = getattr(chunk_usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(chunk_usage, "completion_tokens", 0) or 0
                total_tokens = getattr(chunk_usage, "total_tokens", 0) or 0
                if prompt_tokens:
                    usage.prompt_tokens = prompt_tokens
                if completion_tokens:
                    usage.completion_tokens = completion_tokens
                if total_tokens:
                    usage.total_tokens = total_tokens

        content = "".join(content_parts)
        if not usage.total_tokens:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        if not content:
            logger.warning(
                f"[LLM] 返回内容为空, finish_reason={finish_reason}"
            )

        return LLMResponse(
            content=content,
            model=self.model_config.name,
            usage=usage,
            finish_reason=finish_reason,
        )
