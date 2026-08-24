"""
M11 LLM 客户端模块
LLM 调用业务逻辑（多服务商管理、重试、缓存、健康检查、断线重连），通过适配层调用具体LLM服务。
对应文档：03_接口设计/M11_LLM客户端模块接口.md
"""

import os
import time
import hashlib
import json
from typing import List, Iterator, Optional

from config import Config, LLMConfig
from utils.models import LLMResponse, LLMChunk
from utils.file_utils import ensure_dir, save_json, load_json
from utils.logger import setup_logger
from utils.exceptions import (
    LLMError, LLMTimeoutError, LLMRateLimitError, LLMConnectionError,
    LLMContextOverflowError, InvalidBackendError
)
from adapters.llm_adapter import create_llm_adapter, LLMAdapter

logger = setup_logger("M11_llm")


class LLMClient:
    """LLM 客户端（支持多服务商：deepseek / volcengine 等）"""

    def __init__(self, config: LLMConfig, cache_dir: str = "./cache/llm"):
        self.config = config
        self.cache_dir = cache_dir
        self._adapters: dict = {}  # key: 服务商名, value: LLMAdapter 实例
        self._health_status = {}
        self._last_health_check = {}

    def _resolve_backend(self, backend: str) -> str:
        """解析后端别名

        - "cloud" -> config.default_provider（向后兼容）
        - 其他 -> 直接作为服务商名（deepseek/volcengine 等）
        """
        if backend == "cloud":
            return self.config.default_provider
        return backend

    def _get_adapter(self, backend: str) -> LLMAdapter:
        """获取指定服务商的适配器实例（懒加载+缓存）

        Args:
            backend: 服务商名（deepseek/volcengine/mock）或别名（cloud）

        Returns:
            LLMAdapter 实例
        """
        provider_name = self._resolve_backend(backend)

        if provider_name in self._adapters:
            return self._adapters[provider_name]

        # 从配置中读取服务商配置
        provider_config = self.config.providers.get(provider_name)
        if provider_config is None:
            raise InvalidBackendError(
                f"未配置的LLM服务商: {provider_name}，"
                f"已配置: {list(self.config.providers.keys())}"
            )

        # 构建适配器配置
        adapter_config = {
            "base_url": provider_config.base_url,
            "model": provider_config.model,
            "context_length": provider_config.context_length,
            "api_key": provider_config.api_key,
        }

        adapter = create_llm_adapter(provider_name, adapter_config)
        self._adapters[provider_name] = adapter
        logger.info(f"[M11] 创建LLM适配器: {provider_name} (model={provider_config.model})")
        return adapter

    def _count_messages_tokens(self, messages: List[dict]) -> int:
        """统计消息列表的总 token 数"""
        total = 0
        for msg in messages:
            total += len(msg.get("content", "")) // 2  # 简单估算
        return total

    def chat(
        self,
        messages: List[dict],
        backend: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        use_cache: bool = True
    ) -> LLMResponse:
        """非流式对话

        Args:
            messages: 对话消息列表
            backend: 服务商名（deepseek/volcengine）或别名（cloud）
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
            use_cache: 是否使用缓存

        Returns:
            LLMResponse 对象
        """
        provider_name = self._resolve_backend(backend)
        adapter = self._get_adapter(provider_name)

        # 2. Token 超限检测
        input_tokens = self._count_messages_tokens(messages)
        context_length = adapter.get_context_length()
        if input_tokens + max_tokens > context_length:
            raise LLMContextOverflowError(
                f"输入token({input_tokens}) + 输出预留({max_tokens}) 超过上下文长度({context_length})"
            )

        # 3. 缓存检查
        if use_cache:
            model_name = adapter.get_model_name()
            cache_key = self._get_cache_key(messages, provider_name, model_name, temperature, top_p, max_tokens)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                logger.info(f"[M11] 命中LLM缓存: {cache_key[:8]}")
                data = load_json(cache_path)
                return LLMResponse(**data)

        # 4. 带重试调用
        response = self._call_with_retry(adapter, messages, temperature, top_p, max_tokens)

        # 5. 写入缓存
        if use_cache:
            ensure_dir(self.cache_dir)
            save_json(response.__dict__, cache_path)

        logger.info(f"[M11] LLM调用完成: {provider_name}, {response.usage.total_tokens} tokens")
        return response

    def _call_with_retry(self, adapter, messages, temperature, top_p, max_tokens) -> LLMResponse:
        """带重试的适配器调用"""
        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                return adapter.chat(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            except (LLMTimeoutError, LLMRateLimitError, LLMConnectionError) as e:
                if attempt == max_retries - 1:
                    raise LLMError(f"重试{max_retries}次后失败: {e}")
                wait_time = 5 * (2 ** attempt)
                logger.warning(f"[M11] 调用失败，{wait_time}s后重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
        raise LLMError("重试耗尽")

    def chat_stream(
        self,
        messages: List[dict],
        backend: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000
    ) -> Iterator[LLMChunk]:
        """流式对话"""
        provider_name = self._resolve_backend(backend)
        adapter = self._get_adapter(provider_name)
        return adapter.chat_stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def health_check(self, backend: str) -> bool:
        """健康检查"""
        provider_name = self._resolve_backend(backend)
        try:
            adapter = self._get_adapter(provider_name)
            # 简单的健康检查：发送一个极短的请求
            adapter.chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0,
            )
            return True
        except Exception as e:
            logger.warning(f"[M11] 健康检查失败 ({provider_name}): {e}")
            return False

    def reconnect(self, backend: str) -> bool:
        """断线重连"""
        provider_name = self._resolve_backend(backend)
        adapter = self._get_adapter(provider_name)
        adapter.rebuild_client()
        return True

    def count_tokens(self, text: str, backend: str = "cloud") -> int:
        """统计文本的 token 数"""
        provider_name = self._resolve_backend(backend)
        adapter = self._get_adapter(provider_name)
        return adapter.count_tokens(text)

    def get_context_length(self, backend: str) -> int:
        """获取指定服务商的上下文长度"""
        provider_name = self._resolve_backend(backend)
        adapter = self._get_adapter(provider_name)
        return adapter.get_context_length()

    def _get_cache_key(self, messages, backend, model, temperature, top_p, max_tokens) -> str:
        """计算缓存键（包含model，避免同一服务商切换模型时缓存错误命中）"""
        key_data = {
            "messages": messages,
            "backend": backend,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
