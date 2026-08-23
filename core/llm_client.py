"""
M11 LLM 客户端模块
LLM 调用业务逻辑（双后端管理、重试、缓存、健康检查、断线重连），通过适配层调用具体LLM服务。
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
    """LLM 客户端"""

    def __init__(self, config: LLMConfig, cache_dir: str = "./cache/llm"):
        self.config = config
        self.cache_dir = cache_dir
        self._local_adapter: Optional[LLMAdapter] = None
        self._cloud_adapter: Optional[LLMAdapter] = None
        self._health_status = {"local": None, "cloud": None}
        self._last_health_check = {"local": 0, "cloud": 0}

    def _get_adapter(self, backend: str) -> LLMAdapter:
        if backend == "local":
            if self._local_adapter is None:
                self._local_adapter = create_llm_adapter("vllm", {
                    "base_url": self.config.local.base_url,
                    "model": self.config.local.model,
                    "context_length": self.config.local.context_length,
                })
            return self._local_adapter
        elif backend == "cloud":
            if self._cloud_adapter is None:
                self._cloud_adapter = create_llm_adapter("deepseek", {
                    "base_url": self.config.cloud.base_url,
                    "model": self.config.cloud.model,
                    "context_length": self.config.cloud.context_length,
                    "api_key": self.config.cloud.api_key,
                })
            return self._cloud_adapter
        else:
            raise InvalidBackendError(f"无效的后端: {backend}，必须是 'local' 或 'cloud'")

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
            backend: 后端选择（"local"/"cloud"）
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
            use_cache: 是否使用缓存

        Returns:
            LLMResponse 对象
        """
        # 1. 校验 backend
        if backend not in ("local", "cloud"):
            raise InvalidBackendError(f"无效的后端: {backend}")

        adapter = self._get_adapter(backend)

        # 2. Token 超限检测
        input_tokens = self._count_messages_tokens(messages)
        context_length = adapter.get_context_length()
        if input_tokens + max_tokens > context_length:
            raise LLMContextOverflowError(
                f"输入token({input_tokens}) + 输出预留({max_tokens}) 超过上下文长度({context_length})"
            )

        # 3. 缓存检查
        if use_cache:
            cache_key = self._get_cache_key(messages, backend, temperature, top_p, max_tokens)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                logger.info(f"[M11] 命中LLM缓存: {cache_key[:8]}")
                data = load_json(cache_path)
                return LLMResponse(**data)

        # 4. 健康检查（mock阶段跳过，直接认为健康）
        # 5. 带重试调用
        response = self._call_with_retry(adapter, messages, temperature, top_p, max_tokens)

        # 6. 写入缓存
        if use_cache:
            ensure_dir(self.cache_dir)
            save_json(response.__dict__, cache_path)

        logger.info(f"[M11] LLM调用完成: {backend}, {response.usage.total_tokens} tokens")
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
        if backend not in ("local", "cloud"):
            raise InvalidBackendError(f"无效的后端: {backend}")
        adapter = self._get_adapter(backend)
        return adapter.chat_stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def health_check(self, backend: str) -> bool:
        """健康检查（mock版直接返回True）"""
        return True

    def reconnect(self, backend: str) -> bool:
        """断线重连（mock版直接返回True）"""
        adapter = self._get_adapter(backend)
        adapter.rebuild_client()
        return True

    def count_tokens(self, text: str, backend: str = "local") -> int:
        """统计文本的 token 数"""
        adapter = self._get_adapter(backend)
        return adapter.count_tokens(text)

    def get_context_length(self, backend: str) -> int:
        """获取指定后端的上下文长度"""
        adapter = self._get_adapter(backend)
        return adapter.get_context_length()

    def _get_cache_key(self, messages, backend, temperature, top_p, max_tokens) -> str:
        """计算缓存键"""
        key_data = {
            "messages": messages,
            "backend": backend,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
