"""
LLM 适配层
定义统一 LLM 接口，封装具体 LLM 服务。
"""

from adapters.llm.base import LLMAdapter
from adapters.llm.mock import MockLLMAdapter
from adapters.llm.litellm_adapter import LiteLLMAdapter
from adapters.llm.factory import create_llm_adapter

__all__ = [
    "LLMAdapter",
    "MockLLMAdapter",
    "LiteLLMAdapter",
    "create_llm_adapter",
]
