"""
LLM 适配层
定义统一 LLM 接口，封装具体 LLM 服务。
"""

from adapters.llm.base import LLMAdapter
from adapters.llm.mock import MockLLMAdapter
from adapters.llm.openai_compatible import OpenAICompatibleAdapter
from adapters.llm.deepseek import DeepSeekAdapter
from adapters.llm.volcengine import VolcengineAdapter
from adapters.llm.factory import create_llm_adapter

__all__ = [
    "LLMAdapter",
    "MockLLMAdapter",
    "OpenAICompatibleAdapter",
    "DeepSeekAdapter",
    "VolcengineAdapter",
    "create_llm_adapter",
]
