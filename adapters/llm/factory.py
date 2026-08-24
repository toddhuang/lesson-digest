"""
LLM 适配器工厂函数
根据 backend 类型创建对应的 LLM 适配器实例。
"""

from adapters.llm.base import LLMAdapter
from adapters.llm.mock import MockLLMAdapter
from adapters.llm.deepseek import DeepSeekAdapter
from adapters.llm.volcengine import VolcengineAdapter


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
