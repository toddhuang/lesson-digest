"""
LLM 适配器工厂函数
根据适配器类型创建对应的 LLM 适配器实例。

M11-M17 重构：
- 统一使用 LiteLLM 适配器（支持所有 OpenAI 兼容服务商）
- 保留 Mock 适配器用于测试
- 适配器实例绑定具体模型和服务商配置
"""

from config import ModelConfig, ProviderConfig
from adapters.llm.base import LLMAdapter
from adapters.llm.mock import MockLLMAdapter
from adapters.llm.litellm_adapter import LiteLLMAdapter


def create_llm_adapter(
    model_config: ModelConfig,
    provider_config: ProviderConfig,
    max_retries: int = 3,
    timeout: int = 120,
    mock: bool = False,
) -> LLMAdapter:
    """创建 LLM 适配器

    Args:
        model_config: 模型配置
        provider_config: 服务商配置
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        mock: 是否使用 Mock 适配器（用于测试）

    Returns:
        LLMAdapter 实例
    """
    if mock:
        return MockLLMAdapter(model_config, provider_config, max_retries, timeout)
    return LiteLLMAdapter(model_config, provider_config, max_retries, timeout)
