"""
LLM 适配器抽象基类
定义统一 LLM 接口，所有具体 LLM 适配器必须继承此类。

M11-M17 重构：
- 接口从 chat/chat_stream 简化为 generate(prompt, payload, temperature)
- 适配器内部处理：消息组装、流式接收、分块调用、异常映射
- 上层只看到 generate(prompt, payload) -> LLMResponse
"""

from abc import ABC, abstractmethod

from config import ModelConfig, ProviderConfig
from utils.models import LLMResponse


class LLMAdapter(ABC):
    """LLM 适配器抽象基类

    每个适配器实例绑定一个具体模型（ModelConfig）和服务商（ProviderConfig）。
    适配器负责将 prompt + payload 转换为底层 API 调用，
    内部处理流式接收、分块、异常分类等细节。
    """

    @abstractmethod
    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        pass

    @abstractmethod
    def generate(self, prompt: str, payload: str, temperature: float) -> LLMResponse:
        """生成 LLM 响应

        Args:
            prompt: 系统提示词（任务指令）
            payload: 待处理内容（数据）
            temperature: 温度参数（由 LLMSession 从任务配置传入）

        Returns:
            LLMResponse 对象

        Raises:
            LLMClientError: 参数错误或认证失败（不重试）
            LLMRateLimitError: 速率限制（可重试）
            LLMTimeoutError: 超时（可重试）
            LLMConnectionError: 连接失败（可重试）
            LLMServerError: 服务端错误（可重试）
            LLMContextOverflowError: 输入超过上下文限制
        """
        pass
