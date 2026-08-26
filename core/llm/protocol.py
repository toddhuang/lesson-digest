"""
LLM 生成器协议
业务模块依赖此 Protocol，不依赖具体 LLM 实现（DIP）。

LLMSession 是此协议的具体实现，由 LLMClient.get_session() 创建。
业务模块通过依赖注入接收 LLMGenerator，不知道底层模型和服务商。
"""

from typing import Protocol, runtime_checkable

from utils.models import LLMResponse


@runtime_checkable
class LLMGenerator(Protocol):
    """LLM 生成器协议

    业务模块只看到这一个方法：
    - prompt: 系统提示词（任务指令）
    - payload: 待处理内容（数据）
    返回完整的 LLMResponse
    """

    def generate(self, prompt: str, payload: str) -> LLMResponse:
        ...
