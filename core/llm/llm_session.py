"""
LLM 会话
绑定具体模型和 temperature，实现 LLMGenerator 协议。

由 LLMClient.get_session(task_name) 创建，注入到业务模块。
会话是轻量级的，可以为每个任务创建独立会话。
"""

from adapters.llm.base import LLMAdapter
from core.llm.protocol import LLMGenerator
from utils.models import LLMResponse


class LLMSession(LLMGenerator):
    """LLM 会话

    每个会话绑定一个适配器实例（模型）和一个 temperature 值（任务）。
    业务模块通过此会话调用 LLM，不需要知道模型名、temperature 等细节。
    """

    def __init__(self, adapter: LLMAdapter, temperature: float, model_name: str = ""):
        self._adapter = adapter
        self._temperature = temperature
        self._model_name = model_name

    def generate(self, prompt: str, payload: str) -> LLMResponse:
        """生成 LLM 响应

        Args:
            prompt: 系统提示词（任务指令）
            payload: 待处理内容（数据）

        Returns:
            LLMResponse 对象
        """
        return self._adapter.generate(prompt, payload, self._temperature)

    @property
    def model_name(self) -> str:
        """返回此会话绑定的模型名"""
        return self._model_name

    @property
    def temperature(self) -> float:
        """返回此会话的 temperature 值"""
        return self._temperature
