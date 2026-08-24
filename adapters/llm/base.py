"""
LLM 适配器抽象基类
定义统一 LLM 接口，所有具体 LLM 适配器必须继承此类。
"""

from abc import ABC, abstractmethod
from typing import List, Iterator

from utils.models import LLMResponse, LLMChunk


class LLMAdapter(ABC):
    """LLM 适配器抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """非流式对话"""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> Iterator[LLMChunk]:
        """流式对话"""
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """返回模型上下文长度"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """统计文本的 token 数"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名"""
        pass

    @abstractmethod
    def rebuild_client(self) -> None:
        """重建底层 HTTP 客户端（用于断线重连）"""
        pass
