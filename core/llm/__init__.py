"""
LLM 核心层
包含 LLMClient（客户端）、LLMSession（会话）、LLMGenerator（协议）、RecursiveTextSplitter（分块器）。

业务模块依赖 LLMGenerator 协议，通过依赖注入接收 LLMSession。
"""

from core.llm.protocol import LLMGenerator
from core.llm.llm_session import LLMSession
from core.llm.llm_client import LLMClient
from core.llm.text_splitter import RecursiveTextSplitter

__all__ = [
    "LLMGenerator",
    "LLMSession",
    "LLMClient",
    "RecursiveTextSplitter",
]
