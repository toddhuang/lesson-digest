"""
ASR 适配层
定义统一 ASR 接口，封装具体 ASR 引擎。
"""

from adapters.asr.base import ASRAdapter
from adapters.asr.mock import MockASRAdapter
from adapters.asr.funasr import FunASRAdapter
from adapters.asr.factory import create_asr_adapter

__all__ = [
    "ASRAdapter",
    "MockASRAdapter",
    "FunASRAdapter",
    "create_asr_adapter",
]
