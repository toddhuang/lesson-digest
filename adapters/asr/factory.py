"""
ASR 适配器工厂函数
根据 adapter_type 创建对应的 ASR 适配器实例。
"""

from adapters.asr.base import ASRAdapter
from adapters.asr.mock import MockASRAdapter
from adapters.asr.funasr import FunASRAdapter


def create_asr_adapter(adapter_type: str, config: dict) -> ASRAdapter:
    """ASR 适配器工厂函数

    Args:
        adapter_type: 适配器类型（"funasr"/"whisper"/"mock"）
        config: 适配器配置

    Returns:
        ASRAdapter 实例
    """
    adapters = {
        "mock": MockASRAdapter,
        "funasr": FunASRAdapter,
        "whisper": MockASRAdapter,  # Whisper 尚未实现，降级为 mock
    }
    if adapter_type not in adapters:
        raise ValueError(f"不支持的ASR适配器类型: {adapter_type}")
    adapter = adapters[adapter_type]()
    adapter.load_model(config)
    return adapter
