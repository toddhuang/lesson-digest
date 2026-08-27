"""
ASR 适配器抽象基类
定义统一 ASR 接口，所有具体 ASR 适配器必须继承此类。
"""

from abc import ABC, abstractmethod

from utils.models import RawTranscript


class ASRAdapter(ABC):
    """ASR 适配器抽象基类"""

    @abstractmethod
    def load_model(self, config: dict) -> None:
        """加载 ASR 模型"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型，释放 GPU 显存"""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> RawTranscript:
        """语音识别，返回完整文本和字级时间戳

        Args:
            audio_path: 音频文件路径

        Returns:
            RawTranscript，text 与 char_timestamps 等长，
            标点等无语音字符对应位置为 None
        """
        pass
