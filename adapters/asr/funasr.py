"""
FunASR 适配器
使用 paraformer-zh + fsmn-vad + ct-punc 模型组合。
返回完整文本和字级时间戳，不做句子切分。
"""

import subprocess
from typing import List, Optional

from utils.models import RawTranscript, CharTime
from adapters.asr.base import ASRAdapter

# 中英文标点和空白字符，这些字符在 ct-punc 添加后没有对应语音时间戳
_PUNCTUATION = set(
    "，。！？、；：""''（）【】《》〈〉「」『』…—·"
    ",.!?;:\"'()[]{}<>-\n\r\t "
    "~`@#$%^&*_+=|\\/"
)


class FunASRAdapter(ASRAdapter):
    """FunASR 适配器，使用 paraformer-zh + fsmn-vad + ct-punc 模型组合"""

    def __init__(self):
        self._model = None
        self._config = {}

    def load_model(self, config: dict) -> None:
        """加载 FunASR 模型

        Args:
            config: 配置字典，支持 model_name, vad_model, punc_model 等参数
        """
        self._config = config
        model_name = config.get("model_name", "paraformer-zh")

        from funasr import AutoModel
        from utils.logger import setup_logger
        logger = setup_logger("FunASR")

        logger.info(f"加载 FunASR 模型: {model_name}")

        self._model = AutoModel(
            model=model_name,
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="ct-punc",
            punc_model_revision="v2.0.4",
        )
        logger.info("FunASR 模型加载完成")

    def unload_model(self) -> None:
        """卸载模型，释放 GPU 显存"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_path: str) -> RawTranscript:
        """语音识别，返回完整文本和字级时间戳

        Args:
            audio_path: 音频文件路径（WAV 格式，16kHz 单声道）

        Returns:
            RawTranscript，text 与 char_timestamps 等长，
            标点/空白字符对应位置为 None
        """
        if self._model is None:
            raise RuntimeError("FunASR 模型未加载，请先调用 load_model()")

        from utils.logger import setup_logger
        logger = setup_logger("FunASR")

        logger.info(f"FunASR 识别: {audio_path}")

        res = self._model.generate(
            input=audio_path,
            batch_size_s=300,
        )

        if not res:
            return RawTranscript(text="", char_timestamps=[])

        result = res[0]
        text = result.get("text", "")
        timestamp = result.get("timestamp", [])

        char_timestamps = self._align_timestamps(text, timestamp, logger)

        logger.info(
            f"FunASR 识别完成: {len(text)}字, "
            f"{sum(1 for ct in char_timestamps if ct is not None)}字有时间戳"
        )
        return RawTranscript(text=text, char_timestamps=char_timestamps)

    def _align_timestamps(
        self,
        text: str,
        timestamp: List[List[int]],
        logger,
    ) -> List[Optional[CharTime]]:
        """将 FunASR 返回的时间戳数组与文本字符对齐

        ct-punc 添加的标点没有对应语音，timestamp 数组只包含有语音的字符。
        本方法按顺序将时间戳分配给非标点字符，标点位置设为 None。

        若 timestamp 长度与 text 长度相等（某些 FunASR 版本可能给标点也
        填了时间戳），则直接使用，但 [0,0] 的条目标记为 None。

        Args:
            text: 识别文本（含标点）
            timestamp: FunASR 返回的 [[start_ms, end_ms], ...]
            logger: 日志器

        Returns:
            与 text 等长的列表，每个元素是 CharTime 或 None
        """
        if not timestamp:
            return [None] * len(text)

        # 情况1：时间戳数量与文本长度相等，直接使用
        if len(timestamp) == len(text):
            result = []
            for i, (start_ms, end_ms) in enumerate(timestamp):
                if start_ms == 0 and end_ms == 0 and text[i] in _PUNCTUATION:
                    result.append(None)
                else:
                    result.append(CharTime(start_ms=int(start_ms), end_ms=int(end_ms)))
            return result

        # 情况2：时间戳数量少于文本长度（标点无时间戳），按顺序分配给非标点字符
        result: List[Optional[CharTime]] = []
        ts_idx = 0
        for char in text:
            if char in _PUNCTUATION:
                result.append(None)
            else:
                if ts_idx < len(timestamp):
                    start_ms, end_ms = timestamp[ts_idx]
                    result.append(CharTime(start_ms=int(start_ms), end_ms=int(end_ms)))
                    ts_idx += 1
                else:
                    # 时间戳不足，剩余非标点字符标记为 None
                    result.append(None)
                    logger.warning(
                        f"FunASR 时间戳数量不足: 已分配 {ts_idx} 个，"
                        f"文本还有非标点字符 '{char}' 无时间戳"
                    )

        if ts_idx < len(timestamp):
            logger.warning(
                f"FunASR 时间戳数量多于非标点字符: "
                f"文本 {len(text)} 字，时间戳 {len(timestamp)} 个，"
                f"已分配 {ts_idx} 个，剩余 {len(timestamp) - ts_idx} 个未使用"
            )

        return result
