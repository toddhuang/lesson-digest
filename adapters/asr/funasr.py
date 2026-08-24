"""
FunASR 适配器
使用 paraformer-zh + fsmn-vad + ct-punc 模型组合。
"""

import re
import subprocess
from typing import List

from utils.models import Sentence
from adapters.asr.base import ASRAdapter


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

    def transcribe(self, audio_path: str) -> List[Sentence]:
        """语音识别，返回带时间戳的句子列表

        Args:
            audio_path: 音频文件路径（WAV 格式，16kHz 单声道）

        Returns:
            Sentence 列表
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
            return []

        result = res[0]
        text = result.get("text", "")
        timestamp = result.get("timestamp", [])

        sentences = self._split_to_sentences(text, timestamp, audio_path)

        logger.info(f"FunASR 识别完成: {len(sentences)}句")
        return sentences

    def _split_to_sentences(self, text: str, timestamp: list, audio_path: str) -> List[Sentence]:
        """将识别文本按标点拆分为句子，并估算时间戳

        Args:
            text: 识别文本（带标点）
            timestamp: 字级时间戳列表 [[start_ms, end_ms], ...]
            audio_path: 音频文件路径（用于获取总时长）

        Returns:
            Sentence 列表
        """
        duration = self._get_audio_duration(audio_path)

        sentence_endings = r'[。？！]'
        parts = re.split(f'({sentence_endings})', text)

        sentences = []
        current_text = ""
        char_index = 0

        for part in parts:
            if not part:
                continue
            if re.match(sentence_endings, part):
                current_text += part
                start_time, end_time = self._get_sentence_timestamp(
                    char_index, char_index + len(current_text), timestamp, duration
                )
                sentences.append(Sentence(
                    start_time=start_time,
                    end_time=end_time,
                    text=current_text.strip(),
                    confidence=0.9,
                ))
                char_index += len(current_text)
                current_text = ""
            else:
                current_text += part

        if current_text.strip():
            start_time, end_time = self._get_sentence_timestamp(
                char_index, char_index + len(current_text), timestamp, duration
            )
            sentences.append(Sentence(
                start_time=start_time,
                end_time=end_time,
                text=current_text.strip(),
                confidence=0.9,
            ))

        return sentences

    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频总时长（秒）"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0.0

    def _get_sentence_timestamp(self, start_char: int, end_char: int,
                                  timestamp: list, duration: float) -> tuple:
        """根据字级时间戳估算句子的时间戳范围

        Args:
            start_char: 句子起始字符索引
            end_char: 句子结束字符索引
            timestamp: 字级时间戳列表 [[start_ms, end_ms], ...]
            duration: 音频总时长（秒）

        Returns:
            (start_time, end_time) 元组（秒）
        """
        if timestamp and start_char < len(timestamp):
            start_ms = timestamp[start_char][0] if start_char < len(timestamp) else 0
            end_idx = min(end_char - 1, len(timestamp) - 1)
            end_ms = timestamp[end_idx][1] if end_idx >= 0 else 0
            return start_ms / 1000.0, end_ms / 1000.0
        else:
            if duration > 0 and timestamp:
                total_chars = len(timestamp)
                start_time = (start_char / total_chars) * duration if total_chars > 0 else 0
                end_time = (end_char / total_chars) * duration if total_chars > 0 else duration
                return start_time, end_time
            return 0.0, 0.0
