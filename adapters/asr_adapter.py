"""
M15 语音识别适配层
定义统一 ASR 接口，封装具体 ASR 引擎。
mock阶段使用 MockASRAdapter 返回假数据。
对应文档：03_接口设计/M15_语音识别适配层接口.md
"""

from abc import ABC, abstractmethod
from typing import List

from utils.models import Sentence


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
    def transcribe(self, audio_path: str) -> List[Sentence]:
        """语音识别，返回带时间戳的句子列表

        Args:
            audio_path: 音频文件路径

        Returns:
            Sentence 列表
        """
        pass


class MockASRAdapter(ASRAdapter):
    """Mock ASR 适配器，返回假数据用于链路测试"""

    def __init__(self):
        self._loaded = False

    def load_model(self, config: dict) -> None:
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    def transcribe(self, audio_path: str) -> List[Sentence]:
        """返回模拟的教学视频语音识别结果"""
        return [
            Sentence(start_time=0.0, end_time=5.2, text="同学们好，今天我们来学习一元二次方程。", confidence=0.95),
            Sentence(start_time=5.2, end_time=12.0, text="首先我们来看定义，只含有一个未知数，并且未知数的最高次数是2的整式方程叫做一元二次方程。", confidence=0.93),
            Sentence(start_time=12.0, end_time=20.5, text="它的一般形式是ax平方加bx加c等于0，其中a不等于0。", confidence=0.94),
            Sentence(start_time=20.5, end_time=28.0, text="接下来我们看一道例题，已知方程x平方减5x加6等于0，求方程的解。", confidence=0.92),
            Sentence(start_time=28.0, end_time=35.0, text="这道题我们可以用因式分解的方法，x平方减5x加6等于(x-2)(x-3)，所以x等于2或x等于3。", confidence=0.91),
            Sentence(start_time=35.0, end_time=42.0, text="下面我们来推导求根公式，对于一般形式ax平方加bx加c等于0，我们用配方法来求解。", confidence=0.93),
            Sentence(start_time=42.0, end_time=50.0, text="首先将方程两边同时除以a，得到x平方加(b/a)x加(c/a)等于0。", confidence=0.90),
            Sentence(start_time=50.0, end_time=58.0, text="然后配方，x平方加(b/a)x等于(x加b/2a)的平方减(b平方/4a平方)。", confidence=0.89),
            Sentence(start_time=58.0, end_time=65.0, text="代入后整理得到(x加b/2a)的平方等于(b平方减4ac)/4a平方。", confidence=0.91),
            Sentence(start_time=65.0, end_time=72.0, text="两边开方，x加b/2a等于正负根号(b平方减4ac)/2a。", confidence=0.92),
            Sentence(start_time=72.0, end_time=80.0, text="所以x等于(-b正负根号(b平方减4ac))/2a，这就是求根公式。", confidence=0.94),
            Sentence(start_time=80.0, end_time=88.0, text="我们来看判别式，b平方减4ac叫做一元二次方程的判别式，通常用希腊字母德尔塔表示。", confidence=0.93),
            Sentence(start_time=88.0, end_time=95.0, text="当德尔塔大于0时，方程有两个不相等的实数根；当德尔塔等于0时，方程有两个相等的实数根。", confidence=0.92),
            Sentence(start_time=95.0, end_time=100.0, text="当德尔塔小于0时，方程没有实数根。好，今天的课就到这里，同学们再见。", confidence=0.95),
        ]


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

        # 使用 paraformer-zh + fsmn-vad + ct-punc 组合
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

        # 调用 FunASR
        res = self._model.generate(
            input=audio_path,
            batch_size_s=300,
        )

        if not res:
            return []

        result = res[0]
        text = result.get("text", "")
        timestamp = result.get("timestamp", [])

        # 按标点分句
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
        import re
        import os
        import subprocess

        # 获取音频总时长
        duration = 0.0
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
        except Exception:
            pass

        # 按句号、问号、感叹号分句
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
                # 计算句子的时间戳范围
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

        # 处理剩余文本（没有结束标点的情况）
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
            # 使用字级时间戳
            start_ms = timestamp[start_char][0] if start_char < len(timestamp) else 0
            end_idx = min(end_char - 1, len(timestamp) - 1)
            end_ms = timestamp[end_idx][1] if end_idx >= 0 else 0
            return start_ms / 1000.0, end_ms / 1000.0
        else:
            # 没有字级时间戳，按字符比例估算
            if duration > 0 and timestamp:
                total_chars = len(timestamp)
                start_time = (start_char / total_chars) * duration if total_chars > 0 else 0
                end_time = (end_char / total_chars) * duration if total_chars > 0 else duration
                return start_time, end_time
            return 0.0, 0.0


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
