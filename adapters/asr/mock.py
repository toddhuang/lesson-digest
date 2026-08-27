"""
Mock ASR 适配器
返回假数据用于链路测试。
"""

from utils.models import RawTranscript, CharTime
from adapters.asr.base import ASRAdapter

# 模拟教学视频文本（与旧 Mock 数据一致）
_MOCK_TEXT = (
    "同学们好，今天我们来学习一元二次方程。"
    "首先我们来看定义，只含有一个未知数，并且未知数的最高次数是2的整式方程叫做一元二次方程。"
    "它的一般形式是ax平方加bx加c等于0，其中a不等于0。"
    "接下来我们看一道例题，已知方程x平方减5x加6等于0，求方程的解。"
    "这道题我们可以用因式分解的方法，x平方减5x加6等于(x-2)(x-3)，所以x等于2或x等于3。"
    "下面我们来推导求根公式，对于一般形式ax平方加bx加c等于0，我们用配方法来求解。"
    "首先将方程两边同时除以a，得到x平方加(b/a)x加(c/a)等于0。"
    "然后配方，x平方加(b/a)x等于(x加b/2a)的平方减(b平方/4a平方)。"
    "代入后整理得到(x加b/2a)的平方等于(b平方减4ac)/4a平方。"
    "两边开方，x加b/2a等于正负根号(b平方减4ac)/2a。"
    "所以x等于(-b正负根号(b平方减4ac))/2a，这就是求根公式。"
    "我们来看判别式，b平方减4ac叫做一元二次方程的判别式，通常用希腊字母德尔塔表示。"
    "当德尔塔大于0时，方程有两个不相等的实数根；当德尔塔等于0时，方程有两个相等的实数根。"
    "当德尔塔小于0时，方程没有实数根。好，今天的课就到这里，同学们再见。"
)

# 标点集合（与 FunASRAdapter 保持一致）
_PUNCTUATION = set("，。！？、；：""''（）【】《》"
                   ",.!?;:\"'()[]{}<>-\n\r\t ")


def _build_mock_timestamps(text: str) -> list:
    """为模拟文本生成字级时间戳，按每字 0.2 秒估算"""
    timestamps = []
    current_ms = 0
    for char in text:
        if char in _PUNCTUATION:
            timestamps.append(None)
        else:
            start = current_ms
            end = current_ms + 200
            timestamps.append(CharTime(start_ms=start, end_ms=end))
            current_ms = end
    return timestamps


class MockASRAdapter(ASRAdapter):
    """Mock ASR 适配器，返回假数据用于链路测试"""

    def __init__(self):
        self._loaded = False

    def load_model(self, config: dict) -> None:
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    def transcribe(self, audio_path: str) -> RawTranscript:
        """返回模拟的教学视频语音识别结果"""
        return RawTranscript(
            text=_MOCK_TEXT,
            char_timestamps=_build_mock_timestamps(_MOCK_TEXT),
        )
