"""
Mock ASR 适配器
返回假数据用于链路测试。
"""

from typing import List

from utils.models import Sentence
from adapters.asr.base import ASRAdapter


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
