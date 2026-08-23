"""
M16 文字识别适配层
定义统一 OCR 接口，封装具体 OCR 引擎。
mock阶段使用 MockOCRAdapter 返回假数据。
对应文档：03_接口设计/M16_文字识别适配层接口.md
"""

from abc import ABC, abstractmethod
from typing import List

from utils.models import OCRResult


class OCRAdapter(ABC):
    """OCR 适配器抽象基类"""

    @abstractmethod
    def load_model(self, config: dict) -> None:
        """加载 OCR 模型"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型，释放 GPU 显存"""
        pass

    @abstractmethod
    def recognize(self, image_path: str) -> List[OCRResult]:
        """文字识别，返回识别结果列表

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 列表
        """
        pass


class MockOCRAdapter(OCRAdapter):
    """Mock OCR 适配器，返回假数据用于链路测试"""

    def __init__(self):
        self._loaded = False

    def load_model(self, config: dict) -> None:
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    def recognize(self, image_path: str) -> List[OCRResult]:
        """返回模拟的课件文字识别结果"""
        # 根据图片路径中的时间戳返回不同的模拟内容
        if "frame_000001" in image_path:
            return [
                OCRResult(text="一元二次方程", confidence=0.98, bounding_box=[0.3, 0.1, 0.7, 0.2]),
                OCRResult(text="定义：只含有一个未知数，并且未知数的最高次数是2的整式方程", confidence=0.95, bounding_box=[0.1, 0.3, 0.9, 0.5]),
                OCRResult(text="一般形式：ax²+bx+c=0 (a≠0)", confidence=0.96, bounding_box=[0.2, 0.6, 0.8, 0.75]),
            ]
        elif "frame_000002" in image_path:
            return [
                OCRResult(text="例题1", confidence=0.97, bounding_box=[0.1, 0.1, 0.3, 0.2]),
                OCRResult(text="解方程：x²-5x+6=0", confidence=0.96, bounding_box=[0.2, 0.3, 0.8, 0.45]),
                OCRResult(text="解：(x-2)(x-3)=0", confidence=0.94, bounding_box=[0.2, 0.5, 0.7, 0.65]),
                OCRResult(text="x₁=2, x₂=3", confidence=0.95, bounding_box=[0.3, 0.7, 0.6, 0.85]),
            ]
        elif "frame_000003" in image_path:
            return [
                OCRResult(text="求根公式推导", confidence=0.97, bounding_box=[0.3, 0.1, 0.7, 0.2]),
                OCRResult(text="ax²+bx+c=0", confidence=0.96, bounding_box=[0.3, 0.3, 0.7, 0.4]),
                OCRResult(text="x = (-b±√(b²-4ac)) / 2a", confidence=0.93, bounding_box=[0.2, 0.5, 0.8, 0.65]),
            ]
        elif "frame_000004" in image_path:
            return [
                OCRResult(text="判别式 Δ = b²-4ac", confidence=0.97, bounding_box=[0.25, 0.15, 0.75, 0.3]),
                OCRResult(text="Δ>0：两个不相等的实数根", confidence=0.95, bounding_box=[0.15, 0.4, 0.85, 0.55]),
                OCRResult(text="Δ=0：两个相等的实数根", confidence=0.95, bounding_box=[0.15, 0.55, 0.85, 0.7]),
                OCRResult(text="Δ<0：没有实数根", confidence=0.96, bounding_box=[0.15, 0.7, 0.85, 0.85]),
            ]
        else:
            return [
                OCRResult(text="教学课件", confidence=0.90, bounding_box=[0.4, 0.4, 0.6, 0.5]),
            ]


def create_ocr_adapter(adapter_type: str, config: dict) -> OCRAdapter:
    """OCR 适配器工厂函数

    Args:
        adapter_type: 适配器类型（"paddleocr"/"easyocr"/"mock"）
        config: 适配器配置

    Returns:
        OCRAdapter 实例
    """
    adapters = {
        "mock": MockOCRAdapter,
        "paddleocr": MockOCRAdapter,  # mock阶段统一用Mock
        "easyocr": MockOCRAdapter,
    }
    if adapter_type not in adapters:
        raise ValueError(f"不支持的OCR适配器类型: {adapter_type}")
    adapter = adapters[adapter_type]()
    adapter.load_model(config)
    return adapter
