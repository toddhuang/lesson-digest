"""
OCR 适配器抽象基类
定义统一 OCR 接口，所有具体 OCR 适配器必须继承此类。
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
