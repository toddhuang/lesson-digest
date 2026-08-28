"""
OCR 适配器工厂函数
根据 adapter_type 创建对应的 OCR 适配器实例。
R-008 定案：支持 paddleocr（PP-OCRv6 文字）和 formula_net（PP-FormulaNet 公式）两个引擎。
"""

from adapters.ocr.base import OCRAdapter
from adapters.ocr.mock import MockOCRAdapter
from adapters.ocr.paddleocr import PaddleOCRAdapter
from adapters.ocr.formula_net import FormulaNetAdapter


def create_ocr_adapter(adapter_type: str, config: dict) -> OCRAdapter:
    """OCR 适配器工厂函数

    Args:
        adapter_type: 适配器类型（"paddleocr"/"formula_net"/"easyocr"/"mock"）
        config: 适配器配置

    Returns:
        OCRAdapter 实例
    """
    adapters = {
        "mock": MockOCRAdapter,
        "paddleocr": PaddleOCRAdapter,
        "formula_net": FormulaNetAdapter,
        "easyocr": MockOCRAdapter,  # EasyOCR 尚未实现，降级为 mock
    }
    if adapter_type not in adapters:
        raise ValueError(f"不支持的OCR适配器类型: {adapter_type}")
    adapter = adapters[adapter_type]()
    adapter.load_model(config)
    return adapter
