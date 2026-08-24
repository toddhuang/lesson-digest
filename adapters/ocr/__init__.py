"""
OCR 适配层
定义统一 OCR 接口，封装具体 OCR 引擎。
"""

from adapters.ocr.base import OCRAdapter
from adapters.ocr.mock import MockOCRAdapter
from adapters.ocr.paddleocr import PaddleOCRAdapter
from adapters.ocr.factory import create_ocr_adapter

__all__ = [
    "OCRAdapter",
    "MockOCRAdapter",
    "PaddleOCRAdapter",
    "create_ocr_adapter",
]
