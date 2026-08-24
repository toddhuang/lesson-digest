"""
PaddleOCR 适配器
使用 PaddleOCR 检测+识别模型。
"""

import subprocess
from typing import List

from utils.models import OCRResult
from adapters.ocr.base import OCRAdapter


class PaddleOCRAdapter(OCRAdapter):
    """PaddleOCR 适配器，使用 PaddleOCR 检测+识别模型"""

    def __init__(self):
        self._ocr = None
        self._config = {}

    def load_model(self, config: dict) -> None:
        """加载 PaddleOCR 模型

        Args:
            config: 配置字典，支持 language, use_angle_cls, use_gpu 等参数
        """
        self._config = config
        language = config.get("language", "ch")
        use_angle_cls = config.get("use_angle_cls", True)

        from paddleocr import PaddleOCR
        from utils.logger import setup_logger
        logger = setup_logger("PaddleOCR")

        logger.info(f"加载 PaddleOCR 模型: language={language}, use_angle_cls={use_angle_cls}")

        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=language,
            use_gpu=True,
        )
        logger.info("PaddleOCR 模型加载完成")

    def unload_model(self) -> None:
        """卸载模型，释放 GPU 显存"""
        if self._ocr is not None:
            import paddle
            del self._ocr
            self._ocr = None
            paddle.device.cuda.empty_cache()

    def recognize(self, image_path: str) -> List[OCRResult]:
        """文字识别，返回识别结果列表

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 列表
        """
        if self._ocr is None:
            raise RuntimeError("PaddleOCR 模型未加载，请先调用 load_model()")

        from utils.logger import setup_logger
        logger = setup_logger("PaddleOCR")

        img_width, img_height = self._get_image_size(image_path)

        result = self._ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return []

        ocr_results = []
        for line in result[0]:
            box = line[0]
            text, confidence = line[1]

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1 = min(xs) / img_width if img_width > 0 else 0
            y1 = min(ys) / img_height if img_height > 0 else 0
            x2 = max(xs) / img_width if img_width > 0 else 1
            y2 = max(ys) / img_height if img_height > 0 else 1

            ocr_results.append(OCRResult(
                text=text,
                confidence=float(confidence),
                bounding_box=[x1, y1, x2, y2],
            ))

        logger.info(f"PaddleOCR 识别完成: {len(ocr_results)}条文本")
        return ocr_results

    def _get_image_size(self, image_path: str) -> tuple:
        """获取图片尺寸

        Args:
            image_path: 图片文件路径

        Returns:
            (width, height) 元组
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.width, img.height
        except (FileNotFoundError, OSError, ImportError):
            try:
                cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                       "-show_entries", "stream=width,height", "-of", "csv=p=0", image_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        return int(parts[0]), int(parts[1])
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                pass
        return 1920, 1080
