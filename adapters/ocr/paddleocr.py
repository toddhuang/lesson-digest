"""
PaddleOCR 适配器
使用 PaddleOCR 3.x 检测+识别模型（PP-OCRv6）。
R-008 定案：PP-OCRv6 作为文字识别引擎，与 FormulaNet 并行运行。
"""

import subprocess
from typing import List

from utils.models import OCRResult
from adapters.ocr.base import OCRAdapter


class PaddleOCRAdapter(OCRAdapter):
    """PaddleOCR 3.x 适配器，使用 PP-OCRv6 检测+识别模型"""

    def __init__(self):
        self._ocr = None
        self._config = {}

    def load_model(self, config: dict) -> None:
        """加载 PaddleOCR 3.x 模型

        Args:
            config: 配置字典，支持 text_det_model, text_rec_model 等参数
        """
        self._config = config
        det_model = config.get("text_detection_model_name", "PP-OCRv6_small_det")
        rec_model = config.get("text_recognition_model_name", "PP-OCRv6_small_rec")

        from paddleocr import PaddleOCR
        from utils.logger import setup_logger
        logger = setup_logger("PaddleOCR")

        logger.info(f"加载 PaddleOCR 3.x: det={det_model}, rec={rec_model}")

        self._ocr = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        logger.info("PaddleOCR 3.x 模型加载完成")

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
            OCRResult 列表（block_type=text）
        """
        if self._ocr is None:
            raise RuntimeError("PaddleOCR 模型未加载，请先调用 load_model()")

        from utils.logger import setup_logger
        logger = setup_logger("PaddleOCR")

        img_width, img_height = self._get_image_size(image_path)

        result = self._ocr.predict(image_path)

        if not result:
            return []

        ocr_results = []
        for item in result:
            # PaddleOCR 3.x 返回 dict-like 对象
            texts = item.get("rec_texts", [])
            scores = item.get("rec_scores", [])
            polys = item.get("dt_polys", [])

            for i, text in enumerate(texts):
                confidence = float(scores[i]) if i < len(scores) else 0.0

                # bounding box
                bounding_box = []
                if i < len(polys):
                    poly = polys[i]
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x1 = min(xs) / img_width if img_width > 0 else 0
                    y1 = min(ys) / img_height if img_height > 0 else 0
                    x2 = max(xs) / img_width if img_width > 0 else 1
                    y2 = max(ys) / img_height if img_height > 0 else 1
                    bounding_box = [x1, y1, x2, y2]

                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    block_type="text",
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
