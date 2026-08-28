"""
PP-FormulaNet 公式识别适配器
使用 PaddleOCR 3.x 公式识别产线（版面检测 + PP-FormulaNet_plus-M），
输出印刷体公式 LaTeX 和版面分类。
R-008 定案：与 PP-OCRv6 并行运行，分别识别公式和文字。
"""

from typing import List

from utils.models import OCRResult
from adapters.ocr.base import OCRAdapter


class FormulaNetAdapter(OCRAdapter):
    """PP-FormulaNet 公式识别适配器（PaddleOCR 3.x）"""

    def __init__(self):
        self._pipeline = None
        self._config = {}

    def load_model(self, config: dict) -> None:
        """加载 PP-FormulaNet 产线模型

        Args:
            config: 配置字典，支持 formula_model_name 等参数
        """
        self._config = config
        model_name = config.get("formula_model_name", "PP-FormulaNet_plus-M")

        from paddleocr import FormulaRecognitionPipeline
        from utils.logger import setup_logger
        logger = setup_logger("FormulaNet")

        logger.info(f"加载 PP-FormulaNet 产线: model={model_name}")

        self._pipeline = FormulaRecognitionPipeline(
            formula_recognition_model_name=model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        logger.info("PP-FormulaNet 产线加载完成")

    def unload_model(self) -> None:
        """卸载模型，释放 GPU 显存"""
        if self._pipeline is not None:
            import paddle
            del self._pipeline
            self._pipeline = None
            paddle.device.cuda.empty_cache()

    def recognize(self, image_path: str) -> List[OCRResult]:
        """公式识别，返回 OCRResult 列表（block_type=formula/title/text/image）

        Args:
            image_path: 图片文件路径

        Returns:
            OCRResult 列表，包含公式 LaTeX 和版面分类结果
        """
        if self._pipeline is None:
            raise RuntimeError("FormulaNet 模型未加载，请先调用 load_model()")

        from utils.logger import setup_logger
        logger = setup_logger("FormulaNet")

        result = self._pipeline.predict(image_path)
        if not result:
            return []

        res = result[0]  # dict-like 对象
        img_w, img_h = self._get_image_size(image_path)

        ocr_results: List[OCRResult] = []

        # 版面检测结果
        layout = res.get("layout_det_res", None)
        if layout:
            for box in layout.get("boxes", []):
                label = box["label"]
                score = float(box["score"])
                coords = box["coordinate"]
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1 = min(xs) / img_w if img_w > 0 else 0
                y1 = min(ys) / img_h if img_h > 0 else 0
                x2 = max(xs) / img_w if img_w > 0 else 1
                y2 = max(ys) / img_h if img_h > 0 else 1

                ocr_results.append(OCRResult(
                    text=f"[{label}]",
                    confidence=score,
                    bounding_box=[x1, y1, x2, y2],
                    block_type=label,  # formula / text / title / image / header
                ))

        # 公式识别结果
        for item in res.get("formula_res_list", []):
            latex = item.get("rec_formula", "")
            rec_score = float(item.get("rec_score", 0.0))

            # dt_polys 可能是 numpy array 或 list
            polys = item.get("dt_polys", [])
            box_coords = None
            if polys is not None and len(polys) > 0:
                try:
                    poly = polys[0] if hasattr(polys, '__len__') else polys
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x1 = min(xs) / img_w if img_w > 0 else 0
                    y1 = min(ys) / img_h if img_h > 0 else 0
                    x2 = max(xs) / img_w if img_w > 0 else 1
                    y2 = max(ys) / img_h if img_h > 0 else 1
                    box_coords = [x1, y1, x2, y2]
                except (IndexError, TypeError, ValueError):
                    box_coords = None

            ocr_results.append(OCRResult(
                text=latex,
                confidence=rec_score,
                bounding_box=box_coords or [],
                block_type="formula",
                latex=latex,
            ))

        logger.info(f"FormulaNet 识别完成: {len(ocr_results)}条（含版面和公式）")
        return ocr_results

    def _get_image_size(self, image_path: str) -> tuple:
        """获取图片尺寸"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.width, img.height
        except (FileNotFoundError, OSError, ImportError):
            return 1920, 1080
