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

        # 获取图片尺寸用于坐标归一化
        img_width, img_height = self._get_image_size(image_path)

        # 调用 PaddleOCR
        result = self._ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return []

        ocr_results = []
        for line in result[0]:
            # line 格式: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", confidence)]
            box = line[0]
            text, confidence = line[1]

            # 计算归一化的 bounding box [x1, y1, x2, y2]
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
        except Exception:
            # PIL 不可用时使用 ffprobe
            try:
                import subprocess
                cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                       "-show_entries", "stream=width,height", "-of", "csv=p=0", image_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        return int(parts[0]), int(parts[1])
            except Exception:
                pass
        return 1920, 1080  # 默认尺寸


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
        "paddleocr": PaddleOCRAdapter,
        "easyocr": MockOCRAdapter,  # EasyOCR 尚未实现，降级为 mock
    }
    if adapter_type not in adapters:
        raise ValueError(f"不支持的OCR适配器类型: {adapter_type}")
    adapter = adapters[adapter_type]()
    adapter.load_model(config)
    return adapter
