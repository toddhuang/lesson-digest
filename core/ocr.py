"""
M5 文字识别模块
文字识别业务逻辑（两引擎并行：PP-OCRv6 识文字 + PP-FormulaNet 识公式），
通过适配层调用具体 OCR 引擎。识不准的手写板书交 LLM 补全。

R-008 定案：两引擎并行 + LLM 补全；颜色过滤降级 P2 可选，MVP 默认关闭。
对应文档：03_接口设计/M5_文字识别模块接口.md
"""

import os
from typing import List

from utils.models import OCRFrameResult, OCRResult
from utils.file_utils import ensure_dir, save_json, load_json
from utils.logger import setup_logger
from adapters.ocr import create_ocr_adapter, OCRAdapter

logger = setup_logger("M5_ocr")


class OCRRecognizer:
    """文字识别器，两引擎并行（PP-OCRv6 + PP-FormulaNet）"""

    def __init__(self, text_adapter_type: str = "paddleocr",
                 formula_adapter_type: str = "formula_net",
                 config=None, cache_dir: str = "./cache/ocr"):
        """初始化

        Args:
            text_adapter_type: 文字识别引擎类型（默认 paddleocr）
            formula_adapter_type: 公式识别引擎类型（默认 formula_net）
            config: 配置（dict 或带 ocr/text_ocr/formula_ocr 属性的对象）
            cache_dir: 缓存目录
        """
        self.text_adapter_type = text_adapter_type
        self.formula_adapter_type = formula_adapter_type
        self.config = config
        self.cache_dir = cache_dir
        self._text_adapter: OCRAdapter = None
        self._formula_adapter: OCRAdapter = None

        # 颜色过滤：MVP 默认关闭，P2 可选
        if hasattr(config, 'enable_color_filter'):
            self.enable_color_filter = config.enable_color_filter
            self.black_threshold = getattr(config, 'black_threshold', 80)
        elif isinstance(config, dict):
            ocr_config = config.get("ocr", {})
            self.enable_color_filter = ocr_config.get("enable_color_filter", False)
            self.black_threshold = ocr_config.get("black_threshold", 80)
        else:
            self.enable_color_filter = False
            self.black_threshold = 80

        self.processed_dir = os.path.join(cache_dir, "processed")

    def _get_text_adapter(self) -> OCRAdapter:
        if self._text_adapter is None:
            text_config = self.config
            if isinstance(self.config, dict):
                text_config = self.config.get("text_ocr", self.config.get("ocr", {}))
            elif hasattr(self.config, "text_ocr"):
                text_config = self.config.text_ocr
            elif hasattr(self.config, "ocr"):
                text_config = self.config.ocr
            self._text_adapter = create_ocr_adapter(self.text_adapter_type, text_config)
        return self._text_adapter

    def _get_formula_adapter(self) -> OCRAdapter:
        if self._formula_adapter is None:
            formula_config = {}
            if isinstance(self.config, dict):
                formula_config = self.config.get("formula_ocr", {})
            elif hasattr(self.config, "formula_ocr"):
                formula_config = self.config.formula_ocr
            self._formula_adapter = create_ocr_adapter(self.formula_adapter_type, formula_config)
        return self._formula_adapter

    def _preprocess_frame(self, frame_path: str) -> str:
        """颜色过滤预处理（P2 可选，MVP 默认关闭）"""
        if not self.enable_color_filter:
            return frame_path

        try:
            from utils.image_preprocess import preprocess_frame_for_ocr
            ensure_dir(self.processed_dir)
            processed_path = preprocess_frame_for_ocr(
                frame_path,
                output_dir=self.processed_dir,
                black_threshold=self.black_threshold,
            )
            return processed_path
        except (FileNotFoundError, OSError, ValueError, ImportError) as e:
            logger.warning(f"[M5] 颜色过滤预处理失败，使用原图: {e}")
            return frame_path

    def recognize_frames(self, frame_paths: List[str], frame_timestamps: List[float],
                         use_cache: bool = True) -> List[OCRFrameResult]:
        """对关键帧列表进行文字识别（两引擎并行）

        Args:
            frame_paths: 关键帧图片路径列表
            frame_timestamps: 关键帧时间戳列表（与frame_paths一一对应）
            use_cache: 是否使用缓存

        Returns:
            OCRFrameResult 列表
        """
        logger.info(f"[M5] 文字识别: {len(frame_paths)}帧, "
                     f"text={self.text_adapter_type}, formula={self.formula_adapter_type}, "
                     f"颜色过滤={'开' if self.enable_color_filter else '关'}")

        results = []
        text_adapter = self._get_text_adapter()
        formula_adapter = self._get_formula_adapter()

        for i, (frame_path, timestamp) in enumerate(zip(frame_paths, frame_timestamps)):
            # 缓存检查
            import hashlib
            cache_key = hashlib.md5(os.path.abspath(frame_path).encode()).hexdigest()[:16]
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

            if use_cache and os.path.exists(cache_path):
                data = load_json(cache_path)
                frame_result = OCRFrameResult(
                    timestamp=data["timestamp"],
                    image_path=data["image_path"],
                    results=[OCRResult(**r) for r in data["results"]],
                    full_text=data["full_text"],
                    is_duplicate=data.get("is_duplicate", False),
                )
            else:
                ocr_frame_path = self._preprocess_frame(frame_path)

                # 两引擎并行识别
                text_results = text_adapter.recognize(ocr_frame_path)
                formula_results = formula_adapter.recognize(ocr_frame_path)

                # 合并结果：文字 + 公式
                merged_results = text_results + formula_results

                # 生成 full_text（文字拼接 + 公式 LaTeX）
                text_parts = [r.text for r in merged_results if r.block_type != "formula"]
                formula_parts = [r.latex or r.text for r in merged_results if r.block_type == "formula"]
                full_text = "\n".join(text_parts + formula_parts)

                # 简单去重：与前一帧 full_text 比较
                is_duplicate = False
                if results and results[-1].full_text == full_text:
                    is_duplicate = True

                frame_result = OCRFrameResult(
                    timestamp=timestamp,
                    image_path=frame_path,
                    results=merged_results,
                    full_text=full_text,
                    is_duplicate=is_duplicate,
                )

                # 写入缓存
                if use_cache:
                    ensure_dir(self.cache_dir)
                    save_json({
                        "timestamp": timestamp,
                        "image_path": frame_path,
                        "results": [r.__dict__ for r in merged_results],
                        "full_text": full_text,
                        "is_duplicate": is_duplicate,
                    }, cache_path)

            results.append(frame_result)
            logger.info(f"[M5] 帧{i+1}/{len(frame_paths)}: t={timestamp}s, "
                         f"{len(frame_result.results)}条（text={sum(1 for r in frame_result.results if r.block_type != 'formula')}, "
                         f"formula={sum(1 for r in frame_result.results if r.block_type == 'formula')}）")

        logger.info(f"[M5] 文字识别完成: {len(results)}帧")
        return results
