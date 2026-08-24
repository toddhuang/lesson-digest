"""
M5 文字识别模块
文字识别业务逻辑（帧去重、缓存、时间戳管理），通过适配层调用具体OCR引擎。
支持颜色过滤预处理：去除老师彩色手写，只保留黑色印刷体题目。
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
    """文字识别器"""

    def __init__(self, adapter_type: str = "mock", config=None, cache_dir: str = "./cache/ocr"):
        self.adapter_type = adapter_type
        self.config = config
        self.cache_dir = cache_dir
        self._adapter: OCRAdapter = None
        # 颜色过滤配置（支持OCRConfig对象或dict，向后兼容）
        if hasattr(config, 'enable_color_filter'):
            # OCRConfig对象
            self.enable_color_filter = config.enable_color_filter
            self.black_threshold = config.black_threshold
        elif isinstance(config, dict):
            # dict格式（向后兼容）
            ocr_config = config.get("ocr", {})
            self.enable_color_filter = ocr_config.get("enable_color_filter", True)
            self.black_threshold = ocr_config.get("black_threshold", 80)
        else:
            # 默认值
            self.enable_color_filter = True
            self.black_threshold = 120
        self.processed_dir = os.path.join(cache_dir, "processed")

    def _get_adapter(self) -> OCRAdapter:
        if self._adapter is None:
            self._adapter = create_ocr_adapter(self.adapter_type, self.config)
        return self._adapter

    def _preprocess_frame(self, frame_path: str) -> str:
        """对视频帧进行预处理（颜色过滤）

        Args:
            frame_path: 原始帧路径

        Returns:
            处理后的帧路径（如果未启用过滤则返回原路径）
        """
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
        """对关键帧列表进行文字识别

        Args:
            frame_paths: 关键帧图片路径列表
            frame_timestamps: 关键帧时间戳列表（与frame_paths一一对应）
            use_cache: 是否使用缓存

        Returns:
            OCRFrameResult 列表
        """
        logger.info(f"[M5] 文字识别: {len(frame_paths)}帧, 颜色过滤={'开' if self.enable_color_filter else '关'}")

        results = []
        adapter = self._get_adapter()

        for i, (frame_path, timestamp) in enumerate(zip(frame_paths, frame_timestamps)):
            # 缓存检查（使用文件绝对路径的md5，避免不同视频的frame_0001.jpg冲突）
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
                # 颜色过滤预处理
                ocr_frame_path = self._preprocess_frame(frame_path)

                # 调用适配层
                ocr_results = adapter.recognize(ocr_frame_path)
                full_text = "\n".join(r.text for r in ocr_results)

                # 简单去重：与前一帧full_text比较
                is_duplicate = False
                if results and results[-1].full_text == full_text:
                    is_duplicate = True

                frame_result = OCRFrameResult(
                    timestamp=timestamp,
                    image_path=frame_path,
                    results=ocr_results,
                    full_text=full_text,
                    is_duplicate=is_duplicate,
                )

                # 写入缓存
                if use_cache:
                    ensure_dir(self.cache_dir)
                    save_json({
                        "timestamp": timestamp,
                        "image_path": frame_path,
                        "results": [r.__dict__ for r in ocr_results],
                        "full_text": full_text,
                        "is_duplicate": is_duplicate,
                    }, cache_path)

            results.append(frame_result)
            logger.info(f"[M5] 帧{i+1}/{len(frame_paths)}: t={timestamp}s, {len(frame_result.results)}条文本")

        logger.info(f"[M5] 文字识别完成: {len(results)}帧")
        return results
