"""
M9 题目截图模块
根据题目提取结果，在视频中截取题目画面。
mock阶段不调用ffmpeg，直接创建占位图片。
对应文档：03_接口设计/M9_题目截图模块接口.md
"""

import os
from typing import List

from utils.models import Problem
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from core.frame_extractor import FrameExtractor

logger = setup_logger("M9_screenshot")


class ScreenshotCapture:
    """题目截图器"""

    def __init__(self, frame_extractor: FrameExtractor = None):
        self.frame_extractor = frame_extractor or FrameExtractor()

    def capture_screenshots(self, video_path: str, problems: List[Problem],
                              output_dir: str, enable_color_filter: bool = True) -> List[str]:
        """为每道题截取题目画面

        Args:
            video_path: 视频文件路径
            problems: 题目列表
            output_dir: 截图输出目录
            enable_color_filter: 是否启用颜色过滤（去除老师彩色手写，只保留黑色印刷体题目）

        Returns:
            截图路径列表（与problems一一对应，失败的为None）
        """
        logger.info(f"[M9] 题目截图: {len(problems)}道题 -> {output_dir}, 颜色过滤={'开' if enable_color_filter else '关'}")
        ensure_dir(output_dir)

        screenshot_paths = []
        for problem in problems:
            try:
                # MVP: 直接在题目开始时间点截帧，不做最佳帧选取和裁剪
                filename = f"题目{problem.index:02d}.jpg"
                output_path = os.path.join(output_dir, filename)
                self.frame_extractor.extract_frame_at(video_path, problem.start_time, output_path)

                # 颜色过滤：去除老师彩色手写，只保留黑色印刷体题目
                if enable_color_filter and os.path.exists(output_path):
                    try:
                        from utils.image_preprocess import remove_color_keep_black
                        remove_color_keep_black(output_path, output_path, black_threshold=80)
                    except Exception as e:
                        logger.warning(f"[M9] 题目{problem.index:02d}颜色过滤失败，保留原图: {e}")

                screenshot_paths.append(output_path)
                logger.info(f"[M9] 题目{problem.index:02d}截图: t={problem.start_time}s -> {output_path}")
            except Exception as e:
                logger.warning(f"[M9] 题目{problem.index:02d}截图失败: {e}")
                screenshot_paths.append(None)

        logger.info(f"[M9] 题目截图完成: {len([p for p in screenshot_paths if p])}/{len(problems)}成功")
        return screenshot_paths
