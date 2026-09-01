"""
M9 截图模块
根据题目/知识点定位时间戳，在视频中截取画面。
- 题目截图：保存到 output，做颜色过滤（去除彩色手写，保留黑色印刷体原题）
- 知识点截图：保存到 debug，不做颜色过滤（保留原画面供参考）
对应文档：03_接口设计/M9_题目截图模块接口.md
"""

import os
from typing import List

from utils.models import KnowledgePoint, Problem
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from utils.exceptions import InvalidVideoError, TimestampOutOfRangeError, FFmpegError
from core.frame_extractor import FrameExtractor

logger = setup_logger("M9_screenshot")


class ScreenshotCapture:
    """截图器（题目 + 知识点）"""

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
                    except (FileNotFoundError, OSError, ValueError, ImportError) as e:
                        logger.warning(f"[M9] 题目{problem.index:02d}颜色过滤失败，保留原图: {e}")

                screenshot_paths.append(output_path)
                logger.info(f"[M9] 题目{problem.index:02d}截图: t={problem.start_time}s -> {output_path}")
            except (InvalidVideoError, TimestampOutOfRangeError, FFmpegError, FileNotFoundError, OSError) as e:
                logger.warning(f"[M9] 题目{problem.index:02d}截图失败: {e}")
                screenshot_paths.append(None)

        logger.info(f"[M9] 题目截图完成: {len([p for p in screenshot_paths if p])}/{len(problems)}成功")
        return screenshot_paths

    def capture_knowledge_screenshots(
        self,
        video_path: str,
        knowledge_points: List[KnowledgePoint],
        output_dir: str,
    ) -> List[str]:
        """为每个知识点截取画面（debug 用，不做颜色过滤）

        复用 FrameExtractor.extract_frame_at 的 ffmpeg 单帧提取能力。
        文件名格式：知识点01_t=05m23s.jpg（issue #12）

        Args:
            video_path: 视频文件路径
            knowledge_points: 知识点列表
            output_dir: 截图输出目录

        Returns:
            截图路径列表（与 knowledge_points 一一对应，失败的为 None）
        """
        logger.info(f"[M9] 知识点截图: {len(knowledge_points)}个 -> {output_dir}")
        ensure_dir(output_dir)

        screenshot_paths = []
        for kp in knowledge_points:
            try:
                ts = self._format_t(kp.start_time)
                filename = f"知识点{kp.index:02d}_t={ts}.jpg"
                output_path = os.path.join(output_dir, filename)
                self.frame_extractor.extract_frame_at(video_path, kp.start_time, output_path)
                screenshot_paths.append(output_path)
                logger.info(f"[M9] 知识点{kp.index:02d}截图: t={kp.start_time}s -> {output_path}")
            except (InvalidVideoError, TimestampOutOfRangeError, FFmpegError, FileNotFoundError, OSError) as e:
                logger.warning(f"[M9] 知识点{kp.index:02d}截图失败: {e}")
                screenshot_paths.append(None)

        logger.info(f"[M9] 知识点截图完成: {len([p for p in screenshot_paths if p])}/{len(knowledge_points)}成功")
        return screenshot_paths

    @staticmethod
    def _format_t(seconds: float) -> str:
        """秒数转文件名时间戳格式：05m23s 或 01h05m23s"""
        total = int(max(0.0, seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02d}h{m:02d}m{s:02d}s"
        return f"{m:02d}m{s:02d}s"
