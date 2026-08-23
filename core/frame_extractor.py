"""
M3 关键帧提取模块
从视频中按间隔提取关键帧图片。
mock阶段不调用ffmpeg，直接创建空图片文件占位。
对应文档：03_接口设计/M3_关键帧提取模块接口.md
"""

import os
from typing import List

from utils.models import FrameInfo
from utils.file_utils import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("M3_frame")


class FrameExtractor:
    """关键帧提取器"""

    def __init__(self, interval: int = 30, fmt: str = "jpg", quality: int = 90):
        self.interval = interval
        self.format = fmt
        self.quality = quality

    def extract_frames(self, video_path: str, output_dir: str) -> List[FrameInfo]:
        """按间隔提取关键帧（mock版）

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出图片目录

        Returns:
            FrameInfo 列表
        """
        logger.info(f"[M3] 提取关键帧: {video_path} -> {output_dir}, 间隔={self.interval}s")
        ensure_dir(output_dir)

        # mock: 创建4帧占位图片（对应0s/30s/60s/90s）
        frames = []
        for i in range(4):
            timestamp = i * self.interval
            frame_path = os.path.join(output_dir, f"frame_{i+1:06d}.{self.format}")
            if not os.path.exists(frame_path):
                # 创建最小JPEG文件占位
                with open(frame_path, "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")
            frames.append(FrameInfo(
                path=frame_path,
                timestamp=float(timestamp),
                format=self.format,
                size_bytes=os.path.getsize(frame_path),
                width=1920,
                height=1080,
            ))

        logger.info(f"[M3] 关键帧提取完成: {len(frames)}帧")
        return frames

    def extract_frame_at(self, video_path: str, timestamp: float, output_path: str) -> str:
        """在指定时间点提取单帧（mock版，用于题目截图）

        Args:
            video_path: 输入视频文件路径
            timestamp: 提取时间点（秒）
            output_path: 输出图片路径

        Returns:
            输出图片路径
        """
        ensure_dir(os.path.dirname(output_path))
        if not os.path.exists(output_path):
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        logger.info(f"[M3] 单帧提取: t={timestamp}s -> {output_path}")
        return output_path
