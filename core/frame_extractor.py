"""
M3 关键帧提取模块
从视频中按间隔提取关键帧图片。
使用 ffmpeg 命令行工具。
对应文档：03_接口设计/M3_关键帧提取模块接口.md
"""

import os
import subprocess
from typing import List

from utils.models import FrameInfo
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from utils.exceptions import FFmpegError, InvalidVideoError, TimestampOutOfRangeError

logger = setup_logger("M3_frame")


class FrameExtractor:
    """关键帧提取器"""

    def __init__(self, interval: int = 30, fmt: str = "jpg", quality: int = 90):
        self.interval = interval
        self.format = fmt
        self.quality = quality

    def extract_frames(self, video_path: str, output_dir: str) -> List[FrameInfo]:
        """按间隔提取关键帧

        使用 ffmpeg 的 fps 滤镜按固定间隔提取关键帧。

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出图片目录

        Returns:
            FrameInfo 列表

        Raises:
            InvalidVideoError: 视频文件不存在
            FFmpegError: ffmpeg 执行失败
        """
        if not os.path.exists(video_path):
            raise InvalidVideoError(f"视频文件不存在: {video_path}")

        logger.info(f"[M3] 提取关键帧: {video_path} -> {output_dir}, 间隔={self.interval}s")
        ensure_dir(output_dir)

        # 检查是否已有提取的帧（缓存机制）
        existing_frames = sorted([
            f for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(f".{self.format}")
        ])
        if existing_frames:
            logger.info(f"[M3] 关键帧已存在，跳过提取: {len(existing_frames)}帧")
            return self._build_frame_list(output_dir, existing_frames)

        # ffmpeg 质量参数映射：quality 1-100 → q:v 2-31（值越小质量越高）
        qv = max(2, min(31, int(31 - (self.quality / 100) * 29)))

        # ffmpeg 命令：按间隔抽帧
        output_pattern = os.path.join(output_dir, f"frame_%06d.{self.format}")
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps=1/{self.interval}",
            "-q:v", str(qv),
            "-y",
            output_pattern
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError:
            raise FFmpegError("ffmpeg 未找到，请确认 ffmpeg 已安装并加入 PATH")
        except subprocess.TimeoutExpired:
            raise FFmpegError(f"ffmpeg 关键帧提取超时: {video_path}")

        if result.returncode != 0:
            raise FFmpegError(
                f"ffmpeg 关键帧提取失败: {result.stderr[-500:]}",
                stderr=result.stderr,
                returncode=result.returncode
            )

        # 收集提取的帧
        frame_files = sorted([
            f for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(f".{self.format}")
        ])

        frames = self._build_frame_list(output_dir, frame_files)
        logger.info(f"[M3] 关键帧提取完成: {len(frames)}帧")
        return frames

    def _build_frame_list(self, output_dir: str, frame_files: List[str]) -> List[FrameInfo]:
        """构建 FrameInfo 列表，按文件名序号计算时间戳"""
        frames = []
        for i, filename in enumerate(frame_files):
            frame_path = os.path.join(output_dir, filename)
            timestamp = i * self.interval  # 按序号计算时间戳
            file_size = os.path.getsize(frame_path) if os.path.exists(frame_path) else 0

            # 获取图片尺寸（使用 ffprobe）
            width, height = self._get_image_size(frame_path)

            frames.append(FrameInfo(
                path=frame_path,
                timestamp=float(timestamp),
                format=self.format,
                size_bytes=file_size,
                width=width,
                height=height,
            ))
        return frames

    def _get_image_size(self, image_path: str) -> tuple:
        """使用 ffprobe 获取图片尺寸

        Args:
            image_path: 图片文件路径

        Returns:
            (width, height) 元组
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                image_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0, 0

    def extract_frame_at(self, video_path: str, timestamp: float, output_path: str) -> str:
        """在指定时间点提取单帧（用于题目截图）

        Args:
            video_path: 输入视频文件路径
            timestamp: 提取时间点（秒）
            output_path: 输出图片路径

        Returns:
            输出图片路径

        Raises:
            InvalidVideoError: 视频文件不存在
            TimestampOutOfRangeError: 时间戳超出视频时长
            FFmpegError: ffmpeg 执行失败
        """
        if not os.path.exists(video_path):
            raise InvalidVideoError(f"视频文件不存在: {video_path}")

        if timestamp < 0:
            raise TimestampOutOfRangeError(f"时间戳不能为负数: {timestamp}")

        ensure_dir(os.path.dirname(output_path))

        # 如果已存在，跳过
        if os.path.exists(output_path):
            logger.info(f"[M3] 单帧已存在，跳过: {output_path}")
            return output_path

        # ffmpeg 命令：在指定时间点提取单帧
        # -ss 放在 -i 前面可以更快定位（但精度稍低），放在后面精度更高
        qv = max(2, min(31, int(31 - (self.quality / 100) * 29)))
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", str(qv),
            "-y",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            raise FFmpegError("ffmpeg 未找到，请确认 ffmpeg 已安装并加入 PATH")
        except subprocess.TimeoutExpired:
            raise FFmpegError(f"ffmpeg 单帧提取超时: {video_path}")

        if result.returncode != 0:
            raise FFmpegError(
                f"ffmpeg 单帧提取失败 (t={timestamp}s): {result.stderr[-500:]}",
                stderr=result.stderr,
                returncode=result.returncode
            )

        logger.info(f"[M3] 单帧提取: t={timestamp}s -> {output_path}")
        return output_path
