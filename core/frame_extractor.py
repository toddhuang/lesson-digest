"""
M3 关键帧提取模块
R-009 定案：1 秒间隔抽帧 + dHash 差异哈希去重（阈值 0.02）。
抽帧使用 OpenCV VideoCapture，单点截图（extract_frame_at）仍用 ffmpeg 保证时间精度。
对应文档：03_接口设计/M3_关键帧提取模块接口.md
"""

import os
import subprocess
from typing import List

import cv2

from core.frame_dedup import FrameDeduplicator
from utils.models import FrameInfo
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from utils.exceptions import FFmpegError, InvalidVideoError, TimestampOutOfRangeError

logger = setup_logger("M3_frame")


class FrameExtractor:
    """关键帧提取器：1 秒间隔抽帧 + dHash 去重（R-009 定案）

    抽帧使用 OpenCV VideoCapture，帧去重通过组合 FrameDeduplicator 实现。
    单点截图（extract_frame_at）仍用 ffmpeg 保证时间精度，供 M9 题目截图使用。
    """

    def __init__(
        self,
        interval: float = 1.0,
        fmt: str = "jpg",
        quality: int = 90,
        dedup_threshold: float = 0.02,
        enable_dedup: bool = True,
    ):
        """初始化

        Args:
            interval: 抽帧间隔（秒），R-009 默认 1.0
            fmt: 输出图片格式（jpg/png）
            quality: JPG 质量 1-100，仅 jpg 生效
            dedup_threshold: dHash 去重阈值 0~1，R-009 默认 0.02
            enable_dedup: 是否启用 dHash 去重
        """
        self.interval = float(interval)
        self.format = fmt
        self.quality = quality
        self.dedup_threshold = dedup_threshold
        self.enable_dedup = enable_dedup
        self.deduplicator = FrameDeduplicator(threshold=dedup_threshold)

    def extract_frames(self, video_path: str, output_dir: str) -> List[FrameInfo]:
        """按间隔抽帧并 dHash 去重，输出到指定目录

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出图片目录

        Returns:
            FrameInfo 列表（仅保留通过去重的帧，按时间排序）

        Raises:
            InvalidVideoError: 视频文件不存在
            FFmpegError: OpenCV 无法打开视频
        """
        if not os.path.exists(video_path):
            raise InvalidVideoError(f"视频文件不存在: {video_path}")

        logger.info(
            f"[M3] 提取关键帧: {video_path} -> {output_dir}, "
            f"间隔={self.interval}s, "
            f"dHash去重={'开(阈值=' + str(self.dedup_threshold) + ')' if self.enable_dedup else '关'}"
        )
        ensure_dir(output_dir)

        # 缓存检查：已有 frame_ 前缀图片直接按文件名加载
        existing_frames = sorted([
            f for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(f".{self.format}")
        ])
        if existing_frames:
            logger.info(f"[M3] 关键帧已存在，跳过提取: {len(existing_frames)}帧")
            return self._build_frame_list_from_disk(output_dir, existing_frames)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FFmpegError(f"OpenCV 无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * self.interval)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[M3] OpenCV抽帧参数: fps={fps:.2f}, 步长={step}帧, 总帧数={total_frames}")

        self.deduplicator.reset()
        write_params = self._get_write_params()

        frame_infos: List[FrameInfo] = []
        kept_count = 0
        sampled_count = 0
        idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break

            sampled_count += 1
            timestamp = idx / fps

            # dHash 去重判定
            if self.enable_dedup and not self.deduplicator.should_keep(frame):
                idx += step
                continue

            kept_count += 1
            filename = f"frame_{kept_count:06d}.{self.format}"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, frame, write_params)

            height, width = frame.shape[:2]
            size_bytes = os.path.getsize(frame_path) if os.path.exists(frame_path) else 0

            frame_infos.append(FrameInfo(
                path=frame_path,
                timestamp=float(timestamp),
                format=self.format,
                size_bytes=size_bytes,
                width=int(width),
                height=int(height),
            ))

            idx += step

        cap.release()

        if sampled_count > 0:
            logger.info(
                f"[M3] 关键帧提取完成: 抽样={sampled_count}帧, 保留={kept_count}帧, "
                f"压缩比={(sampled_count - kept_count) / sampled_count * 100:.1f}%"
            )
        else:
            logger.info("[M3] 关键帧提取完成: 0帧")
        return frame_infos

    def _build_frame_list_from_disk(self, output_dir: str, frame_files: List[str]) -> List[FrameInfo]:
        """从磁盘缓存的帧文件恢复 FrameInfo 列表

        文件名序号映射为时间戳（序号从1开始，时间戳 = (序号-1) * interval）。
        """
        frames: List[FrameInfo] = []
        for filename in frame_files:
            frame_path = os.path.join(output_dir, filename)
            size_bytes = os.path.getsize(frame_path) if os.path.exists(frame_path) else 0

            # 用 OpenCV 读取尺寸
            width, height = self._get_image_size_cv2(frame_path)

            # 从 frame_000001.jpg 中提取序号
            try:
                stem = os.path.splitext(filename)[0]
                seq = int(stem.split("_")[-1])
                timestamp = (seq - 1) * self.interval
            except (ValueError, IndexError):
                timestamp = len(frames) * self.interval

            frames.append(FrameInfo(
                path=frame_path,
                timestamp=float(timestamp),
                format=self.format,
                size_bytes=size_bytes,
                width=width,
                height=height,
            ))
        return frames

    def _get_image_size_cv2(self, image_path: str) -> tuple:
        """用 OpenCV 读取图片尺寸

        Args:
            image_path: 图片文件路径

        Returns:
            (width, height) 元组，失败返回 (0, 0)
        """
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            return int(width), int(height)
        return 0, 0

    def _get_write_params(self) -> list:
        """获取 OpenCV 写图参数"""
        fmt_lower = self.format.lower()
        if fmt_lower in ("jpg", "jpeg"):
            q = max(0, min(100, int(self.quality)))
            return [int(cv2.IMWRITE_JPEG_QUALITY), q]
        if fmt_lower == "png":
            # quality 1-100 映射到 PNG 压缩等级 0-9（100 质量最高→0 压缩最低）
            level = max(0, min(9, int(9 - (self.quality / 100) * 9)))
            return [int(cv2.IMWRITE_PNG_COMPRESSION), level]
        return []

    def extract_frame_at(self, video_path: str, timestamp: float, output_path: str) -> str:
        """在指定时间点提取单帧（用于题目截图）

        使用 ffmpeg 单点截帧保证时间精度，不参与 dHash 去重。

        Args:
            video_path: 输入视频文件路径
            timestamp: 提取时间点（秒）
            output_path: 输出图片路径

        Returns:
            输出图片路径

        Raises:
            InvalidVideoError: 视频文件不存在
            TimestampOutOfRangeError: 时间戳为负数
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
