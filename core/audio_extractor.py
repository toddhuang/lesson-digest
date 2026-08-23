"""
M2 音轨提取模块
从视频中提取音轨，转换为 ASR 要求的格式（16kHz，单声道，WAV）。
使用 ffmpeg 命令行工具。
对应文档：03_接口设计/M2_音轨提取模块接口.md
"""

import os
import subprocess
from utils.models import AudioInfo
from utils.file_utils import ensure_dir
from utils.logger import setup_logger
from utils.exceptions import FFmpegError, InvalidVideoError

logger = setup_logger("M2_audio")


class AudioExtractor:
    """音轨提取器"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def extract_audio(self, video_path: str, output_path: str) -> AudioInfo:
        """从视频中提取音轨并转换为指定格式

        使用 ffmpeg 提取音轨，转换为 16kHz 单声道 WAV 格式（FunASR 要求）。

        Args:
            video_path: 输入视频文件路径
            output_path: 输出 WAV 文件路径

        Returns:
            AudioInfo 对象

        Raises:
            InvalidVideoError: 视频文件不存在
            FFmpegError: ffmpeg 执行失败
        """
        if not os.path.exists(video_path):
            raise InvalidVideoError(f"视频文件不存在: {video_path}")

        logger.info(f"[M2] 提取音轨: {video_path} -> {output_path}")
        ensure_dir(os.path.dirname(output_path))

        # 如果输出已存在，跳过（缓存机制）
        if os.path.exists(output_path):
            logger.info(f"[M2] 音轨已存在，跳过提取: {output_path}")
            return self._build_audio_info(output_path)

        # ffmpeg 命令：提取音轨，转换为 16kHz 单声道 WAV
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                    # 不处理视频流
            "-acodec", "pcm_s16le",  # PCM 16位小端编码（WAV标准）
            "-ar", str(self.sample_rate),  # 采样率
            "-ac", str(self.channels),     # 声道数
            "-y",                     # 覆盖已存在文件
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            raise FFmpegError("ffmpeg 未找到，请确认 ffmpeg 已安装并加入 PATH")
        except subprocess.TimeoutExpired:
            raise FFmpegError(f"ffmpeg 执行超时: {video_path}")

        if result.returncode != 0:
            raise FFmpegError(
                f"ffmpeg 音轨提取失败: {result.stderr[-500:]}",
                stderr=result.stderr,
                returncode=result.returncode
            )

        info = self._build_audio_info(output_path)
        logger.info(f"[M2] 音轨提取完成: {info.duration:.1f}s, {info.sample_rate}Hz, {info.channels}声道")
        return info

    def _build_audio_info(self, output_path: str) -> AudioInfo:
        """构建 AudioInfo 对象，使用 ffprobe 获取音频时长"""
        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        # 使用 ffprobe 获取音频时长
        duration = 0.0
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
        except Exception:
            # ffprobe 失败时使用文件大小估算（16kHz 16bit 单声道 = 32000 bytes/秒）
            if file_size > 44:  # 减去 WAV 头
                duration = (file_size - 44) / (self.sample_rate * self.channels * 2)

        return AudioInfo(
            path=output_path,
            duration=duration,
            sample_rate=self.sample_rate,
            channels=self.channels,
            size_bytes=file_size,
        )

    def has_audio(self, video_path: str) -> bool:
        """检测视频是否包含音轨

        Args:
            video_path: 视频文件路径

        Returns:
            True 表示包含音轨
        """
        from utils.video_probe import has_audio_stream
        return has_audio_stream(video_path)
