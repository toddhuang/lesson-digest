"""
视频信息探测工具
使用 ffprobe 探测视频文件信息。
对应文档：M14 工具集
"""

import os
import json
import subprocess
from typing import Optional

from utils.models import VideoInfo
from utils.exceptions import InvalidVideoError, FFmpegError
from utils.logger import setup_logger

logger = setup_logger("video_probe")


def probe_video(video_path: str) -> VideoInfo:
    """探测视频文件信息

    使用 ffprobe 探测视频的时长、分辨率、帧率、编码、音轨信息等。

    Args:
        video_path: 视频文件路径

    Returns:
        VideoInfo 对象

    Raises:
        InvalidVideoError: 视频文件不存在或无效
        FFmpegError: ffprobe 执行失败
    """
    if not os.path.exists(video_path):
        raise InvalidVideoError(f"视频文件不存在: {video_path}")

    file_size = os.path.getsize(video_path)

    # 调用 ffprobe
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise FFmpegError("ffprobe 未找到，请确认 ffmpeg 已安装并加入 PATH")
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"ffprobe 执行超时: {video_path}")

    if result.returncode != 0:
        raise FFmpegError(f"ffprobe 执行失败: {result.stderr}", stderr=result.stderr, returncode=result.returncode)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise FFmpegError(f"ffprobe 输出解析失败: {e}")

    # 解析视频流
    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    # 解析格式信息
    format_info = data.get("format", {})
    duration = float(format_info.get("duration", 0))

    # 构建 VideoInfo
    info = VideoInfo(
        path=video_path,
        duration=duration,
        width=int(video_stream.get("width", 0)) if video_stream else 0,
        height=int(video_stream.get("height", 0)) if video_stream else 0,
        fps=_parse_fps(video_stream.get("r_frame_rate", "0/0")) if video_stream else 0.0,
        video_codec=video_stream.get("codec_name", "") if video_stream else "",
        audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
        audio_channels=int(audio_stream.get("channels", 0)) if audio_stream else 0,
        audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        size_bytes=file_size,
        has_audio=audio_stream is not None,
        has_video=video_stream is not None,
    )

    logger.info(f"视频探测完成: {info.width}x{info.height}, {info.duration:.1f}s, "
                f"视频={info.video_codec}, 音频={info.audio_codec}")
    return info


def _parse_fps(rate_str: str) -> float:
    """解析帧率字符串（如 "30000/1001" → 29.97）

    Args:
        rate_str: 帧率字符串，格式为 "num/den"

    Returns:
        帧率（fps）
    """
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            num = float(num)
            den = float(den)
            if den == 0:
                return 0.0
            return round(num / den, 3)
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def has_audio_stream(video_path: str) -> bool:
    """检测视频是否包含音轨

    Args:
        video_path: 视频文件路径

    Returns:
        True 表示包含音轨
    """
    try:
        info = probe_video(video_path)
        return info.has_audio
    except Exception:
        return False
