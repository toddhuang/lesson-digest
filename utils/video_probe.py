"""
视频信息探测工具
mock阶段返回假数据，真实运行时使用 ffprobe
"""

import os
from utils.models import VideoInfo
from utils.exceptions import InvalidVideoError


def probe_video(video_path: str) -> VideoInfo:
    """探测视频文件信息（mock版）

    Args:
        video_path: 视频文件路径

    Returns:
        VideoInfo 对象

    Raises:
        InvalidVideoError: 视频文件不存在或无效
    """
    if not os.path.exists(video_path):
        raise InvalidVideoError(f"视频文件不存在: {video_path}")

    # mock数据：返回一个45分钟的1080p视频
    file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 500 * 1024 * 1024

    return VideoInfo(
        path=video_path,
        duration=2700.0,  # 45分钟
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=44100,
        size_bytes=file_size,
        has_audio=True,
        has_video=True,
    )


def has_audio_stream(video_path: str) -> bool:
    """检测视频是否包含音轨（mock版）"""
    return True
