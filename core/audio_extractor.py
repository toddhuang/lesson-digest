"""
M2 音轨提取模块
从视频中提取音轨，转换为 ASR 要求的格式。
mock阶段不调用ffmpeg，直接创建空WAV文件占位。
对应文档：03_接口设计/M2_音轨提取模块接口.md
"""

import os
from utils.models import AudioInfo
from utils.file_utils import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("M2_audio")


class AudioExtractor:
    """音轨提取器"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def extract_audio(self, video_path: str, output_path: str) -> AudioInfo:
        """从视频中提取音轨（mock版）

        Args:
            video_path: 输入视频文件路径
            output_path: 输出 WAV 文件路径

        Returns:
            AudioInfo 对象
        """
        logger.info(f"[M2] 提取音轨: {video_path} -> {output_path}")
        ensure_dir(os.path.dirname(output_path))

        # mock: 创建一个空文件占位
        if not os.path.exists(output_path):
            with open(output_path, "wb") as f:
                f.write(b"RIFF....WAVEfmt ")  # 最小WAV文件头占位

        file_size = os.path.getsize(output_path)

        info = AudioInfo(
            path=output_path,
            duration=2700.0,  # 45分钟（mock）
            sample_rate=self.sample_rate,
            channels=self.channels,
            size_bytes=file_size,
        )
        logger.info(f"[M2] 音轨提取完成: {info.duration}s, {info.sample_rate}Hz")
        return info

    def has_audio(self, video_path: str) -> bool:
        """检测视频是否包含音轨（mock版）"""
        return True
