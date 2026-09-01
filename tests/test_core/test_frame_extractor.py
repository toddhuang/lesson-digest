"""core/frame_extractor.py 测试：参数 + 错误路径 + 缓存恢复

extract_frames / extract_frame_at 需要真实视频文件，本测试只覆盖参数构造、
错误路径、缓存恢复等不依赖真实视频的部分。
"""

import os

import pytest

from core.frame_extractor import FrameExtractor
from utils.exceptions import InvalidVideoError, TimestampOutOfRangeError


class TestInit:
    def test_defaults(self):
        e = FrameExtractor()
        assert e.interval == 1.0
        assert e.format == "jpg"
        assert e.quality == 90
        assert e.dedup_threshold == 0.02
        assert e.enable_dedup is True

    def test_custom_params(self):
        e = FrameExtractor(
            interval=2.5, fmt="png", quality=80,
            dedup_threshold=0.05, enable_dedup=False,
        )
        assert e.interval == 2.5
        assert e.format == "png"
        assert e.quality == 80
        assert e.dedup_threshold == 0.05
        assert e.enable_dedup is False

    def test_interval_cast_to_float(self):
        e = FrameExtractor(interval=2)
        assert isinstance(e.interval, float)
        assert e.interval == 2.0

    def test_deduplicator_initialized(self):
        e = FrameExtractor(dedup_threshold=0.05)
        assert e.deduplicator is not None
        assert e.deduplicator.threshold == 0.05


class TestGetWriteParams:
    def test_jpg_returns_jpeg_quality(self):
        e = FrameExtractor(fmt="jpg", quality=85)
        params = e._get_write_params()
        assert len(params) == 2
        # 第一个是 IMWRITE_JPEG_QUALITY 标识
        import cv2
        assert params[0] == int(cv2.IMWRITE_JPEG_QUALITY)
        assert params[1] == 85

    def test_jpeg_alias(self):
        e = FrameExtractor(fmt="jpeg", quality=50)
        params = e._get_write_params()
        import cv2
        assert params[0] == int(cv2.IMWRITE_JPEG_QUALITY)
        assert params[1] == 50

    def test_png_returns_compression_level(self):
        e = FrameExtractor(fmt="png", quality=100)
        params = e._get_write_params()
        import cv2
        assert params[0] == int(cv2.IMWRITE_PNG_COMPRESSION)
        # quality=100 → level = 9 - (100/100)*9 = 0
        assert params[1] == 0

    def test_png_low_quality_high_compression(self):
        e = FrameExtractor(fmt="png", quality=0)
        params = e._get_write_params()
        # quality=0 → level = 9 - 0 = 9
        assert params[1] == 9

    def test_quality_clamped(self):
        # 200 应 clamp 到 100
        e = FrameExtractor(fmt="jpg", quality=200)
        params = e._get_write_params()
        assert params[1] == 100

    def test_unknown_format_returns_empty(self):
        e = FrameExtractor(fmt="bmp")
        params = e._get_write_params()
        assert params == []


class TestExtractFramesErrorPath:
    def test_nonexistent_video_raises(self, tmp_debug_dir):
        e = FrameExtractor()
        with pytest.raises(InvalidVideoError):
            e.extract_frames("/nonexistent/video.mp4", tmp_debug_dir)


class TestExtractFrameAtErrorPath:
    def test_nonexistent_video_raises(self, tmp_debug_dir):
        e = FrameExtractor()
        with pytest.raises(InvalidVideoError):
            e.extract_frame_at("/nonexistent/video.mp4", 5.0, os.path.join(tmp_debug_dir, "out.jpg"))

    def test_negative_timestamp_raises(self, tmp_debug_dir):
        e = FrameExtractor()
        # 视频不存在先于时间戳检查，所以用任意不存在的视频
        with pytest.raises((InvalidVideoError, TimestampOutOfRangeError)):
            e.extract_frame_at("/nonexistent/video.mp4", -1.0, os.path.join(tmp_debug_dir, "out.jpg"))


class TestBuildFrameListFromDisk:
    def test_recovers_frame_list_from_cached_files(self, tmp_debug_dir):
        """模拟已有缓存帧文件，验证 _build_frame_list_from_disk 恢复"""
        import cv2
        import numpy as np
        e = FrameExtractor(interval=1.0, fmt="jpg")

        # 写 3 个假帧文件 frame_000001.jpg / frame_000002.jpg / frame_000003.jpg
        for seq in (1, 2, 3):
            path = os.path.join(tmp_debug_dir, f"frame_{seq:06d}.jpg")
            # 写一张 10x10 的真实 jpg（cv2 能读取尺寸）
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img[:] = (seq * 50, seq * 50, seq * 50)
            cv2.imwrite(path, img)

        files = sorted(f for f in os.listdir(tmp_debug_dir) if f.startswith("frame_") and f.endswith(".jpg"))
        frames = e._build_frame_list_from_disk(tmp_debug_dir, files)

        assert len(frames) == 3
        # 时间戳 = (seq-1) * interval
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 1.0
        assert frames[2].timestamp == 2.0
        # 尺寸应正确读取
        for f in frames:
            assert f.width == 10 and f.height == 10
            assert f.format == "jpg"

    def test_invalid_filename_fallbacks_to_index(self, tmp_debug_dir):
        """帧文件名格式不规范时，用列表索引计算时间戳"""
        import cv2
        import numpy as np
        e = FrameExtractor(interval=2.0, fmt="jpg")

        # 写一个不规范的文件名（不带序号）
        path = os.path.join(tmp_debug_dir, "random.jpg")
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        cv2.imwrite(path, img)

        files = ["random.jpg"]
        frames = e._build_frame_list_from_disk(tmp_debug_dir, files)
        assert len(frames) == 1
        # fallback 用 len(frames) * interval = 0 * 2 = 0
        assert frames[0].timestamp == 0.0
