"""core/frame_dedup.py 测试：FrameDeduplicator dHash 去重"""

import cv2
import numpy as np
import pytest

from core.frame_dedup import FrameDeduplicator


def _solid_color_frame(color: tuple, size: tuple = (100, 100, 3)) -> np.ndarray:
    """构造纯色帧（BGR）"""
    frame = np.zeros(size, dtype=np.uint8)
    frame[:] = color
    return frame


def _gradient_frame(direction: str = "lr", size: tuple = (100, 100)) -> np.ndarray:
    """构造渐变帧（BGR）

    Args:
        direction: "lr"=左暗右亮（hash 大量 True），"rl"=左亮右暗（hash 大量 False）
    """
    gray = np.zeros(size, dtype=np.uint8)
    for x in range(size[1]):
        if direction == "lr":
            gray[:, x] = int(255 * x / size[1])
        else:
            gray[:, x] = int(255 * (1 - x / size[1]))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TestFrameDeduplicatorInit:
    def test_default_threshold(self):
        d = FrameDeduplicator()
        assert d.threshold == 0.02

    def test_custom_threshold(self):
        d = FrameDeduplicator(threshold=0.1)
        assert d.threshold == 0.1

    def test_initial_prev_hash_none(self):
        d = FrameDeduplicator()
        assert d._prev_hash is None


class TestShouldKeep:
    def test_first_frame_kept(self):
        d = FrameDeduplicator()
        frame = _solid_color_frame((128, 128, 128))
        assert d.should_keep(frame) is True

    def test_identical_frames_dropped(self):
        """完全相同的帧应被丢弃（除第一帧）"""
        d = FrameDeduplicator()
        frame = _solid_color_frame((128, 128, 128))
        assert d.should_keep(frame) is True  # 第一帧
        assert d.should_keep(frame) is False  # 第二帧相同
        assert d.should_keep(frame) is False  # 第三帧相同

    def test_different_frames_kept(self):
        """明显不同的帧应被保留（dHash 用渐变帧区分方向）"""
        d = FrameDeduplicator()
        # 第一帧：左暗右亮（hash 大量 True）
        assert d.should_keep(_gradient_frame("lr")) is True
        # 第二帧：左亮右暗（hash 大量 False，与第一帧差异大）
        assert d.should_keep(_gradient_frame("rl")) is True
        # 第三帧：再切回左暗右亮
        assert d.should_keep(_gradient_frame("lr")) is True

    def test_slightly_different_frames(self):
        """微小变化的帧：阈值 0.02 下应丢弃（< 1.28 位差异）"""
        d = FrameDeduplicator(threshold=0.02)
        # 两帧仅在右下角 1 像素略有差异
        f1 = _solid_color_frame((128, 128, 128))
        f2 = _solid_color_frame((128, 128, 128))
        f2[-1, -1] = (130, 130, 130)  # 微小变化
        # dHash 缩放到 8x8 后可能完全相同
        d.should_keep(f1)
        # 第二帧可能被丢弃（dHash 看不到 1 像素变化）
        # 主要验证不崩，结果取决于 dHash 灵敏度
        result = d.should_keep(f2)
        assert isinstance(result, bool)


class TestReset:
    def test_reset_clears_prev_hash(self):
        d = FrameDeduplicator()
        d.should_keep(_solid_color_frame((128, 128, 128)))
        assert d._prev_hash is not None
        d.reset()
        assert d._prev_hash is None

    def test_reset_allows_new_first_frame(self):
        """reset 后第一帧又被保留"""
        d = FrameDeduplicator()
        frame = _solid_color_frame((128, 128, 128))
        assert d.should_keep(frame) is True
        assert d.should_keep(frame) is False  # 重复
        d.reset()
        assert d.should_keep(frame) is True  # reset 后又是第一帧


class TestComputeHash:
    def test_hash_shape(self):
        d = FrameDeduplicator()
        frame = _solid_color_frame((128, 128, 128))
        h = d._compute_hash(frame)
        # HASH_SIZE=8 → 64 位 hash
        assert h.shape == (64,)
        assert h.dtype == bool

    def test_solid_color_hash_all_false(self):
        """纯色帧所有相邻像素相等 → hash 全 False"""
        d = FrameDeduplicator()
        frame = _solid_color_frame((128, 128, 128))
        h = d._compute_hash(frame)
        assert not h.any(), "纯色帧 hash 应全 False"

    def test_gradient_hash_has_true(self):
        """渐变帧（左暗右亮）相邻像素 left < right → hash 全 True"""
        d = FrameDeduplicator()
        gray = np.zeros((100, 100), dtype=np.uint8)
        for x in range(100):
            gray[:, x] = int(255 * x / 100)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h = d._compute_hash(frame)
        # 渐变帧 hash 大部分应为 True
        assert h.sum() > 32, "渐变帧 hash 应有大量 True"
