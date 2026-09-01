"""
M3 帧去重器
基于 dHash 差异哈希的帧变化检测去重（R-009 定案：阈值 0.02）。
被 core/frame_extractor.py 的 FrameExtractor 通过组合方式使用。
对应文档：03_接口设计/M3_关键帧提取模块接口.md
"""

from typing import Optional

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("M3_frame_dedup")


class FrameDeduplicator:
    """dHash 差异哈希去重器（R-009 定案：阈值 0.02）

    8x8 = 64 位哈希，比较相邻帧的汉明距离/总位数，
    低于阈值视为同一画面（无变化）丢弃，高于阈值保留。
    """

    HASH_SIZE = 8  # 8x8 = 64 位哈希

    def __init__(self, threshold: float = 0.02):
        """初始化

        Args:
            threshold: 差异分数阈值（0~1），低于此阈值视为同一画面。
                       0.02 = 64 位哈希中仅 1-2 位差异即判定为无变化
        """
        self.threshold = threshold
        self._prev_hash: Optional[np.ndarray] = None

    def _compute_hash(self, frame: np.ndarray) -> np.ndarray:
        """计算 dHash 哈希

        将帧转灰度、缩放到 9x8、比较相邻列像素，
        左>右 为 True，输出 64 位布尔数组。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (self.HASH_SIZE + 1, self.HASH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return (small[:, 1:] > small[:, :-1]).flatten()

    def should_keep(self, frame: np.ndarray) -> bool:
        """判定当前帧是否应保留

        Args:
            frame: BGR 格式的帧图像

        Returns:
            True = 保留本帧（有变化），False = 丢弃本帧（无变化）
        """
        if self._prev_hash is None:
            self._prev_hash = self._compute_hash(frame)
            return True

        curr_hash = self._compute_hash(frame)
        score = float(np.count_nonzero(self._prev_hash != curr_hash)) / (self.HASH_SIZE ** 2)
        keep = score >= self.threshold
        if keep:
            self._prev_hash = curr_hash
        return keep

    def reset(self) -> None:
        """重置去重器状态，用于处理新视频"""
        self._prev_hash = None
