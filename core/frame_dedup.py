"""
M3 关键帧提取模块
从视频中按固定间隔提取关键帧，通过帧变化检测去重，
去重后保留帧交 M5 做 OCR。

R-009 定案：dHash 差异哈希，阈值 0.02。
对应文档：03_接口设计/M3_关键帧提取模块接口.md
"""

import os
from typing import List, Tuple

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("M3_frame_dedup")


class FrameDeduplicator:
    """帧去重器，基于 dHash 差异哈希"""

    HASH_SIZE = 8  # dHash 输出 8x8 哈希，共 64 位

    def __init__(self, threshold: float = 0.02):
        """初始化

        Args:
            threshold: 差异分数阈值（0~1），低于此阈值视为相同帧。
                       0.02 = 64 位哈希中仅 1-2 位差异即判定为变化
        """
        self.threshold = threshold
        self._prev_hash: np.ndarray | None = None

    def _compute_hash(self, frame: np.ndarray) -> np.ndarray:
        """计算 dHash 哈希

        Args:
            frame: BGR 格式的帧图像

        Returns:
            64 位布尔数组，True 表示相邻像素前者 > 后者
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.HASH_SIZE + 1, self.HASH_SIZE),
                           interpolation=cv2.INTER_AREA)
        return (small[:, 1:] > small[:, :-1]).flatten()

    def compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """计算两帧的差异分数（汉明距离 / 哈希长度）

        Args:
            prev: 前一帧 BGR 图像
            curr: 当前帧 BGR 图像

        Returns:
            差异分数 0~1，0 表示完全相同，1 表示完全不同
        """
        h1 = self._compute_hash(prev)
        h2 = self._compute_hash(curr)
        return float(np.count_nonzero(h1 != h2)) / self.HASH_SIZE ** 2

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
        score = float(np.count_nonzero(self._prev_hash != curr_hash)) / self.HASH_SIZE ** 2
        keep = score >= self.threshold
        if keep:
            self._prev_hash = curr_hash
        return keep

    def reset(self) -> None:
        """重置去重器状态，用于处理新视频"""
        self._prev_hash = None


class FrameExtractor:
    """关键帧提取器：1 秒间隔抽帧 + dHash 去重"""

    def __init__(self, interval_sec: float = 1.0,
                 dedup_threshold: float = 0.02):
        """初始化

        Args:
            interval_sec: 抽帧间隔（秒），默认 1.0
            dedup_threshold: dHash 去重阈值，默认 0.02
        """
        self.interval_sec = interval_sec
        self.deduplicator = FrameDeduplicator(threshold=dedup_threshold)

    def extract(self, video_path: str, output_dir: str,
                output_format: str = "png") -> Tuple[List[str], List[float]]:
        """从视频中提取关键帧

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            output_format: 输出图片格式（png/jpg）

        Returns:
            (frame_paths, frame_timestamps) 元组，路径与时间戳一一对应
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * self.interval_sec)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[M3] 开始抽帧: fps={fps:.2f}, 间隔={self.interval_sec}s, "
                     f"步长={step}帧, 总帧数={total_frames}")

        os.makedirs(output_dir, exist_ok=True)
        self.deduplicator.reset()

        frame_paths: List[str] = []
        frame_timestamps: List[float] = []
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

            if self.deduplicator.should_keep(frame):
                fname = f"{kept_count + 1:05d}.{output_format}"
                fpath = os.path.join(output_dir, fname)
                cv2.imwrite(fpath, frame)
                frame_paths.append(fpath)
                frame_timestamps.append(timestamp)
                kept_count += 1

            idx += step

        cap.release()

        logger.info(f"[M3] 抽帧完成: 抽样={sampled_count}帧, 保留={kept_count}帧, "
                     f"压缩比={(sampled_count - kept_count) / sampled_count * 100:.1f}%")

        return frame_paths, frame_timestamps
