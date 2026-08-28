"""帧去重算法对比测试

对 tests/test.mp4 按 1 秒间隔抽帧，相邻帧比对：
- 无变化（差异分数 < threshold）→ 保留前一帧，本帧不输出
- 有变化（差异分数 >= threshold）→ 输出到对应算法目录

4 个算法：
1. dHash   - 差异哈希 8x8（汉明距离 / 64）
2. pHash   - DCT 感知哈希 32x32（汉明距离 / 64）
3. absdiff - 帧差法（变化像素数 / 总像素数）
4. SSIM    - 结构相似性（1 - ssim）

输出目录：
    temp/dhash/00001.png
    temp/phash/00001.png
    temp/absdiff/00001.png
    temp/ssim/00001.png

用法：
    venv\\Scripts\\python.exe scripts\\frame_dedup_test.py
    venv\\Scripts\\python.exe scripts\\frame_dedup_test.py --threshold 0.05
    venv\\Scripts\\python.exe scripts\\frame_dedup_test.py --algorithms dhash,absdiff

依赖：opencv-python、scikit-image、numpy（venv 已就绪）
"""

import argparse
import shutil
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VIDEO_PATH = PROJECT_DIR / "tests" / "test.mp4"
OUTPUT_ROOT = PROJECT_DIR / "temp"
FRAME_INTERVAL_SEC = 1.0


class BaseComparator(ABC):
    """帧差异比较器抽象基类。"""

    name: str = "base"

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.prev_frame = None

    @abstractmethod
    def _compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """返回差异分数 0~1，越大越不同。"""

    def should_emit(self, curr: np.ndarray) -> bool:
        if self.prev_frame is None:
            return True
        score = self._compute_score(self.prev_frame, curr)
        return score >= self.threshold

    def update(self, frame: np.ndarray) -> None:
        self.prev_frame = frame


class DHashComparator(BaseComparator):
    """dHash 差异哈希 8x8：相邻像素亮度差，汉明距离 / 64。"""

    name = "dhash"

    def _to_hash(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        return (small[:, 1:] > small[:, :-1]).flatten()

    def _compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        return float(np.count_nonzero(self._to_hash(prev) != self._to_hash(curr))) / 64.0


class PHashComparator(BaseComparator):
    """pHash DCT 感知哈希 32x32：DCT 后取左上 8x8 均值比较。"""

    name = "phash"

    def _to_hash(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(small))
        low = dct[:8, :8]
        mean = low.mean()
        return (low > mean).flatten()

    def _compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        return float(np.count_nonzero(self._to_hash(prev) != self._to_hash(curr))) / 64.0


class AbsDiffComparator(BaseComparator):
    """帧差法：高斯模糊后 absdiff，二值化阈值 25，变化像素比例。"""

    name = "absdiff"

    def _compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        prev_blur = cv2.GaussianBlur(prev, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr, (5, 5), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        return float(cv2.countNonZero(mask)) / gray.size


class SSIMComparator(BaseComparator):
    """SSIM 结构相似性：1 - ssim，gray 单通道。"""

    name = "ssim"

    def _compute_score(self, prev: np.ndarray, curr: np.ndarray) -> float:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        score = sk_ssim(prev_gray, curr_gray, data_range=255)
        return 1.0 - float(score)


COMPARATORS = {
    DHashComparator.name: DHashComparator,
    PHashComparator.name: PHashComparator,
    AbsDiffComparator.name: AbsDiffComparator,
    SSIMComparator.name: SSIMComparator,
}


def build_comparators(names: list[str], thresholds: dict[str, float]) -> list[BaseComparator]:
    return [COMPARATORS[name](thresholds[name]) for name in names]


def iter_sampled_frames(video_path: Path, interval_sec: float):
    """按固定时间间隔抽帧，yield (frame_idx_in_video, timestamp_sec, frame_bgr)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval_sec)))
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, idx / fps, frame
        idx += step
    cap.release()


def reset_output_dirs(names: list[str]) -> dict[str, Path]:
    """清空并重建每个算法的输出目录。"""
    paths = {}
    for name in names:
        d = OUTPUT_ROOT / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        paths[name] = d
    return paths


def run(names: list[str], thresholds: dict[str, float], save_image: bool) -> int:
    if not VIDEO_PATH.exists():
        print(f"视频不存在：{VIDEO_PATH}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps else 0
    print(f"[info] 视频：{VIDEO_PATH.name}  fps={fps:.2f}  frames={total}  duration={duration:.1f}s")
    print(f"[info] 抽帧间隔：{FRAME_INTERVAL_SEC}s  预计抽样：{int(duration / FRAME_INTERVAL_SEC)} 帧")
    print(f"[info] 阈值：{thresholds}  算法：{','.join(names)}  保存图像：{save_image}")
    print()

    out_dirs = reset_output_dirs(names) if save_image else {}
    comparators = build_comparators(names, thresholds)
    counters = {c.name: 0 for c in comparators}
    sampled = 0
    start = time.perf_counter()

    for frame_idx, ts, frame in iter_sampled_frames(VIDEO_PATH, FRAME_INTERVAL_SEC):
        sampled += 1
        for c in comparators:
            if c.should_emit(frame):
                counters[c.name] += 1
                if save_image:
                    fname = f"{counters[c.name]:05d}.png"
                    cv2.imwrite(str(out_dirs[c.name] / fname), frame)
            c.update(frame)
        if sampled % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"[progress] sampled={sampled}  elapsed={elapsed:.1f}s  counts={counters}")

    elapsed = time.perf_counter() - start
    print()
    print("=" * 60)
    print(f"抽样总帧数：{sampled}  耗时：{elapsed:.1f}s")
    print()
    print(f"{'算法':<10} {'保留帧':<10} {'压缩比':<12} {'占比'}")
    print("-" * 60)
    for name in names:
        kept = counters[name]
        ratio = (sampled - kept) / sampled * 100 if sampled else 0
        pct = kept / sampled * 100 if sampled else 0
        print(f"{name:<10} {kept:<10} {ratio:.1f}%        {pct:.1f}%")
    print()

    if save_image:
        print("输出目录：")
        for name in names:
            d = out_dirs[name]
            size_mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1024 / 1024
            print(f"  {d}  ({counters[name]} 帧, {size_mb:.1f} MB)")

    return 0


DEFAULT_THRESHOLDS: dict[str, float] = {
    "dhash": 0.02,
    "phash": 0.10,
    "absdiff": 0.05,
    "ssim": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="帧去重算法对比测试")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(COMPARATORS.keys()),
        help=f"算法组合，逗号分隔，默认全部：{','.join(COMPARATORS.keys())}",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help=(
            "每算法独立阈值，格式：dhash=0.02,phash=0.10,absdiff=0.05,ssim=0.05\n"
            f"默认值：{DEFAULT_THRESHOLDS}"
        ),
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="不保存图像，只输出统计（用于快速对比）",
    )
    return parser.parse_args()


def parse_thresholds(raw: str | None) -> dict[str, float]:
    if not raw:
        return dict(DEFAULT_THRESHOLDS)
    result = dict(DEFAULT_THRESHOLDS)
    for part in raw.split(","):
        part = part.strip()
        if "=" not in part:
            raise ValueError(f"阈值格式错误：{part}，应为 name=value")
        name, val = part.split("=", 1)
        name = name.strip()
        if name not in COMPARATORS:
            raise ValueError(f"未知算法：{name}，可选：{','.join(COMPARATORS.keys())}")
        result[name] = float(val.strip())
    return result


def main() -> int:
    args = parse_args()
    names = [n.strip() for n in args.algorithms.split(",") if n.strip() in COMPARATORS]
    if not names:
        print(f"无效算法：{args.algorithms}", file=sys.stderr)
        return 1
    try:
        thresholds = parse_thresholds(args.thresholds)
    except ValueError as e:
        print(f"阈值解析错误：{e}", file=sys.stderr)
        return 1
    thresholds = {k: v for k, v in thresholds.items() if k in names}
    return run(names, thresholds, save_image=not args.no_image)


if __name__ == "__main__":
    sys.exit(main())
