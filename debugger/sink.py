"""
DebugSink：debug 产物的统一写入入口。

设计文档：11_debug模块设计.md §四
依赖注入：pipeline 通过 __init__(debugger=...) 接收，None 时跳过所有 debug 调用。
release 退出：config.debug.enabled=False + 删除 debugger/ 包，pipeline 无 import 依赖不崩。
"""

import json
import os
import shutil
from typing import Any, List, Optional

from utils.file_utils import ensure_dir, save_text, save_json
from utils.logger import setup_logger
from utils.models import (
    RawTranscript, AlignedTranscript, KnowledgePoint, Problem,
)
from debugger.formatter import DebugFormatter

logger = setup_logger("debugger")


# 8 类产物子目录名（与设计文档 §3.1 一致）
DIR_ASR_RAW = "01_ASR原始逐字稿"
DIR_CORRECTED = "02_纠错后全文"
DIR_KNOWLEDGE_SEG = "03_知识点文字段"
DIR_PROBLEM_SEG = "04_题目文字段"
DIR_LOCATE_LOG = "05_定位记录"
DIR_KP_SCREENSHOT = "06_知识点截图"
DIR_Q_SCREENSHOT = "07_题目原题截图"
DIR_SOLUTION_SCREENSHOT = "08_解题过程截图"

# 截图 category → 子目录映射
SCREENSHOT_CATEGORIES = {
    "knowledge": DIR_KP_SCREENSHOT,
    "question": DIR_Q_SCREENSHOT,
    "solution": DIR_SOLUTION_SCREENSHOT,
}


class DebugSink:
    """debug 产物统一写入器

    所有 save_xxx 方法均幂等：可重复调用，后调覆盖先调（除 jsonl 是追加）。
    调用方在 _stage_probe 后必须先调 set_video_name() 确定视频名。
    """

    def __init__(self, debug_root: str, video_name: str = ""):
        """
        Args:
            debug_root: debug 根目录（如 ./debug）
            video_name: 视频名（不含扩展名），为空时需后续调 set_video_name
        """
        self._root = debug_root
        self._video_name = video_name
        self._video_dir = os.path.join(debug_root, video_name) if video_name else debug_root

    @property
    def video_dir(self) -> str:
        """debug/{视频名}/"""
        return self._video_dir

    def set_video_name(self, video_name: str) -> None:
        """在 _stage_probe 后调用，确定 debug/{视频名}/ 目录"""
        self._video_name = video_name
        self._video_dir = os.path.join(self._root, video_name)
        ensure_dir(self._video_dir)
        logger.info(f"[debugger] 视频目录: {self._video_dir}")

    # === 1. ASR 原始逐字稿 ===
    def save_asr_raw(self, transcript: RawTranscript) -> None:
        """1. ASR 原始逐字稿：json + txt 双格式"""
        if transcript is None or not transcript.text:
            logger.warning("[debugger] ASR 原始逐字稿为空，跳过")
            return
        d = os.path.join(self._video_dir, DIR_ASR_RAW)
        save_json(DebugFormatter.asr_raw_to_json(transcript), os.path.join(d, "asr_raw.json"))
        save_text(DebugFormatter.asr_raw_to_readable(transcript), os.path.join(d, "asr_readable.txt"))
        logger.info(f"[debugger] 1.ASR原始逐字稿: {len(transcript.text)}字")

    # === 2. 纠错后全文 ===
    def save_corrected_text(self, aligned: AlignedTranscript) -> None:
        """2. 纠错后全文：json + txt 双格式"""
        if aligned is None or not aligned.text:
            logger.warning("[debugger] 纠错后全文为空，跳过")
            return
        d = os.path.join(self._video_dir, DIR_CORRECTED)
        save_json(DebugFormatter.corrected_to_json(aligned), os.path.join(d, "corrected.json"))
        save_text(DebugFormatter.corrected_to_readable(aligned), os.path.join(d, "corrected.txt"))
        logger.info(f"[debugger] 2.纠错后全文: {len(aligned.text)}字")

    # === 3. 知识点文字段 ===
    def save_knowledge_segments(self, kps: List[KnowledgePoint]) -> None:
        """3. 知识点文字段：每段一个 txt"""
        if not kps:
            logger.warning("[debugger] 知识点列表为空，跳过")
            return
        d = os.path.join(self._video_dir, DIR_KNOWLEDGE_SEG)
        for kp in kps:
            path = os.path.join(d, f"知识点{kp.index:02d}.txt")
            save_text(DebugFormatter.knowledge_segment_to_text(kp), path)
        logger.info(f"[debugger] 3.知识点文字段: {len(kps)}个")

    # === 4. 题目文字段 ===
    def save_problem_segments(self, problems: List[Problem]) -> None:
        """4. 题目文字段：每段一个 txt"""
        if not problems:
            logger.warning("[debugger] 题目列表为空，跳过")
            return
        d = os.path.join(self._video_dir, DIR_PROBLEM_SEG)
        for p in problems:
            path = os.path.join(d, f"题目{p.index:02d}.txt")
            save_text(DebugFormatter.problem_segment_to_text(p), path)
        logger.info(f"[debugger] 4.题目文字段: {len(problems)}段")

    # === 5. 定位记录 ===
    def save_locate_record(
        self,
        segment_text: str, strategy: str, confidence: float,
        start_time: float, end_time: float,
        start_idx: int, end_idx: int, keyword: str = "",
    ) -> None:
        """5. 定位记录：追加到 jsonl（每次定位一条）"""
        record = DebugFormatter.locate_record_to_dict(
            segment_text, strategy, confidence,
            start_time, end_time, start_idx, end_idx, keyword,
        )
        path = os.path.join(self._video_dir, DIR_LOCATE_LOG, "locate_log.jsonl")
        ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # === 6/7/8. 截图 ===
    def save_screenshot(
        self,
        category: str, index: int, src_path: str,
        timestamp: float, step: int = 0,
    ) -> str:
        """6/7/8. 截图：复制 src 到 debug 目录

        Args:
            category: 'knowledge' / 'question' / 'solution'
            index: 知识点/题目序号（1-based）
            src_path: 源图片路径（screenshot_capture 已写入的临时位置）
            timestamp: 帧时间戳（秒）
            step: 解题步骤号（仅 category='solution' 时有效，1-based）

        Returns:
            debug 中的目标路径；复制失败返回空字符串
        """
        if category not in SCREENSHOT_CATEGORIES:
            logger.warning(f"[debugger] 未知截图 category: {category}")
            return ""

        if not src_path or not os.path.exists(src_path):
            logger.warning(f"[debugger] 截图源不存在: {src_path}")
            return ""

        target_dir = os.path.join(self._video_dir, SCREENSHOT_CATEGORIES[category])
        ts = self._format_t(timestamp)
        if category == "knowledge":
            filename = f"知识点{index:02d}_t={ts}.jpg"
        elif category == "question":
            filename = f"题目{index:02d}_t={ts}.jpg"
        else:  # solution
            filename = f"题目{index:02d}_步骤{step:02d}_t={ts}.jpg"

        target_path = os.path.join(target_dir, filename)

        # 若 src 已在 target_path（screenshot_capture 直接写到 debug 子目录），无需复制
        if os.path.normpath(src_path) == os.path.normpath(target_path):
            return target_path

        ensure_dir(target_dir)
        try:
            shutil.copy2(src_path, target_path)
        except (OSError, IOError) as e:
            logger.warning(f"[debugger] 截图复制失败 {src_path} -> {target_path}: {e}")
            return ""
        return target_path

    @staticmethod
    def _format_t(seconds: float) -> str:
        """秒数转文件名时间戳：05m23s 或 01h05m23s（与 screenshot_capture 一致）"""
        total = int(max(0.0, seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02d}h{m:02d}m{s:02d}s"
        return f"{m:02d}m{s:02d}s"
