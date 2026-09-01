"""debugger/sink.py attach_log_handler 测试：第 9 类运行日志归档"""

import logging
import os
import shutil
import tempfile

import pytest

from utils.logger import setup_logger, set_log_file
from utils.models import CharTime, RawTranscript
from debugger import DebugSink


@pytest.fixture
def log_sink(tmp_debug_dir):
    """DebugSink 实例 + 切换前先初始化全局 file handler（避免 None handler 错误）"""
    # 先在 logs/ 写一条日志，确保全局 file handler 已初始化
    pre_logger = setup_logger("PreAttach")
    pre_logger.info("[test] 切换前的日志（应出现在 logs/）")

    sink = DebugSink(debug_root=tmp_debug_dir, video_name="测试视频")
    sink.set_video_name("测试视频")
    yield sink

    # 恢复默认 logs/ 路径，避免影响后续测试
    set_log_file(os.path.join(tempfile.gettempdir(), "test_restore.log"))


class TestAttachLogHandler:
    def test_returns_true_when_switch_successful(self, log_sink):
        ok = log_sink.attach_log_handler()
        assert ok is True

    def test_multi_module_logs_archived_to_same_file(self, log_sink):
        """切换后多模块 logger 写入同一 pipeline.log"""
        log_sink.attach_log_handler()

        log_a = setup_logger("ModuleA")
        log_b = setup_logger("ModuleB")
        log_a.info("[test] 模块A的日志")
        log_b.warning("[test] 模块B的警告")

        # 强制 flush
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass

        log_path = os.path.join(log_sink.video_dir, "09_运行日志", "pipeline.log")
        assert os.path.exists(log_path), f"日志文件应存在: {log_path}"
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "模块A的日志" in content
        assert "模块B的警告" in content
        assert "ModuleA" in content and "ModuleB" in content

    def test_debugger_own_log_also_archived(self, log_sink, raw_transcript):
        """debugger 自身的 logger 输出也归档到 pipeline.log"""
        log_sink.attach_log_handler()
        log_sink.save_asr_raw(raw_transcript)  # 内部 logger.info("[debugger] 1.ASR原始逐字稿...")

        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass

        log_path = os.path.join(log_sink.video_dir, "09_运行日志", "pipeline.log")
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "[debugger] 1.ASR原始逐字稿" in content, "debugger 自身的 log 应归档"
