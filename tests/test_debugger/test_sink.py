"""debugger/sink.py 测试：8 类产物写入 + 异常处理 + 截图归档"""

import json
import os

import pytest

from utils.models import (
    KnowledgePoint,
    Problem,
    SolutionStep,
)
from debugger import DebugSink


# === 测试辅助 ===

def _write_fake_jpg(path: str) -> None:
    """写一个假的 JPG 文件（足够通过 os.path.exists 检查）"""
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake JFIF\xff\xd9")


# === fixture ===

@pytest.fixture
def sink(tmp_debug_dir):
    """DebugSink 实例，video_dir = tmp_debug_dir/测试视频/"""
    s = DebugSink(debug_root=tmp_debug_dir, video_name="测试视频")
    s.set_video_name("测试视频")  # 幂等：再次调用应不报错
    return s


# === 1. ASR 原始逐字稿 ===

class TestSaveAsrRaw:
    def test_writes_json_and_readable_txt(self, sink, raw_transcript):
        sink.save_asr_raw(raw_transcript)
        json_path = os.path.join(sink.video_dir, "01_ASR原始逐字稿", "asr_raw.json")
        txt_path = os.path.join(sink.video_dir, "01_ASR原始逐字稿", "asr_readable.txt")
        assert os.path.exists(json_path)
        assert os.path.exists(txt_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["text"] == raw_transcript.text
        assert "char_timestamps" in data

    def test_empty_transcript_skipped(self, sink):
        from utils.models import RawTranscript
        sink.save_asr_raw(RawTranscript(text="", char_timestamps=[]))
        # 不应写任何文件
        d = os.path.join(sink.video_dir, "01_ASR原始逐字稿")
        assert not os.path.exists(d) or not os.listdir(d)

    def test_none_transcript_skipped(self, sink):
        sink.save_asr_raw(None)
        d = os.path.join(sink.video_dir, "01_ASR原始逐字稿")
        assert not os.path.exists(d) or not os.listdir(d)


# === 2. 纠错后全文 ===

class TestSaveCorrectedText:
    def test_writes_json_and_txt(self, sink, aligned_transcript):
        sink.save_corrected_text(aligned_transcript)
        json_path = os.path.join(sink.video_dir, "02_纠错后全文", "corrected.json")
        txt_path = os.path.join(sink.video_dir, "02_纠错后全文", "corrected.txt")
        assert os.path.exists(json_path)
        assert os.path.exists(txt_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["text"] == aligned_transcript.text

    def test_empty_skipped(self, sink):
        from utils.models import AlignedTranscript
        sink.save_corrected_text(AlignedTranscript(text=""))
        d = os.path.join(sink.video_dir, "02_纠错后全文")
        assert not os.path.exists(d) or not os.listdir(d)


# === 3. 知识点文字段 ===

class TestSaveKnowledgeSegments:
    def test_writes_one_txt_per_kp(self, sink):
        kps = [
            KnowledgePoint(index=1, name="定义", start_time=0.0, end_time=5.0,
                           content="函数 $f(x)$", supplement="补充"),
            KnowledgePoint(index=2, name="性质", start_time=5.0, end_time=12.0),
        ]
        sink.save_knowledge_segments(kps)
        for kp in kps:
            path = os.path.join(sink.video_dir, "03_知识点文字段", f"知识点{kp.index:02d}.txt")
            assert os.path.exists(path), f"缺 {path}"

    def test_empty_skipped(self, sink):
        sink.save_knowledge_segments([])
        d = os.path.join(sink.video_dir, "03_知识点文字段")
        assert not os.path.exists(d) or not os.listdir(d)


# === 4. 题目文字段 ===

class TestSaveProblemSegments:
    def test_writes_one_txt_per_problem(self, sink):
        problems = [
            Problem(index=1, start_time=15.0, end_time=30.0,
                    question_text="q1", asr_question_text="asr1"),
            Problem(index=2, start_time=30.0, end_time=50.0,
                    question_text="q2", asr_question_text="asr2",
                    solution_steps=[
                        SolutionStep(step_number=1, content="s1", start_time=32.0, end_time=35.0)
                    ]),
        ]
        sink.save_problem_segments(problems)
        for p in problems:
            path = os.path.join(sink.video_dir, "04_题目文字段", f"题目{p.index:02d}.txt")
            assert os.path.exists(path), f"缺 {path}"

    def test_empty_skipped(self, sink):
        sink.save_problem_segments([])
        d = os.path.join(sink.video_dir, "04_题目文字段")
        assert not os.path.exists(d) or not os.listdir(d)


# === 5. 定位记录（jsonl 追加） ===

class TestSaveLocateRecord:
    def test_appends_jsonl_lines(self, sink):
        sink.save_locate_record("seg1", "exact", 1.0, 0.0, 5.0, 0, 5)
        sink.save_locate_record("seg2", "keyword", 0.0, 12.0, 20.0, 10, 18, keyword="公式")
        sink.save_locate_record("seg3", "failed_short", 0.0, 0.0, 0.0, -1, -1)

        log_path = os.path.join(sink.video_dir, "05_定位记录", "locate_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3, f"jsonl 应有 3 行，实际 {len(lines)}"

        rec1 = json.loads(lines[0])
        assert rec1["strategy"] == "exact"
        rec2 = json.loads(lines[1])
        assert rec2["strategy"] == "keyword"
        assert rec2["keyword"] == "公式"

    def test_single_record(self, sink):
        sink.save_locate_record("only", "mid", 0.5, 1.0, 2.0, 0, 10)
        log_path = os.path.join(sink.video_dir, "05_定位记录", "locate_log.jsonl")
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["confidence"] == 0.5


# === 6/7/8. 截图 ===

class TestSaveScreenshot:
    @pytest.fixture
    def fake_img(self, tmp_debug_dir):
        src_dir = os.path.join(tmp_debug_dir, "_src")
        os.makedirs(src_dir, exist_ok=True)
        src_path = os.path.join(src_dir, "fake.jpg")
        _write_fake_jpg(src_path)
        return src_path

    def test_knowledge_screenshot(self, sink, fake_img):
        target = sink.save_screenshot("knowledge", 1, fake_img, timestamp=323.0)
        assert target and os.path.exists(target)
        assert "06_知识点截图" in target
        assert "知识点01_t=05m23s.jpg" in target

    def test_question_screenshot(self, sink, fake_img):
        target = sink.save_screenshot("question", 1, fake_img, timestamp=65.0)
        assert target and os.path.exists(target)
        assert "07_题目原题截图" in target
        assert "题目01_t=01m05s.jpg" in target

    def test_solution_screenshot(self, sink, fake_img):
        target = sink.save_screenshot("solution", 1, fake_img, timestamp=18.5, step=1)
        assert target and os.path.exists(target)
        assert "08_解题过程截图" in target
        assert "题目01_步骤01_t=00m18s.jpg" in target

    def test_solution_with_hour_format(self, sink, fake_img):
        target = sink.save_screenshot("solution", 1, fake_img, timestamp=3725.0, step=2)
        assert target
        # 3725s = 01:02:05 → 文件名 01h02m05s
        assert "01h02m05s" in target

    def test_unknown_category_returns_empty(self, sink, fake_img):
        target = sink.save_screenshot("unknown", 1, fake_img, timestamp=0.0)
        assert target == ""

    def test_nonexistent_source_returns_empty(self, sink):
        target = sink.save_screenshot("knowledge", 1, "/nonexistent.jpg", timestamp=0.0)
        assert target == ""

    def test_no_copy_when_src_equals_target(self, sink, fake_img):
        # 当 src 已经在 target 路径（screenshot_capture 直接写到 debug 子目录）
        target = sink.save_screenshot("knowledge", 1, fake_img, timestamp=323.0)
        # 再次用 target 作为 src 调用，不应报错也不应复制
        target2 = sink.save_screenshot("knowledge", 1, target, timestamp=323.0)
        assert target2 == target


# === 配置测试 ===

class TestConfigDebugSection:
    """config 加载 debug 段（与 scripts/test_debugger.py 对应）"""

    def test_config_has_debug_section(self, mock_config):
        from config import DebugConfig
        assert hasattr(mock_config, "debug")
        assert isinstance(mock_config.debug, DebugConfig)

    def test_debug_defaults(self, mock_config):
        assert mock_config.debug.enabled is True
        assert mock_config.debug.max_size_gb == 50.0
        assert mock_config.debug.save_intermediate is True
