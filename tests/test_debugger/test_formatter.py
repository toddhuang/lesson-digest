"""debugger/formatter.py 测试：DebugFormatter 7 个静态方法"""

from utils.models import (
    CharTime,
    RawTranscript,
    AlignedTranscript,
    KnowledgePoint,
    Problem,
    SolutionStep,
)
from debugger.formatter import DebugFormatter


class TestAsrRawToJson:
    def test_returns_dict_with_text_and_timestamps(self, raw_transcript):
        d = DebugFormatter.asr_raw_to_json(raw_transcript)
        assert d["text"] == raw_transcript.text
        assert len(d["char_timestamps"]) == len(raw_transcript.text)


class TestAsrRawToReadable:
    def test_contains_header_and_per_char_lines(self, raw_transcript):
        text = DebugFormatter.asr_raw_to_readable(raw_transcript)
        assert "ASR 原始逐字稿" in text
        # 每字一行
        lines = text.split("\n")
        # 头 1 + 空行 1 + 字数行
        assert len(lines) >= len(raw_transcript.text) + 2

    def test_none_timestamp_shows_placeholder(self):
        raw = RawTranscript(
            text="a,b",
            char_timestamps=[CharTime(0, 100), None, CharTime(200, 300)],
        )
        text = DebugFormatter.asr_raw_to_readable(raw)
        assert "(None)" in text, "None 时间戳字应显示 (None) 占位"


class TestCorrectedToJson:
    def test_basic_structure(self, aligned_transcript):
        d = DebugFormatter.corrected_to_json(aligned_transcript)
        assert d["text"] == aligned_transcript.text
        assert d["raw_align"] == aligned_transcript.raw_align


class TestCorrectedToReadable:
    def test_contains_text(self, aligned_transcript):
        text = DebugFormatter.corrected_to_readable(aligned_transcript)
        assert aligned_transcript.text in text
        assert "纠错后全文" in text


class TestKnowledgeSegmentToText:
    def test_basic_with_empty_content(self):
        kp = KnowledgePoint(
            index=1, name="函数定义",
            start_time=0.0, end_time=5.5, confidence=0.9,
        )
        text = DebugFormatter.knowledge_segment_to_text(kp)
        assert "知识点01" in text
        assert "函数定义" in text
        # content 为空不应包含 "核心内容" 段
        assert "核心内容" not in text

    def test_with_content_and_supplement(self):
        kp = KnowledgePoint(
            index=2, name="图像性质",
            start_time=5.0, end_time=12.0, confidence=0.92,
            content="开口由 $a$ 决定",
            supplement="补充：顶点公式",
        )
        text = DebugFormatter.knowledge_segment_to_text(kp)
        assert "核心内容" in text
        assert "开口由 $a$ 决定" in text
        assert "补充内容" in text
        assert "顶点公式" in text

    def test_timestamps_formatted_as_mmss_cc(self):
        kp = KnowledgePoint(index=1, name="x", start_time=5.25, end_time=12.5)
        text = DebugFormatter.knowledge_segment_to_text(kp)
        assert "00:05.25" in text
        assert "00:12.50" in text


class TestProblemSegmentToText:
    def test_basic_with_asr_question_text(self):
        p = Problem(
            index=1, start_time=15.0, end_time=30.0,
            question_text="解方程 x^2-5x+6=0",
            asr_question_text="解方程 x的平方减5x加6等于0",
            confidence=0.93,
        )
        text = DebugFormatter.problem_segment_to_text(p)
        assert "题目01" in text
        # ASR 文字段优先使用 asr_question_text
        assert "x的平方减5x加6等于0" in text

    def test_fallback_to_question_text_when_asr_empty(self):
        p = Problem(
            index=1, start_time=15.0, end_time=30.0,
            question_text="解方程 x^2-5x+6=0",
            asr_question_text="",
            confidence=0.93,
        )
        text = DebugFormatter.problem_segment_to_text(p)
        assert "x^2-5x+6=0" in text

    def test_with_solution_steps(self):
        steps = [
            SolutionStep(step_number=1, content="因式分解", start_time=18.0, end_time=22.0),
            SolutionStep(step_number=2, content="令每个因式为0", start_time=22.0, end_time=28.0),
        ]
        p = Problem(
            index=1, start_time=15.0, end_time=30.0,
            asr_question_text="解方程", solution_steps=steps,
            confidence=0.93,
        )
        text = DebugFormatter.problem_segment_to_text(p)
        assert "解题步骤" in text
        assert "1." in text and "2." in text
        assert "因式分解" in text
        assert "令每个因式为0" in text


class TestLocateRecordToDict:
    def test_basic_fields(self):
        d = DebugFormatter.locate_record_to_dict(
            segment_text="测试段落",
            strategy="exact",
            confidence=1.0,
            start_time=0.0, end_time=5.0,
            start_idx=0, end_idx=10,
        )
        assert d["strategy"] == "exact"
        assert d["confidence"] == 1.0
        assert d["start_time"] == 0.0
        assert d["end_time"] == 5.0
        assert d["start_idx"] == 0
        assert d["end_idx"] == 10
        assert "keyword" not in d, "无 keyword 不应包含该字段"

    def test_with_keyword(self):
        d = DebugFormatter.locate_record_to_dict(
            segment_text="段落",
            strategy="keyword",
            confidence=0.0,
            start_time=10.0, end_time=20.0,
            start_idx=5, end_idx=15,
            keyword="公式",
        )
        assert d["keyword"] == "公式"

    def test_segment_text_truncated_to_200_chars(self):
        long_text = "a" * 300
        d = DebugFormatter.locate_record_to_dict(
            segment_text=long_text,
            strategy="exact", confidence=1.0,
            start_time=0.0, end_time=0.0,
            start_idx=0, end_idx=0,
        )
        assert len(d["segment"]) == 200

    def test_confidence_rounded_to_4_digits(self):
        d = DebugFormatter.locate_record_to_dict(
            segment_text="x",
            strategy="mid", confidence=0.123456789,
            start_time=0.0, end_time=0.0,
            start_idx=0, end_idx=0,
        )
        assert d["confidence"] == 0.1235, "应四舍五入到 4 位"
