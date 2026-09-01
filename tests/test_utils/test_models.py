"""utils/models.py 测试：dataclass to_dict/from_dict + get_time_range"""

import pytest

from utils.models import (
    CharTime,
    RawTranscript,
    AlignedTranscript,
    OCRResult,
    OCRFrameResult,
    KnowledgePoint,
    Problem,
    SolutionStep,
)


class TestCharTime:
    def test_to_dict(self):
        ct = CharTime(start_ms=100, end_ms=500)
        assert ct.to_dict() == {"start_ms": 100, "end_ms": 500}

    def test_from_dict(self):
        ct = CharTime.from_dict({"start_ms": 200, "end_ms": 600})
        assert ct.start_ms == 200 and ct.end_ms == 600

    def test_roundtrip(self):
        original = CharTime(start_ms=1000, end_ms=2000)
        d = original.to_dict()
        restored = CharTime.from_dict(d)
        assert restored == original


class TestRawTranscript:
    def test_to_dict_basic(self):
        text = "ab"
        raw = RawTranscript(
            text=text,
            char_timestamps=[CharTime(0, 100), CharTime(100, 200)],
        )
        d = raw.to_dict()
        assert d["text"] == text
        assert len(d["char_timestamps"]) == 2
        assert d["char_timestamps"][0] == {"start_ms": 0, "end_ms": 100}

    def test_to_dict_with_none(self):
        # 标点位置为 None
        text = "a,"
        raw = RawTranscript(
            text=text,
            char_timestamps=[CharTime(0, 100), None],
        )
        d = raw.to_dict()
        assert d["char_timestamps"][1] is None

    def test_from_dict_roundtrip(self):
        text = "abc"
        raw = RawTranscript(
            text=text,
            char_timestamps=[
                CharTime(0, 100),
                None,
                CharTime(200, 300),
            ],
        )
        d = raw.to_dict()
        restored = RawTranscript.from_dict(d)
        assert restored.text == text
        assert len(restored.char_timestamps) == 3
        assert restored.char_timestamps[1] is None
        assert restored.char_timestamps[2] == CharTime(200, 300)

    def test_get_time_range_normal(self):
        # 字 0-2 (索引) → 0-300ms → 0.0-0.3s
        raw = RawTranscript(
            text="abc",
            char_timestamps=[
                CharTime(0, 100),
                CharTime(100, 200),
                CharTime(200, 300),
            ],
        )
        start, end = raw.get_time_range(0, 3)
        assert start == 0.0 and end == 0.3

    def test_get_time_range_skip_none(self):
        # 中间 None 跳过
        raw = RawTranscript(
            text="a,b",
            char_timestamps=[
                CharTime(0, 100),
                None,
                CharTime(200, 300),
            ],
        )
        start, end = raw.get_time_range(0, 3)
        assert start == 0.0 and end == 0.3

    def test_get_time_range_empty_returns_zero(self):
        raw = RawTranscript(text="ab", char_timestamps=[])
        start, end = raw.get_time_range(0, 2)
        assert start == 0.0 and end == 0.0


class TestAlignedTranscript:
    def test_get_time_range_direct_mapping(self):
        text = "abc"
        raw = RawTranscript(
            text=text,
            char_timestamps=[
                CharTime(0, 100),
                CharTime(100, 200),
                CharTime(200, 300),
            ],
        )
        aligned = AlignedTranscript(
            text=text,
            raw_align=[0, 1, 2],
            raw=raw,
        )
        start, end = aligned.get_time_range(0, 3)
        assert start == 0.0 and end == 0.3

    def test_get_time_range_skip_none_align(self):
        # raw_align 中 None 表示 LLM 新增字，跳过
        text = "abc"
        raw = RawTranscript(
            text="ac",
            char_timestamps=[
                CharTime(0, 100),
                CharTime(200, 300),
            ],
        )
        aligned = AlignedTranscript(
            text=text,  # "abc"，b 是新增字
            raw_align=[0, None, 1],
            raw=raw,
        )
        start, end = aligned.get_time_range(0, 3)
        # 字 0 → raw 0 → 0ms；字 2 → raw 1 → 200-300ms
        assert start == 0.0 and end == 0.3

    def test_get_time_range_no_raw(self):
        aligned = AlignedTranscript(text="abc", raw_align=None, raw=None)
        start, end = aligned.get_time_range(0, 3)
        assert start == 0.0 and end == 0.0


class TestOCRResult:
    def test_default_text_type(self):
        r = OCRResult(text="hello")
        assert r.block_type == "text"
        assert r.latex == ""

    def test_formula_type(self):
        r = OCRResult(text="", block_type="formula", latex="x^2+1")
        assert r.block_type == "formula"
        assert r.latex == "x^2+1"


class TestKnowledgePoint:
    def test_default_empty_fields(self):
        kp = KnowledgePoint(index=1, name="x", start_time=0.0, end_time=10.0)
        assert kp.content == ""
        assert kp.supplement == ""


class TestProblemAndSolution:
    def test_problem_with_steps(self):
        steps = [
            SolutionStep(step_number=1, content="step1", start_time=10.0, end_time=15.0),
            SolutionStep(step_number=2, content="step2", start_time=15.0, end_time=20.0),
        ]
        p = Problem(
            index=1, start_time=10.0, end_time=20.0,
            question_text="q", solution_steps=steps,
        )
        assert len(p.solution_steps) == 2
        assert p.solution_steps[0].start_time == 10.0
        assert p.solution_steps[1].end_time == 20.0

    def test_problem_default_no_steps(self):
        p = Problem(index=1, start_time=0.0, end_time=10.0, question_text="q")
        assert p.solution_steps == []
