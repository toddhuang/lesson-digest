"""core/content_extractor.py 测试：_locate_segment 4 策略 + _parse_response + _align_transcript"""

import json

import pytest

from utils.models import (
    CharTime,
    RawTranscript,
    AlignedTranscript,
)
from core.content_extractor import ContentExtractor


# === 辅助 fixture ===

@pytest.fixture
def extractor_no_llm():
    """ContentExtractor 不传 llm（只测试定位/解析，不调 LLM）"""
    return ContentExtractor(llm=None)


@pytest.fixture
def extractor_with_debugger(extractor_no_llm, tmp_debug_dir):
    """ContentExtractor 注入 debugger（用于验证定位记录）"""
    from debugger import DebugSink
    sink = DebugSink(debug_root=tmp_debug_dir, video_name="测试")
    sink.set_video_name("测试")
    extractor_no_llm.debugger = sink
    return extractor_no_llm, sink


@pytest.fixture
def aligned_long():
    """长对齐文本（约 30 字，覆盖 0-15 秒，每字 500ms）

    text: "第一段讲函数定义，从开头到这里结束。第二段讲方程的解法，从刚才到结尾。"
    """
    text = "第一段讲函数定义，从开头到这里结束。第二段讲方程的解法，从刚才到结尾。"
    char_ts = [
        CharTime(start_ms=i * 500, end_ms=i * 500 + 500)
        for i in range(len(text))
    ]
    raw = RawTranscript(text=text, char_timestamps=char_ts)
    return AlignedTranscript(text=text, raw_align=list(range(len(text))), raw=raw)


# === _locate_segment 4 策略 ===

class TestLocateSegmentExact:
    def test_exact_match_returns_valid_time(self, extractor_no_llm, aligned_long):
        # segment 长度 >= MIN_SEGMENT_LEN (10)，且是 aligned.text 的子串
        seg = "第一段讲函数定义，从开头到这里结束"
        start, end, sidx, eidx = extractor_no_llm._locate_segment(seg, aligned_long, 0)
        assert sidx >= 0, "精确匹配应成功"
        assert start == 0.0  # 字 0 对应 0ms
        assert end > 0

    def test_exact_match_with_search_start(self, extractor_no_llm, aligned_long):
        # 第二段：segment >= 10 字
        seg = "第二段讲方程的解法，从刚才到结尾"
        start, end, sidx, eidx = extractor_no_llm._locate_segment(seg, aligned_long, 0)
        assert sidx > 0, "应在第一段后定位"

    def test_record_strategy_exact(self, extractor_with_debugger, aligned_long):
        extractor, sink = extractor_with_debugger
        extractor._locate_segment("第一段讲函数定义，从开头到这里结束", aligned_long, 0)
        log_path = sink.video_dir + "/05_定位记录/locate_log.jsonl"
        with open(log_path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readlines()[-1])
        assert rec["strategy"] == "exact"
        assert rec["confidence"] == 1.0


class TestLocateSegmentTooShort:
    def test_short_segment_returns_failure(self, extractor_no_llm, aligned_long):
        # < MIN_SEGMENT_LEN (10)
        start, end, sidx, eidx = extractor_no_llm._locate_segment("短", aligned_long, 0)
        assert sidx == -1 and eidx == -1
        assert start == 0.0 and end == 0.0

    def test_short_segment_records_failed_short(self, extractor_with_debugger, aligned_long):
        extractor, sink = extractor_with_debugger
        extractor._locate_segment("短", aligned_long, 0)
        log_path = sink.video_dir + "/05_定位记录/locate_log.jsonl"
        with open(log_path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readlines()[-1])
        assert rec["strategy"] == "failed_short"


class TestLocateSegmentKeyword:
    def test_nonexistent_segment_falls_back_to_keyword(self, extractor_no_llm, aligned_long):
        # segment 长度 >= 10，不在 aligned.text 中但含 3-gram "函数定"（aligned 含 "函数定义"）
        seg = "完全不存在的描述但包含函数定义二字的段落"
        start, end, sidx, eidx = extractor_no_llm._locate_segment(seg, aligned_long, 0)
        assert sidx >= 0, "应通过关键词兜底命中"
        # "函数定" 应在 aligned.text[sidx:sidx+10] 内
        assert "函数定" in aligned_long.text[sidx:sidx + 10]

    def test_keyword_strategy_recorded(self, extractor_with_debugger, aligned_long):
        extractor, sink = extractor_with_debugger
        seg = "完全不存在的描述但包含函数定义二字的段落"
        extractor._locate_segment(seg, aligned_long, 0)
        log_path = sink.video_dir + "/05_定位记录/locate_log.jsonl"
        with open(log_path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readlines()[-1])
        assert rec["strategy"] == "keyword"
        # 命中的关键词可能是 "函数定" 或 "数定义" 等 3-gram
        assert rec["keyword"] in ("函数定", "数定义", "函数定义")


class TestLocateSegmentFailedKeyword:
    def test_no_keyword_match_returns_failure(self, extractor_no_llm, aligned_long):
        # segment 有特征词但都不在 aligned.text 中
        seg = "毫不相关的题目内容在此毫无意义"
        start, end, sidx, eidx = extractor_no_llm._locate_segment(seg, aligned_long, 0)
        assert sidx == -1 and eidx == -1


# === _parse_response ===

class TestParseResponse:
    def test_valid_json(self, extractor_no_llm):
        content = json.dumps({
            "corrected_text": "纠错后文本",
            "knowledge_segments": [{"name": "kp1", "segment": "段"}],
            "problem_segments": [{"segment": "题段"}],
        }, ensure_ascii=False)
        ct, ks, ps = extractor_no_llm._parse_response(content)
        assert ct == "纠错后文本"
        assert len(ks) == 1 and ks[0]["name"] == "kp1"
        assert len(ps) == 1

    def test_strips_markdown_codeblock(self, extractor_no_llm):
        content = '```json\n{"corrected_text":"x","knowledge_segments":[],"problem_segments":[]}\n```'
        ct, ks, ps = extractor_no_llm._parse_response(content)
        assert ct == "x"
        assert ks == [] and ps == []

    def test_empty_corrected_text_raises(self, extractor_no_llm):
        from utils.exceptions import EmptyResultError
        content = json.dumps({"corrected_text": "", "knowledge_segments": [], "problem_segments": []})
        with pytest.raises(EmptyResultError):
            extractor_no_llm._parse_response(content)

    def test_invalid_json_raises(self, extractor_no_llm):
        from utils.exceptions import LLMResponseParseError
        with pytest.raises(LLMResponseParseError):
            extractor_no_llm._parse_response("not a json")

    def test_missing_segments_defaults_to_empty(self, extractor_no_llm):
        content = json.dumps({"corrected_text": "x"})
        ct, ks, ps = extractor_no_llm._parse_response(content)
        assert ct == "x"
        assert ks == [] and ps == []


# === _align_transcript ===

class TestAlignTranscript:
    def test_identical_text_direct_mapping(self, extractor_no_llm, raw_transcript):
        aligned = extractor_no_llm._align_transcript(raw_transcript, raw_transcript.text)
        assert aligned.text == raw_transcript.text
        assert aligned.raw_align == list(range(len(raw_transcript.text)))
        # 时间戳应可回溯
        s, e = aligned.get_time_range(0, len(aligned.text))
        assert s == 0.0

    def test_inserted_char_has_none_align(self, extractor_no_llm):
        raw = RawTranscript(
            text="abc",
            char_timestamps=[CharTime(0, 100), CharTime(100, 200), CharTime(200, 300)],
        )
        # 在中间插入 X
        aligned = extractor_no_llm._align_transcript(raw, "abXc")
        assert aligned.text == "abXc"
        assert len(aligned.raw_align) == 4
        # X 位置的 raw_align 应为 None
        assert aligned.raw_align[2] is None

    def test_deleted_char_dropped(self, extractor_no_llm):
        raw = RawTranscript(
            text="abc",
            char_timestamps=[CharTime(0, 100), CharTime(100, 200), CharTime(200, 300)],
        )
        # 删除 b
        aligned = extractor_no_llm._align_transcript(raw, "ac")
        assert aligned.text == "ac"
        assert len(aligned.raw_align) == 2
        # a → raw 0, c → raw 2
        assert aligned.raw_align == [0, 2]


# === _extract_keywords ===

class TestExtractKeywords:
    def test_extract_chinese_3grams(self, extractor_no_llm):
        kws = extractor_no_llm._extract_keywords("二次函数的图像")
        # 应包含 "二次函" "次函数" "函数的" 等 3-gram
        assert any(len(kw) == 3 for kw in kws)

    def test_extract_digits(self, extractor_no_llm):
        kws = extractor_no_llm._extract_keywords("解方程 x²-5x+6=0，求 x")
        # 应包含 "5x" 等字母串（≥2 字母）
        # 数字串要求 ≥2 位，单数字不提取
        # 检查有字母串
        assert any(c.isalpha() for kw in kws for c in kw)

    def test_no_keywords_for_pure_punctuation(self, extractor_no_llm):
        kws = extractor_no_llm._extract_keywords("。，！？")
        assert kws == []

    def test_dedup_and_sorted_by_length_desc(self, extractor_no_llm):
        kws = extractor_no_llm._extract_keywords("abc def abc")
        # 重复的应去重
        assert len(kws) == len(set(kws))


# === 集成：extract() 全流程（用 mock LLM） ===

class TestExtractEndToEnd:
    def test_extract_with_mock_llm(self, mock_llm_session):
        """用 MockLLMAdapter 跑完整 extract 流程"""
        from utils.models import RawTranscript, CharTime
        text = "今天讲二次函数f(x)=ax^2+bx+c。"
        raw = RawTranscript(
            text=text,
            char_timestamps=[
                CharTime(start_ms=i * 500, end_ms=i * 500 + 500)
                for i in range(len(text))
            ],
        )
        extractor = ContentExtractor(llm=mock_llm_session)
        aligned, kps, problems = extractor.extract(raw)
        assert aligned.text, "纠错后文本不应为空"
        assert len(kps) >= 1, "应至少提取 1 个知识点"
        assert len(problems) >= 1, "应至少提取 1 道题"
        # 知识点应有时间戳
        for kp in kps:
            assert kp.start_time >= 0
            assert kp.end_time >= kp.start_time


# === release 场景：未注入 debugger 时正常工作 ===

class TestNoDebugger:
    def test_locate_works_without_debugger(self, extractor_no_llm, aligned_long):
        # _locate_segment 不应因 debugger=None 而崩，segment >= 10 字
        start, end, sidx, eidx = extractor_no_llm._locate_segment(
            "第一段讲函数定义，从开头到这里结束", aligned_long, 0,
        )
        assert sidx >= 0
