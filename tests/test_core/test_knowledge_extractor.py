"""core/knowledge_extractor.py 测试：enrich_knowledge + extract

从 scripts/test_enrich_knowledge.py 改造为 pytest。
"""

import pytest

from utils.models import KnowledgePoint
from core.knowledge_extractor import (
    KnowledgeExtractor,
    KNOWLEDGE_SUMMARY_PROMPT,
)


# === KnowledgePoint 字段（K1） ===

class TestKnowledgePointFields:
    def test_extended_fields_present(self):
        kp = KnowledgePoint(index=1, name="x", start_time=0.0, end_time=10.0)
        assert hasattr(kp, "end_time"), "缺 end_time 字段"
        assert hasattr(kp, "content"), "缺 content 字段"
        assert hasattr(kp, "supplement"), "缺 supplement 字段"

    def test_default_empty_strings(self):
        kp = KnowledgePoint(index=1, name="x", start_time=0.0, end_time=10.0)
        assert kp.content == "" and kp.supplement == ""


# === _locate_knowledge_points 补存 end_time（K2） ===

class TestLocateKnowledgePoints:
    def test_locate_fills_end_time(self):
        from utils.models import CharTime, RawTranscript, AlignedTranscript
        from core.content_extractor import ContentExtractor

        text = "第一段讲函数定义，从开头到这里结束。第二段讲方程的解法，从刚才到结尾。"
        raw = RawTranscript(
            text=text,
            char_timestamps=[
                CharTime(start_ms=i * 1000, end_ms=i * 1000 + 1000)
                for i in range(len(text))
            ],
        )
        aligned = AlignedTranscript(text=text, raw_align=list(range(len(text))), raw=raw)

        extractor = ContentExtractor(llm=None)
        segments = [
            {"name": "函数定义", "segment": "第一段讲函数定义，从开头到这里结束。"},
            {"name": "方程解法", "segment": "第二段讲方程的解法，从刚才到结尾。"},
        ]
        kps = extractor._locate_knowledge_points(segments, aligned)
        assert len(kps) == 2, f"应定位 2 个知识点，实际 {len(kps)}"
        for kp in kps:
            assert kp.end_time > kp.start_time, \
                f"知识点{kp.index} end_time({kp.end_time}) 应 > start_time({kp.start_time})"
            assert kp.end_time > 0, f"知识点{kp.index} end_time 应非零"


# === enrich_knowledge（K3 + K5） ===

class TestEnrichKnowledge:
    def test_with_mock_llm(self, mock_llm_session, aligned_50s, sample_ocr_results):
        """enrich_knowledge 调 mock LLM，验证 content+supplement 解析"""
        extractor = KnowledgeExtractor(llm=None, summary_llm=mock_llm_session)
        kp = KnowledgePoint(
            index=1, name="二次函数的图像性质",
            start_time=0.0, end_time=25.0, confidence=0.9,
        )
        # 先验证 mock 对该 prompt 返回正确 JSON
        raw_resp = mock_llm_session.generate(
            prompt=KNOWLEDGE_SUMMARY_PROMPT, payload="test",
        ).content
        import json
        data = json.loads(raw_resp)
        assert "content" in data and "supplement" in data

        # 执行 enrich_knowledge
        result = extractor.enrich_knowledge(kp, aligned_50s, sample_ocr_results)
        assert result.content, "enrich_knowledge 后 content 不应为空"
        assert result.supplement, "enrich_knowledge 后 supplement 不应为空"
        assert "f(x)=ax^2" in result.content or "f(x)" in result.content, \
            f"content 应含公式 LaTeX，实际: {result.content[:80]}"
        assert "补充" in result.supplement or "顶点" in result.supplement, \
            f"supplement 应为补充内容，实际: {result.supplement[:80]}"

    def test_empty_segments_skipped(self, mock_llm_session, aligned_50s, sample_ocr_results):
        """enrich_knowledge 在 ASR/OCR 都空时安全跳过"""
        extractor = KnowledgeExtractor(llm=None, summary_llm=mock_llm_session)
        # 时间范围 100-200s 与 aligned_50s(0-50s) 不重叠
        kp = KnowledgePoint(index=1, name="空", start_time=100.0, end_time=200.0)
        result = extractor.enrich_knowledge(kp, aligned_50s, sample_ocr_results)
        assert result.content == "" and result.supplement == "", \
            "ASR/OCR 都空时应跳过，content/supplement 应保持空"

    def test_no_summary_llm_skipped(self):
        """未配置 summary_llm 时跳过"""
        extractor = KnowledgeExtractor(llm=None, summary_llm=None)
        kp = KnowledgePoint(index=1, name="x", start_time=0.0, end_time=5.0)
        # 不应报错
        result = extractor.enrich_knowledge(kp, None, None)
        assert result.content == "" and result.supplement == ""


# === extract（知识点列表提取） ===

class TestExtract:
    def test_empty_text_does_not_crash(self, mock_llm_session):
        """mock LLM 对空 text 仍返回知识点列表（不抛异常），验证 extract 不崩"""
        extractor = KnowledgeExtractor(llm=mock_llm_session, summary_llm=None)
        # 空 text 不应导致崩溃
        kps = extractor.extract("", video_duration=60.0)
        assert isinstance(kps, list)
        # mock 返回 5 个知识点（按 prompt 内容分支）
        assert len(kps) >= 1

    def test_extract_returns_list(self, mock_llm_session):
        extractor = KnowledgeExtractor(llm=mock_llm_session, summary_llm=None)
        full_text = "全程文本：今天讲二次函数。"
        kps = extractor.extract(full_text, video_duration=60.0)
        assert isinstance(kps, list)
        assert len(kps) >= 1
        for kp in kps:
            assert isinstance(kp, KnowledgePoint)
            assert kp.start_time >= 0
            assert kp.name


# === pipeline 调度（K4） ===

class TestPipelineStageWired:
    def test_stages_contains_summarize_knowledge(self):
        from core.pipeline import STAGES
        assert "summarize_knowledge" in STAGES, "STAGES 缺 summarize_knowledge"
        assert "summarize_solution" in STAGES, "STAGES 缺 summarize_solution"

    def test_run_stage_calls_summarize_knowledge(self):
        """run() 实际调用顺序：summarize_knowledge 应在 correct_and_extract 之后、merge_text 之前"""
        import os
        from core.pipeline import STAGES
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "core", "pipeline.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        # run() 调用顺序
        run_block_start = src.find('self._run_stage("correct_and_extract"')
        run_block_end = src.find('self._run_stage("merge_text"')
        assert run_block_start > 0 and run_block_end > 0, "未找到 run() 中的相关 _run_stage 调用"
        between = src[run_block_start:run_block_end]
        assert 'summarize_solution' in between, "run() 漏调 summarize_solution"
        assert 'summarize_knowledge' in between, "run() 漏调 summarize_knowledge"
        # _run_stage 调度分支
        assert 'elif stage == "summarize_solution":' in src, "_run_stage 缺 summarize_solution 分支"
        assert 'elif stage == "summarize_knowledge":' in src, "_run_stage 缺 summarize_knowledge 分支"
