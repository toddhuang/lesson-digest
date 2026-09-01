"""core/problem_extractor.py 测试：enrich_solution"""

import pytest

from utils.models import Problem
from core.problem_extractor import ProblemExtractor


class TestEnrichSolution:
    def test_with_mock_llm(self, mock_llm_session, aligned_50s, sample_ocr_results):
        """enrich_solution 调 mock LLM，验证 solution_steps 解析"""
        problem = Problem(
            index=1, start_time=15.0, end_time=35.0,
            question_text="解方程 x^2-5x+6=0",
            asr_question_text="解方程 x的平方减5x加6等于0",
            confidence=0.9,
        )
        extractor = ProblemExtractor(
            llm=None, solution_llm=mock_llm_session,
        )
        result = extractor.enrich_solution(problem, aligned_50s, sample_ocr_results)
        assert len(result.solution_steps) >= 1, "应有解题步骤"
        for step in result.solution_steps:
            assert step.step_number >= 1
            assert step.content
            assert step.start_time >= 0
            assert step.end_time >= step.start_time
            # 公式应包含 LaTeX（mock 返回的 content 含 $...$）
            assert "$" in step.content, f"步骤 {step.step_number} 应含 LaTeX"

    def test_empty_segments_skipped(self, mock_llm_session, aligned_50s, sample_ocr_results):
        """ASR/OCR 都空时跳过"""
        extractor = ProblemExtractor(llm=None, solution_llm=mock_llm_session)
        # 时间范围 100-200s 与 0-50s 不重叠
        problem = Problem(
            index=1, start_time=100.0, end_time=200.0,
            question_text="q", asr_question_text="asr",
        )
        original_steps_count = len(problem.solution_steps)
        result = extractor.enrich_solution(problem, aligned_50s, sample_ocr_results)
        # 跳过后 solution_steps 不变
        assert len(result.solution_steps) == original_steps_count

    def test_no_solution_llm_skipped(self):
        """未配置 solution_llm 时跳过"""
        extractor = ProblemExtractor(llm=None, solution_llm=None)
        problem = Problem(index=1, start_time=0.0, end_time=10.0,
                          question_text="q", asr_question_text="asr")
        # 不应报错
        result = extractor.enrich_solution(problem, None, None)
        assert result.solution_steps == []


class TestExtract:
    def test_extract_with_mock_llm(self, mock_llm_session):
        """extract 用 mock LLM 返回题目列表"""
        extractor = ProblemExtractor(llm=mock_llm_session, solution_llm=None)
        full_text = "今天讲一道例题，已知方程x平方减5x加6等于0，求方程的解。"
        problems = extractor.extract(full_text, video_duration=60.0)
        assert isinstance(problems, list)
        assert len(problems) >= 1
        for p in problems:
            assert isinstance(p, Problem)
            assert p.start_time >= 0
            assert p.question_text
