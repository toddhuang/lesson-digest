"""adapters/llm/mock.py 测试：MockLLMAdapter 各 prompt 分支响应"""

import json

import pytest

from utils.models import LLMResponse, TokenUsage


class TestGenerateSignature:
    def test_returns_llm_response(self, mock_llm_adapter):
        resp = mock_llm_adapter.generate(prompt="default", payload="hello", temperature=0.3)
        assert isinstance(resp, LLMResponse)
        assert resp.model == "mock-model"
        assert resp.finish_reason == "stop"
        assert isinstance(resp.usage, TokenUsage)

    def test_usage_tokens_estimated_from_length(self, mock_llm_adapter):
        resp = mock_llm_adapter.generate(prompt="default", payload="hello", temperature=0.3)
        # prompt_tokens = len(payload) // 2 = 5//2 = 2
        assert resp.usage.prompt_tokens == 2
        assert resp.usage.total_tokens > 0


class TestCorrectAndExtractBranch:
    """一次性纠错+知识点+题目段提取（AGENTS.md 约定的合并调用）"""

    def test_corrected_text_branch(self, mock_llm_adapter):
        prompt = "请按 corrected_text 格式一次完成纠错+知识点+题目段"
        payload = "今天讲二次函数。"
        resp = mock_llm_adapter.generate(prompt=prompt, payload=payload, temperature=0.1)
        data = json.loads(resp.content)
        assert "corrected_text" in data
        assert data["corrected_text"] == payload
        assert "knowledge_segments" in data and "problem_segments" in data
        assert len(data["knowledge_segments"]) >= 1
        assert len(data["problem_segments"]) >= 1


class TestSolutionSummaryBranch:
    """解题过程整理（09 设计 issue #13）"""

    def test_returns_steps_array(self, mock_llm_adapter):
        prompt = "你是解题过程整理助手，请综合 ASR+OCR 输出步骤"
        resp = mock_llm_adapter.generate(prompt=prompt, payload="题目1", temperature=0.3)
        data = json.loads(resp.content)
        assert isinstance(data, list)
        assert len(data) >= 1
        step = data[0]
        assert "step_number" in step
        assert "content" in step
        assert "start_time" in step
        assert "end_time" in step
        # 公式应包含 LaTeX
        assert "$" in step["content"], "解题步骤应包含 LaTeX 公式"


class TestKnowledgeSummaryBranch:
    """知识点深度整理（10 设计 issue #9）"""

    def test_returns_content_and_supplement(self, mock_llm_adapter):
        prompt = "你是知识点深度整理助手，请综合 ASR+OCR 整理"
        resp = mock_llm_adapter.generate(prompt=prompt, payload="二次函数", temperature=0.3)
        data = json.loads(resp.content)
        assert "content" in data and "supplement" in data
        assert "f(x)=ax^2" in data["content"], "content 应含公式 LaTeX"
        assert "补充" in data["supplement"] or "顶点" in data["supplement"]


class TestMindmapBranch:
    def test_returns_opml_xml(self, mock_llm_adapter):
        prompt = "请生成思维导图 OPML 格式"
        resp = mock_llm_adapter.generate(prompt=prompt, payload="知识点", temperature=0.3)
        assert resp.content.startswith("<?xml")
        assert "<opml" in resp.content
        assert "<outline" in resp.content


class TestKnowledgeListBranch:
    def test_returns_knowledge_array(self, mock_llm_adapter):
        prompt = "请识别所有知识点"
        resp = mock_llm_adapter.generate(prompt=prompt, payload="全程文本", temperature=0.3)
        data = json.loads(resp.content)
        assert isinstance(data, list)
        assert len(data) >= 1
        kp = data[0]
        for key in ("index", "name", "start_time", "confidence"):
            assert key in kp, f"知识点缺字段 {key}"


class TestProblemListBranch:
    def test_returns_problem_array(self, mock_llm_adapter):
        prompt = "请识别所有题目/习题"
        resp = mock_llm_adapter.generate(prompt=prompt, payload="全文", temperature=0.3)
        data = json.loads(resp.content)
        assert isinstance(data, list)
        assert len(data) >= 1
        p = data[0]
        for key in ("index", "start_time", "end_time", "question_text", "solution_steps"):
            assert key in p, f"题目缺字段 {key}"


class TestDefaultBranch:
    def test_unknown_prompt_returns_default(self, mock_llm_adapter):
        resp = mock_llm_adapter.generate(prompt="unknown", payload="x", temperature=0.3)
        assert resp.content == "这是一个mock响应。"
