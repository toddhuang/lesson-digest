"""验证知识点深度整理（issue #9，10 设计）的端到端链路。

复用 MockLLMAdapter，覆盖：
1. KnowledgePoint.end_time/content/supplement 字段已就位
2. content_extractor._locate_knowledge_points 已补存 end_time
3. knowledge_extractor.enrich_knowledge 融合 ASR+OCR 调 LLM 并解析
4. mock.py 对 "知识点深度整理助手" prompt 返回正确 JSON
5. pipeline STAGES 含 summarize_knowledge 且 run()/_run_stage 调度覆盖
"""

import json
import os
import sys

# 项目根目录加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.models import (
    KnowledgePoint, RawTranscript, CharTime, AlignedTranscript,
    OCRFrameResult, OCRResult,
)
from core.knowledge_extractor import KnowledgeExtractor, KNOWLEDGE_SUMMARY_PROMPT
from core.pipeline import STAGES
from adapters.llm.mock import MockLLMAdapter
from core.llm.llm_session import LLMSession


def make_mock_session(temperature: float = 0.3) -> LLMSession:
    """构造绑定 temperature 的 mock LLM 会话（实现 LLMGenerator 协议）"""
    return LLMSession(adapter=MockLLMAdapter(), temperature=temperature, model_name="mock")


def build_aligned():
    """构造 aligned（纠错后文本 + raw 字级时间戳），覆盖 0-50 秒"""
    text = "今天讲二次函数f(x)=ax^2+bx+c的图像性质。开口方向由a决定，a为正开口向上。"
    char_ts = []
    for i, _ in enumerate(text):
        # 每字 0.5 秒
        char_ts.append(CharTime(start_ms=i * 500, end_ms=i * 500 + 500))
    raw = RawTranscript(text=text, char_timestamps=char_ts)
    return AlignedTranscript(text=text, raw_align=list(range(len(text))), raw=raw)


def build_ocr_results():
    """构造 OCR 帧结果（含 1 个公式帧 + 1 个文本帧，落在知识点时间范围 0-25s）"""
    formula_frame = OCRFrameResult(
        timestamp=5.0,
        image_path="/tmp/f1.jpg",
        results=[
            OCRResult(text="", confidence=0.9, block_type="formula", latex="f(x)=ax^2+bx+c"),
        ],
        full_text="",
        is_duplicate=False,
    )
    text_frame = OCRFrameResult(
        timestamp=10.0,
        image_path="/tmp/f2.jpg",
        results=[
            OCRResult(text="二次函数图像性质", confidence=0.95, block_type="text"),
        ],
        full_text="二次函数图像性质",
        is_duplicate=False,
    )
    return [formula_frame, text_frame]


def test_models_fields():
    """K1: KnowledgePoint 字段检查"""
    kp = KnowledgePoint(index=1, name="二次函数", start_time=0.0, end_time=25.0)
    assert hasattr(kp, "end_time"), "KnowledgePoint 缺 end_time 字段"
    assert hasattr(kp, "content"), "KnowledgePoint 缺 content 字段"
    assert hasattr(kp, "supplement"), "KnowledgePoint 缺 supplement 字段"
    assert kp.content == "" and kp.supplement == "", "新字段默认应为空字符串"
    print("[PASS] K1 KnowledgePoint 字段已就位")


def test_content_extractor_locate_end_time():
    """K2: _locate_knowledge_points 补存 end_time"""
    from core.content_extractor import ContentExtractor

    # 构造 aligned.text，含两个明确知识点段
    text = "第一段讲函数定义，从开头到这里结束。第二段讲方程的解法，从刚才到结尾。"
    raw = RawTranscript(text=text, char_timestamps=[
        CharTime(start_ms=i * 1000, end_ms=i * 1000 + 1000) for i in range(len(text))
    ])
    aligned = AlignedTranscript(text=text, raw_align=list(range(len(text))), raw=raw)

    extractor = ContentExtractor(llm=None)  # 不调 LLM，只验证定位
    segments = [
        {"name": "函数定义", "segment": "第一段讲函数定义，从开头到这里结束。"},
        {"name": "方程解法", "segment": "第二段讲方程的解法，从刚才到结尾。"},
    ]
    kps = extractor._locate_knowledge_points(segments, aligned)
    assert len(kps) == 2, f"应定位 2 个知识点，实际 {len(kps)}"
    for kp in kps:
        assert kp.end_time > kp.start_time, \
            f"知识点 {kp.index} end_time({kp.end_time}) 应 > start_time({kp.start_time})"
        assert kp.end_time > 0, f"知识点 {kp.index} end_time 应非零，实际 {kp.end_time}"
    print(f"[PASS] K2 _locate_knowledge_points 补存 end_time: "
          f"kp1=[{kps[0].start_time:.1f}-{kps[0].end_time:.1f}], "
          f"kp2=[{kps[1].start_time:.1f}-{kps[1].end_time:.1f}]")


def test_enrich_knowledge_with_mock():
    """K3 + K5: enrich_knowledge 调 mock LLM，验证 content+supplement 解析"""
    session = make_mock_session(temperature=0.3)
    extractor = KnowledgeExtractor(llm=None, summary_llm=session)

    kp = KnowledgePoint(
        index=1, name="二次函数的图像性质",
        start_time=0.0, end_time=25.0, confidence=0.9,
    )
    aligned = build_aligned()
    ocr_results = build_ocr_results()

    # 确认 mock 对该 prompt 返回正确 JSON
    raw_resp = session.generate(prompt=KNOWLEDGE_SUMMARY_PROMPT, payload="test").content
    data = json.loads(raw_resp)
    assert "content" in data and "supplement" in data, "mock 响应缺 content/supplement"

    # 执行 enrich_knowledge
    result = extractor.enrich_knowledge(kp, aligned, ocr_results)

    assert result.content, "enrich_knowledge 后 content 不应为空"
    assert result.supplement, "enrich_knowledge 后 supplement 不应为空"
    assert "f(x)=ax^2" in result.content or "f(x)" in result.content, \
        f"content 应含公式 LaTeX，实际: {result.content[:80]}"
    assert "补充" in result.supplement or "顶点" in result.supplement, \
        f"supplement 应为补充内容，实际: {result.supplement[:80]}"
    print(f"[PASS] K3+K5 enrich_knowledge: content {len(result.content)}字, "
          f"supplement {len(result.supplement)}字")
    print(f"      content 预览: {result.content[:60]}...")
    print(f"      supplement 预览: {result.supplement[:60]}...")


def test_enrich_knowledge_empty_segments():
    """enrich_knowledge 在 ASR/OCR 都空时安全跳过"""
    session = make_mock_session(temperature=0.3)
    extractor = KnowledgeExtractor(llm=None, summary_llm=session)

    kp = KnowledgePoint(index=1, name="空知识点", start_time=100.0, end_time=200.0)
    aligned = build_aligned()  # 0-50s，与 100-200s 不重叠
    ocr_results = build_ocr_results()  # 5s/10s，与 100-200s 不重叠

    result = extractor.enrich_knowledge(kp, aligned, ocr_results)
    assert result.content == "" and result.supplement == "", \
        "ASR/OCR 都空时应跳过，content/supplement 应保持空"
    print("[PASS] enrich_knowledge ASR/OCR 均空时安全跳过")


def test_pipeline_stage_wired():
    """K4: pipeline STAGES 含 summarize_knowledge + run()/_run_stage 调度覆盖"""
    assert "summarize_knowledge" in STAGES, "STAGES 缺 summarize_knowledge"
    assert "summarize_solution" in STAGES, "STAGES 缺 summarize_solution"

    # 检查 run() 实际调用顺序：summarize_solution/summarize_knowledge 应在 correct_and_extract 之后、merge_text 之前
    src_path = os.path.join(ROOT, "core", "pipeline.py")
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

    print("[PASS] K4 pipeline.run()/_run_stage 已调度 summarize_solution + summarize_knowledge")


def main():
    print("=" * 60)
    print("issue #9 / 10 设计 - 知识点深度整理验证")
    print("=" * 60)
    test_models_fields()
    test_content_extractor_locate_end_time()
    test_enrich_knowledge_with_mock()
    test_enrich_knowledge_empty_segments()
    test_pipeline_stage_wired()
    print("=" * 60)
    print("全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
