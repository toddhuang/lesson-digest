"""pytest 全局 fixture

被所有 test_*.py 共享。提供 mock 适配器、临时目录、样本数据等。

设计原则：
- 不依赖外部资源（无文件 IO、无网络、无 GPU）
- 不读 config.yaml，用 Config() 默认实例
- 临时目录 fixture 自动清理
"""

import os
import sys
import shutil
import tempfile

import pytest

# 项目根目录加入 sys.path，让所有测试可直接 import core/utils/adapters
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config
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
# 必须先 import core.llm（让 llm_client 完整初始化），再 import adapters.llm
# 否则 adapters/llm/__init__.py 触发 litellm_adapter -> core.llm -> llm_client -> adapters.llm.factory 循环
from core.llm.llm_session import LLMSession
from adapters.llm.mock import MockLLMAdapter


# === 配置 ===

@pytest.fixture
def mock_config() -> Config:
    """默认 Config 实例（不读 config.yaml，直接 dataclass 默认值）

    LLM 配置全空（无 providers/models），仅作 paths/asr/ocr 等模块配置使用。
    """
    return Config()


@pytest.fixture
def tmp_debug_dir():
    """临时 debug 目录，测试结束自动清理"""
    tmpdir = tempfile.mkdtemp(prefix="test_debug_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# === LLM mock ===

@pytest.fixture
def mock_llm_adapter() -> MockLLMAdapter:
    """MockLLMAdapter 单例（按 prompt 内容返回不同假数据）"""
    return MockLLMAdapter()


@pytest.fixture
def mock_llm_session(mock_llm_adapter) -> LLMSession:
    """LLMSession 包装 MockLLMAdapter，实现 LLMGenerator 协议"""
    return LLMSession(adapter=mock_llm_adapter, temperature=0.3, model_name="mock")


# === ASR/Aligned 数据 ===

@pytest.fixture
def raw_transcript() -> RawTranscript:
    """ASR 原始逐字稿（10 字，0-5 秒，每字 500ms）"""
    text = "今天讲二次函数。"
    char_ts = [
        CharTime(start_ms=i * 500, end_ms=i * 500 + 500)
        for i in range(len(text))
    ]
    return RawTranscript(text=text, char_timestamps=char_ts)


@pytest.fixture
def aligned_transcript(raw_transcript) -> AlignedTranscript:
    """纠错后对齐文本（text 与 raw_align 等长，每字直接映射）"""
    return AlignedTranscript(
        text=raw_transcript.text,
        raw_align=list(range(len(raw_transcript.text))),
        raw=raw_transcript,
    )


@pytest.fixture
def aligned_50s() -> AlignedTranscript:
    """50 秒长对齐文本，覆盖 0-50 秒（知识点/题目时间范围测试用）

    text 约 100 字，每字 500ms，覆盖 0-50s。
    """
    text = "今天讲二次函数f(x)=ax^2+bx+c的图像性质。开口方向由a决定，a为正开口向上。"
    char_ts = [
        CharTime(start_ms=i * 500, end_ms=i * 500 + 500)
        for i in range(len(text))
    ]
    raw = RawTranscript(text=text, char_timestamps=char_ts)
    return AlignedTranscript(text=text, raw_align=list(range(len(text))), raw=raw)


# === OCR 数据 ===

@pytest.fixture
def sample_ocr_results():
    """OCR 帧结果（2 帧：公式帧 + 文本帧，落在 5-10 秒）"""
    formula_frame = OCRFrameResult(
        timestamp=5.0,
        image_path="/tmp/f1.jpg",
        results=[
            OCRResult(text="", confidence=0.9, block_type="formula",
                      latex="f(x)=ax^2+bx+c"),
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


# === 知识点/题目 ===

@pytest.fixture
def sample_knowledge_points():
    """2 个知识点（覆盖 0-25 秒）"""
    return [
        KnowledgePoint(
            index=1, name="二次函数定义",
            start_time=0.0, end_time=10.0, confidence=0.95,
            content="", supplement="",
        ),
        KnowledgePoint(
            index=2, name="图像性质",
            start_time=10.0, end_time=25.0, confidence=0.92,
            content="", supplement="",
        ),
    ]


@pytest.fixture
def sample_problems():
    """1 道题（15-35 秒，含 2 个解题步骤）"""
    return [
        Problem(
            index=1, start_time=15.0, end_time=35.0,
            question_text="解方程 x^2-5x+6=0",
            asr_question_text="解方程 x的平方减5x加6等于0",
            solution_steps=[
                SolutionStep(
                    step_number=1, content="因式分解：$(x-2)(x-3)=0$",
                    start_time=18.0, end_time=22.0,
                ),
                SolutionStep(
                    step_number=2, content="令每个因式为0，得 $x=2$ 或 $x=3$",
                    start_time=22.0, end_time=28.0,
                ),
            ],
            confidence=0.93,
        ),
    ]
