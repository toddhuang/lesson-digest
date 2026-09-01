"""
统一数据模型定义
所有跨模块共享的数据结构（dataclass）统一定义于此。
对应文档：03_接口设计/00_数据模型.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# === 一、视频与音频相关 ===

@dataclass
class VideoInfo:
    """视频元信息，由 M14 工具集的 probe_video() 返回"""
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0
    size_bytes: int = 0
    has_audio: bool = False
    has_video: bool = False


@dataclass
class AudioInfo:
    """音频提取结果，由 M2 音轨提取模块返回"""
    path: str = ""
    duration: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    size_bytes: int = 0


@dataclass
class FrameInfo:
    """关键帧信息，由 M3 关键帧提取模块返回"""
    path: str = ""
    timestamp: float = 0.0
    format: str = "jpg"
    size_bytes: int = 0
    width: int = 0
    height: int = 0


# === 二、识别结果相关 ===

@dataclass
class Sentence:
    """ASR 语音识别结果（单句）- 旧结构，逐步废弃"""
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    confidence: float = 1.0


@dataclass
class CharTime:
    """单个字符的时间戳（毫秒）"""
    start_ms: int = 0
    end_ms: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "CharTime":
        return cls(start_ms=d["start_ms"], end_ms=d["end_ms"])


@dataclass
class RawTranscript:
    """ASR 原始输出，时间戳的唯一权威来源。

    text 与 char_timestamps 等长，标点/空格等无语音的字符对应位置为 None。
    """
    text: str = ""
    char_timestamps: List[Optional[CharTime]] = field(default_factory=list)

    def get_time_range(self, start_idx: int, end_idx: int) -> tuple:
        """返回 [start_idx, end_idx) 字符区间的起止时间（秒）。

        跳过 None（标点等无时间戳的字符），取区间内第一个和最后一个
        有时间戳的字符。

        Returns:
            (start_sec, end_sec) 元组；区间内无有效时间戳时返回 (0.0, 0.0)
        """
        start_ms = None
        end_ms = None
        for i in range(max(0, start_idx), min(end_idx, len(self.char_timestamps))):
            ct = self.char_timestamps[i]
            if ct is not None:
                if start_ms is None:
                    start_ms = ct.start_ms
                end_ms = ct.end_ms
        if start_ms is None or end_ms is None:
            return (0.0, 0.0)
        return (start_ms / 1000.0, end_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "char_timestamps": [
                ct.to_dict() if ct is not None else None
                for ct in self.char_timestamps
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RawTranscript":
        return cls(
            text=d["text"],
            char_timestamps=[
                CharTime.from_dict(ct) if ct is not None else None
                for ct in d.get("char_timestamps", [])
            ],
        )


@dataclass
class AlignedTranscript:
    """纠错后文本 + 与原始文本的对齐映射。

    text 与 raw_align 等长。raw_align[i] 指向 RawTranscript.text 的字符索引，
    None 表示 LLM 新增字（标点、修正字），无直接时间戳。
    """
    text: str = ""
    raw_align: List[Optional[int]] = field(default_factory=list)
    raw: Optional[RawTranscript] = None

    def get_time_range(self, start_idx: int, end_idx: int) -> tuple:
        """返回 [start_idx, end_idx) 字符区间的起止时间（秒）。

        沿 raw_align 映射回溯到 RawTranscript 取时间戳，
        跳过 None（LLM 新增字）。
        """
        if self.raw is None:
            return (0.0, 0.0)
        raw_start = None
        raw_end = None
        for i in range(max(0, start_idx), min(end_idx, len(self.raw_align))):
            raw_idx = self.raw_align[i]
            if raw_idx is not None:
                if raw_start is None:
                    raw_start = raw_idx
                raw_end = raw_idx
        if raw_start is None or raw_end is None:
            return (0.0, 0.0)
        return self.raw.get_time_range(raw_start, raw_end + 1)


@dataclass
class OCRResult:
    """OCR 识别结果（单条文本或公式）

    R-008 扩展：新增 block_type / latex 字段，支持两引擎并行输出。
    - text + block_type=text/handwritten：PP-OCRv6 识别的文字行
    - text + block_type=formula + latex：PP-FormulaNet 识别的公式
    """
    text: str = ""
    confidence: float = 1.0
    bounding_box: List[float] = field(default_factory=list)  # [x1, y1, x2, y2] 归一化 0-1
    block_type: str = "text"  # text / handwritten / formula / title / image
    latex: str = ""  # 公式 LaTeX（仅 block_type=formula 时有效）


@dataclass
class OCRFrameResult:
    """OCR 单帧识别结果，由 M5 文字识别模块返回"""
    timestamp: float = 0.0
    image_path: str = ""
    results: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    is_duplicate: bool = False


# === 三、提取结果相关 ===

@dataclass
class KnowledgePoint:
    """知识点，由 M7 知识点提取模块返回（10 设计：扩展 end_time+content+supplement 用于深度整理）"""
    index: int = 0
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    content: str = ""        # 核心内容：老师讲解总结（公式 LaTeX 嵌入）
    supplement: str = ""     # 补充内容：豆包补充（高考范围内）


@dataclass
class SolutionStep:
    """解题步骤，属于 Problem 的子结构（09 设计：扩展 start_time/end_time 用于截图定位）"""
    step_number: int = 0
    content: str = ""
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class Problem:
    """题目，由 M8 题目提取模块返回"""
    index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    question_text: str = ""
    solution_steps: List[SolutionStep] = field(default_factory=list)
    has_image: bool = False
    image_description: str = ""
    source: str = ""
    confidence: float = 1.0
    asr_question_text: str = ""  # ASR识别的原题文本（用于与OCR原题对比，保留原始ASR记录）


# === 四、LLM 相关 ===

@dataclass
class TokenUsage:
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """LLM 非流式响应"""
    content: str = ""
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"


@dataclass
class LLMChunk:
    """LLM 流式响应块"""
    delta_content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


# === 五、输出与流水线相关 ===

@dataclass
class ProcessResult:
    """完整处理结果，由 M13 主流程编排模块返回"""
    video_path: str = ""
    video_info: Optional[VideoInfo] = None
    asr_results: Optional[RawTranscript] = None
    corrected_transcript: Optional[AlignedTranscript] = None
    ocr_results: List[OCRFrameResult] = field(default_factory=list)
    full_text: str = ""
    knowledge_points: List[KnowledgePoint] = field(default_factory=list)
    problems: List[Problem] = field(default_factory=list)
    screenshot_paths: List[str] = field(default_factory=list)
    mindmap_opml: str = ""


@dataclass
class OutputFiles:
    """输出文件路径列表，由 M12 输出组装模块返回"""
    transcript_path: str = ""
    knowledge_path: str = ""
    mindmap_path: str = ""
    problem_files: List[str] = field(default_factory=list)
    output_dir: str = ""


@dataclass
class PipelineContext:
    """流水线上下文，M13 主流程编排模块内部使用"""
    video_path: str = ""
    video_info: Optional[VideoInfo] = None
    audio_path: Optional[str] = None
    frame_paths: Optional[List[str]] = None
    frame_timestamps: Optional[List[float]] = None
    asr_results: Optional[RawTranscript] = None
    corrected_transcript: Optional[AlignedTranscript] = None
    ocr_results: Optional[List[OCRFrameResult]] = None
    full_text: Optional[str] = None
    knowledge_points: Optional[List[KnowledgePoint]] = None
    problems: Optional[List[Problem]] = None
    screenshot_paths: Optional[List[str]] = None
    knowledge_screenshot_paths: Optional[List[str]] = None
    solution_screenshot_paths: Optional[List[str]] = None
    mindmap_opml: Optional[str] = None
    output_files: Optional[OutputFiles] = None
    completed_stages: List[str] = field(default_factory=list)


@dataclass
class PipelineProgress:
    """流水线进度信息"""
    current_stage: str = ""
    completed_stages: List[str] = field(default_factory=list)
    total_stages: int = 0
    progress_percent: float = 0.0
    is_running: bool = False
    is_cancelled: bool = False
    error: Optional[str] = None
