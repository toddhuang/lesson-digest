"""
统一数据模型定义
所有跨模块共享的数据结构（dataclass）统一定义于此。
对应文档：03_接口设计/00_数据模型.md
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
    """ASR 语音识别结果（单句）"""
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    confidence: float = 1.0


@dataclass
class OCRResult:
    """OCR 文字识别结果（单条文本）"""
    text: str = ""
    confidence: float = 1.0
    bounding_box: List[float] = field(default_factory=list)  # [x1, y1, x2, y2] 归一化 0-1


@dataclass
class OCRFrameResult:
    """OCR 单帧识别结果，由 M5 文字识别模块返回"""
    timestamp: float = 0.0
    image_path: str = ""
    results: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    is_duplicate: bool = False


@dataclass
class MergedText:
    """合并后的文本片段，由 M6 文本合并模块返回"""
    timestamp: float = 0.0
    text: str = ""
    source: str = "asr"  # "asr" / "ocr"
    confidence: float = 1.0


# === 三、提取结果相关 ===

@dataclass
class KnowledgePoint:
    """知识点，由 M7 知识点提取模块返回"""
    index: int = 0
    name: str = ""
    start_time: float = 0.0
    confidence: float = 1.0


@dataclass
class SolutionStep:
    """解题步骤，属于 Problem 的子结构"""
    step_number: int = 0
    content: str = ""
    timestamp: float = 0.0


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
    asr_results: List[Sentence] = field(default_factory=list)
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
    asr_results: Optional[List[Sentence]] = None
    ocr_results: Optional[List[OCRFrameResult]] = None
    full_text: Optional[str] = None
    knowledge_points: Optional[List[KnowledgePoint]] = None
    problems: Optional[List[Problem]] = None
    screenshot_paths: Optional[List[str]] = None
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
