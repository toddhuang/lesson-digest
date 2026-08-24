"""
M13 主流程编排模块（Pipeline）
串联所有模块，管理执行顺序、并行控制、断点续传、错误处理。
对应文档：03_接口设计/M13_主流程编排模块接口.md
"""

import os
import time
from typing import Optional

from config import Config
from utils.models import (
    ProcessResult, PipelineContext, PipelineProgress,
    VideoInfo, Sentence, OCRFrameResult, KnowledgePoint, Problem
)
from utils.file_utils import ensure_dir, get_file_hash
from utils.logger import setup_logger
from utils.exceptions import PipelineError, VideoNotFoundError

from core.audio_extractor import AudioExtractor
from core.frame_extractor import FrameExtractor
from core.asr import ASRRecognizer
from core.ocr import OCRRecognizer
from core.text_merger import TextMerger
from core.knowledge_extractor import KnowledgeExtractor
from core.problem_extractor import ProblemExtractor
from core.screenshot_capture import ScreenshotCapture
from core.mindmap_generator import MindmapGenerator
from core.llm_client import LLMClient
from core.output_assembler import OutputAssembler

logger = setup_logger("M13_pipeline")

# 阶段列表
STAGES = [
    "probe",
    "extract_audio",
    "extract_frames",
    "asr",
    "ocr",
    "correct_asr",
    "merge_text",
    "extract_knowledge",
    "extract_problems",
    "capture_screenshots",
    "generate_mindmap",
    "assemble_output",
]


class Pipeline:
    """主流程编排器"""

    def __init__(self, config: Config):
        self.config = config
        self._progress = PipelineProgress(total_stages=len(STAGES))
        self._cancelled = False

        # 初始化各模块
        self.audio_extractor = AudioExtractor(
            sample_rate=config.asr.sample_rate,
            channels=config.asr.channels,
        )
        self.frame_extractor = FrameExtractor(
            interval=config.video.frame_interval,
            fmt=config.video.frame_format,
            quality=config.video.frame_quality,
        )
        self.asr_recognizer = ASRRecognizer(
            adapter_type=config.asr.adapter_type,
            cache_dir=os.path.join(config.paths.cache_dir, "asr"),
        )
        self.ocr_recognizer = OCRRecognizer(
            adapter_type=config.ocr.adapter_type,
            config=config.ocr,
            cache_dir=os.path.join(config.paths.cache_dir, "ocr"),
        )
        self.text_merger = TextMerger()
        self.llm_client = LLMClient(
            config.llm,
            cache_dir=os.path.join(config.paths.cache_dir, "llm"),
        )
        self.knowledge_extractor = KnowledgeExtractor(self.llm_client)
        self.problem_extractor = ProblemExtractor(self.llm_client)
        self.screenshot_capture = ScreenshotCapture(self.frame_extractor)
        self.mindmap_generator = MindmapGenerator(self.llm_client)
        self.output_assembler = OutputAssembler(config.output, self.llm_client)

    def run(self, video_path: str, output_dir: str, force: bool = False) -> ProcessResult:
        """执行完整的视频处理流水线

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出根目录
            force: 是否强制重新处理（忽略缓存）

        Returns:
            ProcessResult 对象

        Raises:
            VideoNotFoundError: 视频文件不存在
            PipelineError: 流水线执行失败
        """
        logger.info(f"=" * 60)
        logger.info(f"[Pipeline] 开始处理: {video_path}")
        logger.info(f"=" * 60)

        if not os.path.exists(video_path):
            raise VideoNotFoundError(f"视频文件不存在: {video_path}")

        self._cancelled = False
        self._progress = PipelineProgress(total_stages=len(STAGES), is_running=True)
        context = PipelineContext(video_path=video_path)

        try:
            # 阶段1: 探测视频信息
            self._run_stage("probe", context, force)

            # 阶段2-3: 音轨提取 + 关键帧提取（并行，mock阶段顺序执行）
            self._run_stage("extract_audio", context, force)
            self._run_stage("extract_frames", context, force)

            # 阶段4-5: ASR + OCR（并行，mock阶段顺序执行）
            self._run_stage("asr", context, force)
            self._run_stage("ocr", context, force)

            # 阶段6: ASR纠错（豆包后端，纠错后替换context.asr_results，后续所有模块基于纠错后文本）
            self._run_stage("correct_asr", context, force)

            # 阶段7: 文本合并
            self._run_stage("merge_text", context, force)

            # 阶段7-8: 知识点提取 + 题目提取（并行，mock阶段顺序执行）
            self._run_stage("extract_knowledge", context, force)
            self._run_stage("extract_problems", context, force)

            # 阶段9: 题目截图（依赖题目提取）
            self._run_stage("capture_screenshots", context, force)

            # 阶段10: 思维导图生成（可与7/8并行，mock阶段顺序执行）
            self._run_stage("generate_mindmap", context, force)

            # 阶段11: 输出组装
            self._run_stage("assemble_output", context, force, output_dir=output_dir)

            self._progress.is_running = False
            self._progress.progress_percent = 100.0

            logger.info(f"=" * 60)
            logger.info(f"[Pipeline] 处理完成!")
            logger.info(f"=" * 60)

            return self._context_to_result(context)

        except Exception as e:
            self._progress.is_running = False
            self._progress.error = str(e)
            logger.error(f"[Pipeline] 处理失败: {e}")
            raise PipelineError(f"流水线执行失败（阶段: {self._progress.current_stage}）: {e}",
                                stage=self._progress.current_stage, original_error=e)

    def _run_stage(self, stage: str, context: PipelineContext, force: bool, **kwargs):
        """执行单个阶段"""
        if self._cancelled:
            logger.info(f"[Pipeline] 已取消，跳过阶段: {stage}")
            return

        self._progress.current_stage = stage
        completed_count = STAGES.index(stage)
        self._progress.progress_percent = (completed_count / len(STAGES)) * 100

        logger.info(f"[Pipeline] 阶段 [{completed_count+1}/{len(STAGES)}]: {stage}")
        start_time = time.time()

        try:
            if stage == "probe":
                self._stage_probe(context)
            elif stage == "extract_audio":
                self._stage_extract_audio(context)
            elif stage == "extract_frames":
                self._stage_extract_frames(context)
            elif stage == "asr":
                self._stage_asr(context, force)
            elif stage == "ocr":
                self._stage_ocr(context, force)
            elif stage == "correct_asr":
                self._stage_correct_asr(context, force)
            elif stage == "merge_text":
                self._stage_merge_text(context)
            elif stage == "extract_knowledge":
                self._stage_extract_knowledge(context, force)
            elif stage == "extract_problems":
                self._stage_extract_problems(context, force)
            elif stage == "capture_screenshots":
                self._stage_capture_screenshots(context)
            elif stage == "generate_mindmap":
                self._stage_generate_mindmap(context, force)
            elif stage == "assemble_output":
                self._stage_assemble_output(context, kwargs.get("output_dir", "./output"))

            context.completed_stages.append(stage)
            self._progress.completed_stages = context.completed_stages
            elapsed = time.time() - start_time
            logger.info(f"[Pipeline] 阶段 {stage} 完成 ({elapsed:.2f}s)")

        except Exception as e:
            logger.error(f"[Pipeline] 阶段 {stage} 失败: {e}")
            raise

    def _stage_probe(self, context: PipelineContext):
        """探测视频信息"""
        from utils.video_probe import probe_video
        context.video_info = probe_video(context.video_path)

    def _stage_extract_audio(self, context: PipelineContext):
        """提取音轨"""
        cache_dir = os.path.join(self.config.paths.cache_dir, get_file_hash(context.video_path))
        audio_path = os.path.join(cache_dir, "audio.wav")
        info = self.audio_extractor.extract_audio(context.video_path, audio_path)
        context.audio_path = info.path

    def _stage_extract_frames(self, context: PipelineContext):
        """提取关键帧"""
        cache_dir = os.path.join(self.config.paths.cache_dir, get_file_hash(context.video_path), "frames")
        frames = self.frame_extractor.extract_frames(context.video_path, cache_dir)
        context.frame_paths = [f.path for f in frames]
        context.frame_timestamps = [f.timestamp for f in frames]

    def _stage_asr(self, context: PipelineContext, force: bool):
        """语音识别"""
        context.asr_results = self.asr_recognizer.recognize(context.audio_path, use_cache=not force)

    def _stage_ocr(self, context: PipelineContext, force: bool):
        """文字识别"""
        context.ocr_results = self.ocr_recognizer.recognize_frames(
            context.frame_paths, context.frame_timestamps, use_cache=not force
        )

    def _stage_correct_asr(self, context: PipelineContext, force: bool):
        """ASR纠错（豆包后端，纠错后替换context.asr_results）

        重要：此阶段必须在merge_text之前执行，确保后续知识点提取、题目提取、
        思维导图生成都基于纠错后的文本，而不是有同音词错误的原始文本。
        """
        if not context.asr_results:
            logger.warning("[Pipeline] ASR结果为空，跳过纠错")
            return

        try:
            from utils.asr_corrector import ASRCorrector
            corrector = ASRCorrector(self.llm_client)
            # 使用配置的默认服务商（config.llm.default_provider，默认volcengine/豆包）
            default_provider = self.config.llm.default_provider
            corrected = corrector.correct(context.asr_results, backend=default_provider)
            context.asr_results = corrected
            logger.info(f"[Pipeline] ASR纠错完成（服务商: {default_provider}）")
        except Exception as e:
            logger.error(f"[Pipeline] ASR纠错失败，使用原始逐字稿: {e}")
            # 纠错失败不中断流水线，继续使用原始ASR结果

    def _stage_merge_text(self, context: PipelineContext):
        """ASR文本整理（只处理ASR，OCR不混入全文本）"""
        context.full_text = self.text_merger.merge(context.asr_results)

    def _stage_extract_knowledge(self, context: PipelineContext, force: bool):
        """知识点提取"""
        context.knowledge_points = self.knowledge_extractor.extract(
            context.full_text, context.video_info.duration, use_cache=not force
        )

    def _stage_extract_problems(self, context: PipelineContext, force: bool):
        """题目提取"""
        context.problems = self.problem_extractor.extract(
            context.full_text, context.video_info.duration, use_cache=not force,
            ocr_results=context.ocr_results
        )

    def _stage_capture_screenshots(self, context: PipelineContext):
        """题目截图"""
        # 截图直接输出到最终目录
        video_name = os.path.splitext(os.path.basename(context.video_path))[0]
        screenshots_dir = os.path.join(self.config.paths.output_dir, video_name, self.config.output.screenshots_dirname)
        context.screenshot_paths = self.screenshot_capture.capture_screenshots(
            context.video_path, context.problems, screenshots_dir
        )

    def _stage_generate_mindmap(self, context: PipelineContext, force: bool):
        """思维导图生成"""
        video_name = os.path.splitext(os.path.basename(context.video_path))[0]
        context.mindmap_opml = self.mindmap_generator.generate(
            context.knowledge_points, video_title=video_name,
            video_duration=context.video_info.duration, use_cache=not force
        )

    def _stage_assemble_output(self, context: PipelineContext, output_dir: str):
        """输出组装"""
        result = self._context_to_result(context)
        context.output_files = self.output_assembler.assemble(result, output_dir)

    def _context_to_result(self, context: PipelineContext) -> ProcessResult:
        """将 PipelineContext 转换为 ProcessResult"""
        return ProcessResult(
            video_path=context.video_path,
            video_info=context.video_info,
            asr_results=context.asr_results or [],
            ocr_results=context.ocr_results or [],
            full_text=context.full_text or "",
            knowledge_points=context.knowledge_points or [],
            problems=context.problems or [],
            screenshot_paths=context.screenshot_paths or [],
            mindmap_opml=context.mindmap_opml or "",
        )

    def get_progress(self) -> PipelineProgress:
        """获取当前流水线执行进度"""
        return self._progress

    def cancel(self) -> None:
        """取消当前正在执行的流水线"""
        self._cancelled = True
        self._progress.is_cancelled = True
        logger.info("[Pipeline] 收到取消指令")
