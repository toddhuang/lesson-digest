"""
M13 主流程编排模块（Pipeline）
串联所有模块，管理执行顺序、并行控制、断点续传、错误处理。
对应文档：03_接口设计/M13_主流程编排模块接口.md

M11-M17 重构：
- LLMClient 从 core.llm 导入，使用新的模型注册表+任务映射
- pipeline 层负责为每个任务创建 LLMSession 并注入业务模块
- 删除 LLM 层缓存（use_cache），缓存由 pipeline 层统一管理
- ASR 纠错不再传 backend，由任务配置决定模型

issue #11（debug 模块）：
- 接收可选 debugger 实例（duck typing，无 Protocol 约束）
- 各 _stage_xxx 内调 debugger.save_xxx() 输出 debug 产物
- release 时 config.debug.enabled=false + 删除 debugger/ 包，pipeline 无 import 依赖不崩
"""

import os
import time
import traceback
from typing import Any, Optional

from config import Config
from utils.models import (
    ProcessResult, PipelineContext, PipelineProgress,
)
from utils.file_utils import get_file_hash
from utils.logger import setup_logger
from utils.exceptions import PipelineError, VideoNotFoundError, VideoContentError, LLMError

from core.audio_extractor import AudioExtractor
from core.frame_extractor import FrameExtractor
from core.asr import ASRRecognizer
from core.ocr import OCRRecognizer
from core.text_merger import TextMerger
from core.content_extractor import ContentExtractor
from core.problem_extractor import ProblemExtractor
from core.knowledge_extractor import KnowledgeExtractor
from core.screenshot_capture import ScreenshotCapture
from core.mindmap_generator import MindmapGenerator
from core.llm import LLMClient
from core.output_assembler import OutputAssembler

logger = setup_logger("M13_pipeline")

# 阶段列表
# AGENTS.md 约定：一次 LLM 调用返回三样东西（纠错全文+知识点段+题目段）
# correct_and_extract 合并了原 correct_asr + extract_knowledge + extract_problems
STAGES = [
    "probe",
    "extract_audio",
    "extract_frames",
    "asr",
    "ocr",
    "correct_and_extract",
    "summarize_solution",
    "summarize_knowledge",
    "merge_text",
    "capture_screenshots",
    "generate_mindmap",
    "assemble_output",
]


class Pipeline:
    """主流程编排器"""

    def __init__(self, config: Config, mock_llm: bool = False,
                debugger: Optional[Any] = None):
        """
        Args:
            config: 全局配置
            mock_llm: 是否使用 mock LLM（链路测试用）
            debugger: 可选 DebugSink 实例（issue #11），None 时跳过所有 debug 输出。
                      duck typing，无 Protocol 约束，release 时删除 debugger/ 包不影响本类导入。
        """
        self.config = config
        self.debugger = debugger
        self._progress = PipelineProgress(total_stages=len(STAGES))
        self._cancelled = False

        # 初始化非 LLM 模块
        self.audio_extractor = AudioExtractor(
            sample_rate=config.asr.sample_rate,
            channels=config.asr.channels,
        )
        self.frame_extractor = FrameExtractor(
            interval=config.frame_dedup.interval_sec,
            fmt=config.video.frame_format,
            quality=config.video.frame_quality,
            dedup_threshold=config.frame_dedup.threshold,
            enable_dedup=True,
        )
        self.asr_recognizer = ASRRecognizer(
            adapter_type=config.asr.adapter_type,
            cache_dir=os.path.join(config.paths.cache_dir, "asr"),
        )
        self.ocr_recognizer = OCRRecognizer(
            text_adapter_type=config.ocr.text_adapter_type,
            formula_adapter_type=config.ocr.formula_adapter_type,
            config=config,
            cache_dir=os.path.join(config.paths.cache_dir, "ocr"),
        )
        self.text_merger = TextMerger()
        self.screenshot_capture = ScreenshotCapture(self.frame_extractor)
        self.output_assembler = OutputAssembler(config.output)

        # 初始化 LLM 客户端和各业务模块
        # AGENTS.md 约定：一次 LLM 调用返回三样东西（纠错全文+知识点段+题目段）
        self.llm_client = LLMClient(
            llm_config=config.llm,
            tasks=config.tasks,
            mock=mock_llm,
        )
        # content_extractor 注入 debugger，_locate_segment 每次定位调 save_locate_record（issue #11）
        self.content_extractor = ContentExtractor(
            self.llm_client.get_session("asr_correct_and_extract"),
            debugger=debugger,
        )
        # problem_extractor 保留用于后续 OCR 补充原题（题目段已由 ContentExtractor 提取）
        # solution_llm 用于解题过程整理（09 设计 issue #13）
        self.problem_extractor = ProblemExtractor(
            self.llm_client.get_session("problem_extraction"),
            solution_llm=self.llm_client.get_session("solution_summary"),
        )
        self.mindmap_generator = MindmapGenerator(
            self.llm_client.get_session("mindmap_generation")
        )
        # knowledge_extractor 用于知识点深度整理（10 设计 issue #9）
        self.knowledge_extractor = KnowledgeExtractor(
            self.llm_client.get_session("knowledge_extraction"),
            summary_llm=self.llm_client.get_session("knowledge_summary"),
        )

    def run(self, video_path: str, output_dir: str, force: bool = False) -> ProcessResult:
        """执行完整的视频处理流水线

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出根目录
            force: 是否强制重新处理（忽略 ASR/OCR 缓存）

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
            self._run_stage("probe", context, force)
            self._run_stage("extract_audio", context, force)
            self._run_stage("extract_frames", context, force)
            self._run_stage("asr", context, force)
            self._run_stage("ocr", context, force)
            self._run_stage("correct_and_extract", context, force)
            self._run_stage("summarize_solution", context, force)
            self._run_stage("summarize_knowledge", context, force)
            self._run_stage("merge_text", context, force)
            self._run_stage("capture_screenshots", context, force)
            self._run_stage("generate_mindmap", context, force)
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
            logger.error(f"[Pipeline] 处理失败: {e}\n{traceback.format_exc()}")
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
            elif stage == "correct_and_extract":
                self._stage_correct_and_extract(context)
            elif stage == "summarize_solution":
                self._stage_summarize_solution(context)
            elif stage == "summarize_knowledge":
                self._stage_summarize_knowledge(context)
            elif stage == "merge_text":
                self._stage_merge_text(context)
            elif stage == "capture_screenshots":
                self._stage_capture_screenshots(context)
            elif stage == "generate_mindmap":
                self._stage_generate_mindmap(context)
            elif stage == "assemble_output":
                self._stage_assemble_output(context, kwargs.get("output_dir", "./output"))

            context.completed_stages.append(stage)
            self._progress.completed_stages = context.completed_stages
            elapsed = time.time() - start_time
            logger.info(f"[Pipeline] 阶段 {stage} 完成 ({elapsed:.2f}s)")

        except VideoContentError as e:
            logger.error(f"[Pipeline] 阶段 {stage} 失败 ({type(e).__name__}): {e}")
            raise

    def _stage_probe(self, context: PipelineContext):
        """探测视频信息"""
        from utils.video_probe import probe_video
        context.video_info = probe_video(context.video_path)
        # debugger 确定视频名 + 接管运行日志归档（issue #11 第 9 类产物）
        if self.debugger is not None:
            video_name = os.path.splitext(os.path.basename(context.video_path))[0]
            self.debugger.set_video_name(video_name)
            self.debugger.attach_log_handler()

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
        # 1. ASR 原始逐字稿（issue #11 第 1 类产物）
        if self.debugger is not None and context.asr_results is not None:
            self.debugger.save_asr_raw(context.asr_results)

    def _stage_ocr(self, context: PipelineContext, force: bool):
        """文字识别"""
        context.ocr_results = self.ocr_recognizer.recognize_frames(
            context.frame_paths, context.frame_timestamps, use_cache=not force
        )

    def _stage_correct_and_extract(self, context: PipelineContext):
        """一次 LLM 调用完成 ASR 纠错 + 知识点段 + 题目段提取

        AGENTS.md 约定：一次 LLM 调用返回三样东西（纠错全文+知识点段+题目段），
        不做三次独立调用。ContentExtractor 一次调用返回：
        - AlignedTranscript（纠错后文本 + 字级时间戳对齐映射）
        - List[KnowledgePoint]（知识点列表，通过文字段定位时间戳）
        - List[Problem]（题目列表，通过文字段定位时间戳）

        后续用 problem_extractor 的 OCR 补充和去重方法做后处理。
        """
        if not context.asr_results:
            logger.warning("[Pipeline] ASR结果为空，跳过纠错和提取")
            return

        try:
            aligned, knowledge_points, problems = self.content_extractor.extract(
                context.asr_results
            )
            context.corrected_transcript = aligned
            context.knowledge_points = knowledge_points
            context.problems = problems
            logger.info(
                f"[Pipeline] 一次LLM调用完成: 纠错{len(aligned.text)}字, "
                f"知识点{len(knowledge_points)}个, 题目{len(problems)}道"
            )
            # 2/3/4. 纠错后全文 + 知识点段 + 题目段（issue #11）
            if self.debugger is not None:
                self.debugger.save_corrected_text(aligned)
                self.debugger.save_knowledge_segments(knowledge_points)
                self.debugger.save_problem_segments(problems)
        except LLMError as e:
            logger.error(
                f"[Pipeline] 纠错+提取失败 ({type(e).__name__})，使用原始逐字稿: {e}"
            )
            return

        # OCR 补充原题和去重（复用 ProblemExtractor 的后处理逻辑）
        if context.ocr_results and context.problems:
            context.problems = self.problem_extractor._enrich_with_ocr(
                context.problems, context.ocr_results
            )
        if len(context.problems) > 1:
            context.problems = self.problem_extractor._merge_problems(
                context.problems
            )
            logger.info(f"[Pipeline] 题目去重后: {len(context.problems)}道")

    def _stage_summarize_solution(self, context: PipelineContext):
        """解题过程整理（ASR+OCR 融合，每题目独立调 LLM，09 设计 issue #13）"""
        if not context.problems:
            return
        for problem in context.problems:
            self.problem_extractor.enrich_solution(
                problem, context.corrected_transcript, context.ocr_results
            )

    def _stage_summarize_knowledge(self, context: PipelineContext):
        """知识点深度整理（ASR+OCR 融合，每知识点独立调 LLM，10 设计 issue #9）"""
        if not context.knowledge_points:
            return
        for kp in context.knowledge_points:
            self.knowledge_extractor.enrich_knowledge(
                kp, context.corrected_transcript, context.ocr_results
            )

    def _stage_merge_text(self, context: PipelineContext):
        """ASR文本整理（优先使用纠错后文本，OCR不混入全文本）"""
        transcript = context.corrected_transcript or context.asr_results
        context.full_text = self.text_merger.merge(transcript)

    def _stage_capture_screenshots(self, context: PipelineContext):
        """题目截图 + 知识点截图 + 解题过程截图

        issue #11 目录迁移：
        - 06_知识点截图/（原 06_截图/知识点/）
        - 07_题目原题截图/（原 output/截图/，现统一到 debug，不颜色过滤）
        - 08_解题过程截图/（原 06_截图/解题过程/）
        output_assembler 后续从 debug/07/ 复制到 output/截图/ 给学生看
        """
        video_name = os.path.splitext(os.path.basename(context.video_path))[0]

        # 题目原题截图（debug/07_题目原题截图/，不做颜色过滤，issue #11）
        context.screenshot_paths = []
        if context.problems:
            q_dir = os.path.join(self.config.paths.debug_dir, video_name, "07_题目原题截图")
            context.screenshot_paths = self.screenshot_capture.capture_screenshots(
                context.video_path, context.problems, q_dir, enable_color_filter=False
            )

        # 知识点截图（debug/06_知识点截图/，issue #12）
        if context.knowledge_points:
            kp_dir = os.path.join(self.config.paths.debug_dir, video_name, "06_知识点截图")
            context.knowledge_screenshot_paths = self.screenshot_capture.capture_knowledge_screenshots(
                context.video_path, context.knowledge_points, kp_dir
            )

        # 解题过程截图（debug/08_解题过程截图/，issue #13）
        if context.problems and any(p.solution_steps for p in context.problems):
            sol_dir = os.path.join(self.config.paths.debug_dir, video_name, "08_解题过程截图")
            context.solution_screenshot_paths = self.screenshot_capture.capture_solution_screenshots(
                context.video_path, context.problems, sol_dir
            )

    def _stage_generate_mindmap(self, context: PipelineContext):
        """思维导图生成"""
        video_name = os.path.splitext(os.path.basename(context.video_path))[0]
        context.mindmap_opml = self.mindmap_generator.generate(
            context.knowledge_points, video_title=video_name,
            video_duration=context.video_info.duration
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
            asr_results=context.asr_results,
            corrected_transcript=context.corrected_transcript,
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
