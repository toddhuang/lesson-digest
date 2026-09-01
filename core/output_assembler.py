"""
M12 输出组装模块
将各模块结果格式化为输出文件，按规范目录组织。
对应文档：03_接口设计/M12_输出组装模块接口.md
"""

import os
from typing import List

from config import OutputConfig
from utils.models import ProcessResult, OutputFiles, Problem
from utils.file_utils import ensure_dir, save_text
from utils.timestamp import format_timestamp
from utils.logger import setup_logger

logger = setup_logger("M12_output")


class OutputAssembler:
    """输出组装器"""

    def __init__(self, config: OutputConfig):
        self.config = config

    def assemble(self, result: ProcessResult, output_dir: str) -> OutputFiles:
        """将所有处理结果组装为输出文件

        Args:
            result: 完整处理结果
            output_dir: 输出根目录

        Returns:
            OutputFiles 对象
        """
        logger.info(f"[M12] 输出组装: {output_dir}")

        # 创建输出目录结构
        video_name = os.path.splitext(os.path.basename(result.video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        problems_dir = os.path.join(video_output_dir, self.config.problems_dirname)
        screenshots_dir = os.path.join(video_output_dir, self.config.screenshots_dirname)

        ensure_dir(video_output_dir)
        ensure_dir(problems_dir)
        ensure_dir(screenshots_dir)

        output_files = OutputFiles(output_dir=video_output_dir)

        # 1. 逐字稿（优先输出纠错后文本，字级时间戳在缓存 JSON 中）
        transcript_path = os.path.join(video_output_dir, self.config.transcript_filename)
        transcript = result.corrected_transcript or result.asr_results
        if transcript:
            self._write_transcript(transcript.text, transcript_path, result, "纠错后")
            logger.info(f"[M12] 逐字稿已输出（纠错后）: {transcript_path}")
        output_files.transcript_path = transcript_path

        # 2. 知识点清单
        knowledge_path = os.path.join(video_output_dir, self.config.knowledge_filename)
        self._write_knowledge_list(result, knowledge_path)
        output_files.knowledge_path = knowledge_path

        # 3. 思维导图
        mindmap_path = os.path.join(video_output_dir, self.config.mindmap_filename)
        self._write_mindmap(result, mindmap_path)
        output_files.mindmap_path = mindmap_path

        # 4. 习题文件
        problem_files = self._write_problems(result, problems_dir, screenshots_dir)
        output_files.problem_files = problem_files

        logger.info(f"[M12] 输出组装完成: {len(output_files.problem_files)}个习题文件")
        return output_files

    def _write_transcript(self, text: str, output_path: str,
                          result: ProcessResult, title_suffix: str = "") -> None:
        """写逐字稿（纯文本，字级时间戳在缓存 JSON 中供 debug）

        Args:
            text: 逐字稿全文
            output_path: 输出路径
            result: 处理结果（用于视频信息）
            title_suffix: 标题后缀（如"纠错后"）
        """
        lines = [f"# 逐字稿（{title_suffix}）\n"]
        lines.append(f"> 视频：{os.path.basename(result.video_path)}")
        if result.video_info:
            lines.append(f"> 时长：{format_timestamp(result.video_info.duration, 'hh:mm:ss')}")
        lines.append("")
        lines.append(text)

        save_text("\n".join(lines), output_path)

    def _write_knowledge_list(self, result: ProcessResult, output_path: str) -> None:
        """写知识点清单（含深度整理：核心内容 + 补充内容）"""
        lines = ["# 知识点清单\n"]
        lines.append(f"> 视频：{os.path.basename(result.video_path)}")
        lines.append(f"> 知识点数量：{len(result.knowledge_points)}\n")

        for kp in result.knowledge_points:
            ts = format_timestamp(kp.start_time, self.config.timestamp_format)
            if kp.end_time:
                ts_end = format_timestamp(kp.end_time, self.config.timestamp_format)
                lines.append(f"## {kp.index}. {kp.name} [{ts} - {ts_end}]\n")
            else:
                lines.append(f"## {kp.index}. {kp.name} [{ts}]\n")

            if kp.content:
                lines.append("### 核心内容")
                lines.append(kp.content)
                lines.append("")

            if kp.supplement:
                lines.append("### 补充内容（高考范围内）")
                lines.append(kp.supplement)
                lines.append("")

        save_text("\n".join(lines), output_path)
        logger.info(f"[M12] 知识点清单: {output_path}")

    def _write_mindmap(self, result: ProcessResult, output_path: str) -> None:
        """写思维导图"""
        save_text(result.mindmap_opml, output_path)
        logger.info(f"[M12] 思维导图: {output_path}")

    def _write_problems(self, result: ProcessResult, problems_dir: str,
                         screenshots_dir: str) -> List[str]:
        """写习题文件（原题+解析合并为单文件 题目NN.md）"""
        from core.problem_extractor import ProblemExtractor

        formatter = ProblemExtractor(llm=None)
        problem_files = []

        for i, problem in enumerate(result.problems):
            filename = f"题目{problem.index:02d}.md"
            path = os.path.join(problems_dir, filename)

            screenshot_rel = None
            if i < len(result.screenshot_paths) and result.screenshot_paths[i]:
                screenshot_src = result.screenshot_paths[i]
                screenshot_dst = os.path.join(screenshots_dir, f"题目{problem.index:02d}.jpg")
                if os.path.exists(screenshot_src) and screenshot_src != screenshot_dst:
                    import shutil
                    shutil.copy2(screenshot_src, screenshot_dst)
                screenshot_rel = f"../{self.config.screenshots_dirname}/题目{problem.index:02d}.jpg"

            md = formatter.to_problem_markdown(problem, screenshot_rel, self.config.timestamp_format)
            save_text(md, path)
            problem_files.append(path)

        return problem_files
