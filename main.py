#!/usr/bin/env python3
"""
教学视频内容提取与总结工具 - 主入口
用法: python main.py <视频文件路径> [--output <输出目录>] [--config <配置文件>] [--force]
"""

import argparse
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _setup_windows_gpu_path():
    """Windows 平台：设置 CUDA/cuDNN DLL 搜索路径，解决 PyTorch 与 PaddlePaddle 共存冲突。

    PATH 顺序（优先级从高到低）：
    1. torch\lib — PyTorch 自带的 CUDA 库，优先加载，避免与系统 CUDA Toolkit 冲突
    2. cuDNN bin — cuDNN 8.9.x（PaddlePaddle 需要）
    3. CUDA Toolkit bin — cublas 等（PaddlePaddle 需要）

    仅在 Windows 平台生效，Linux/macOS 不处理。
    """
    if sys.platform != "win32":
        return

    prepend_dirs = []

    # 1. torch\lib（从当前 venv 推断）
    torch_lib = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "venv", "Lib", "site-packages", "torch", "lib"
    )
    if os.path.isdir(torch_lib):
        prepend_dirs.append(torch_lib)

    # 2. cuDNN bin（支持环境变量 CUDNN_PATH 覆盖，默认 C:\tools\cudnn\bin）
    cudnn_bin = os.environ.get("CUDNN_PATH", r"C:\tools\cudnn\bin")
    if os.path.isdir(cudnn_bin):
        prepend_dirs.append(cudnn_bin)

    # 3. CUDA Toolkit bin（支持环境变量 CUDA_PATH 覆盖，默认 v12.0）
    cuda_bin = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
    )
    if os.path.isdir(cuda_bin):
        prepend_dirs.append(cuda_bin)

    if prepend_dirs:
        os.environ["PATH"] = os.pathsep.join(prepend_dirs) + os.pathsep + os.environ.get("PATH", "")


# 必须在 import torch / paddle 之前调用
_setup_windows_gpu_path()

from config import load_config
from core.pipeline import Pipeline
from utils.logger import setup_logger

logger = setup_logger("main")


def main():
    parser = argparse.ArgumentParser(description="教学视频内容提取与总结工具")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="./output", help="输出根目录（默认: ./output）")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径（默认: config.yaml）")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新处理（忽略缓存）")

    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 创建并运行流水线
    pipeline = Pipeline(config)

    try:
        result = pipeline.run(
            video_path=args.video,
            output_dir=args.output,
            force=args.force,
        )

        # 输出结果摘要
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"视频: {result.video_path}")
        print(f"逐字稿: {len(result.asr_results)} 句")
        print(f"知识点: {len(result.knowledge_points)} 个")
        print(f"题目: {len(result.problems)} 道")
        print(f"截图: {len([p for p in result.screenshot_paths if p])} 张")
        print(f"输出目录: {result.video_path and os.path.join(args.output, os.path.splitext(os.path.basename(result.video_path))[0])}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"处理失败: {e}")
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
