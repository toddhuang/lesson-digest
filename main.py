#!/usr/bin/env python3
"""
教学视频内容提取与总结工具 - 主入口
用法: python main.py <视频文件路径> [--output <输出目录>] [--config <配置文件>] [--force]
"""

import argparse
import sys
import os
import traceback

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows GPU DLL 路径设置，必须在 import torch / paddle 之前调用
from utils.gpu_env import setup_gpu_path, preload_torch
setup_gpu_path()
preload_torch()  # 先加载 torch 的 CUDA DLL，避免 paddle 命中系统旧版 DLL

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
        video_name = os.path.splitext(os.path.basename(result.video_path))[0]
        output_dir = os.path.join(args.output, video_name)
        print(f"输出目录: {output_dir}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"处理失败: {e}\n{traceback.format_exc()}")
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
