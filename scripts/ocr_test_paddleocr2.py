"""
PaddleOCR 2.8.1 识别测试脚本
对 tests/ 目录下的测试图片进行 OCR 识别，记录识别结果和耗时。
结果保存到 scripts/ocr_results_paddleocr2.md
"""

import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT / "tests"
OUTPUT_FILE = Path(__file__).resolve().parent / "ocr_results_paddleocr2.md"

# 测试图片列表（按预期文档顺序）
TEST_IMAGES = [
    "印刷体01.png",
    "印刷体2.png",
    "印刷体3数学题目.png",
    "印刷体+手写体.png",
    "印刷体+手写体2.png",
    "印刷体+手写体3.png",
    "印刷体+手写体4.png",
]


def main():
    from paddleocr import PaddleOCR
    from PIL import Image

    print("初始化 PaddleOCR 2.8.1 (PP-OCRv4)...")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)

    results_lines = []
    results_lines.append("# PaddleOCR 2.8.1 (PP-OCRv4) 识别结果\n")
    results_lines.append(f"> 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_lines.append(f"> PaddleOCR 版本: 2.8.1\n")
    results_lines.append(f"> PaddlePaddle 版本: 2.6.1\n")
    results_lines.append("---\n")

    total_time = 0.0

    for img_name in TEST_IMAGES:
        img_path = TEST_DIR / img_name
        if not img_path.exists():
            results_lines.append(f"## {img_name}\n\n**文件不存在**\n\n---\n")
            continue

        # 获取图片尺寸
        with Image.open(img_path) as img:
            w, h = img.size

        print(f"识别: {img_name} ({w}x{h})...")

        # 预热（首次运行包含模型初始化开销，单独计时）
        # 正式计时
        start = time.perf_counter()
        result = ocr.ocr(str(img_path), cls=True)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        results_lines.append(f"## {img_name}\n")
        results_lines.append(f"- 图片尺寸: {w}x{h}")
        results_lines.append(f"- 耗时: {elapsed:.3f}s\n")

        if not result or not result[0]:
            results_lines.append("**未识别到任何文字**\n")
        else:
            lines = result[0]
            results_lines.append(f"识别到 {len(lines)} 行文字:\n")
            results_lines.append("| # | 文字 | 置信度 | 坐标(左上→右下) |")
            results_lines.append("|---|---|---|---|")
            for i, line in enumerate(lines):
                box = line[0]
                text, conf = line[1]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                # 转义 markdown 特殊字符
                text_escaped = text.replace("|", "\\|")
                results_lines.append(
                    f"| {i+1} | {text_escaped} | {conf:.4f} | "
                    f"({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f}) |"
                )

            # 纯文本拼接（方便查看整体识别效果）
            results_lines.append("\n**识别全文（按顺序拼接）**:\n")
            results_lines.append("```")
            for line in lines:
                results_lines.append(line[1][0])
            results_lines.append("```\n")

        results_lines.append("---\n")
        print(f"  完成: {len(result[0]) if result and result[0] else 0} 行, {elapsed:.3f}s")

    results_lines.append(f"\n## 汇总\n")
    results_lines.append(f"- 总耗时: {total_time:.3f}s")
    results_lines.append(f"- 平均每帧: {total_time/len(TEST_IMAGES):.3f}s")

    OUTPUT_FILE.write_text("\n".join(results_lines), encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_FILE}")
    print(f"总耗时: {total_time:.3f}s, 平均: {total_time/len(TEST_IMAGES):.3f}s/帧")


if __name__ == "__main__":
    main()
