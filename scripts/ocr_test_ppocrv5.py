"""
PaddleOCR 3.x (PP-OCRv5) 识别测试脚本
对 tests/ 目录下的测试图片进行 OCR 识别，记录识别结果和耗时。
使用独立 venv: venv_paddle3
结果保存到 scripts/ocr_results_ppocrv5.md
"""

import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT / "tests"
OUTPUT_FILE = Path(__file__).resolve().parent / "ocr_results_ppocrv6.md"

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

    print("初始化 PaddleOCR 3.x (默认 PP-OCRv6_medium)...")
    # PaddleOCR 3.7 默认使用 PP-OCRv6_medium 模型
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu",
    )

    results_lines = []
    results_lines.append("# PaddleOCR 3.7 (PP-OCRv6_medium) 识别结果\n")
    results_lines.append(f"> 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    import paddle
    results_lines.append(f"> PaddlePaddle 版本: {paddle.__version__}\n")
    results_lines.append(f"> PaddleOCR 版本: 3.7.0 (默认 PP-OCRv6_medium)\n")
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

        # 正式计时
        start = time.perf_counter()
        result = ocr.predict(str(img_path))
        elapsed = time.perf_counter() - start
        total_time += elapsed

        results_lines.append(f"## {img_name}\n")
        results_lines.append(f"- 图片尺寸: {w}x{h}")
        results_lines.append(f"- 耗时: {elapsed:.3f}s\n")

        if not result:
            results_lines.append("**未识别到任何文字**\n")
        else:
            res = result[0]
            res_dict = res.json if hasattr(res, 'json') else {}

            # PaddleOCR 3.x 输出字段
            rec_texts = res_dict.get("res", {}).get("rec_texts", [])
            rec_scores = res_dict.get("res", {}).get("rec_scores", [])
            rec_boxes = res_dict.get("res", {}).get("rec_boxes", [])

            if not rec_texts:
                results_lines.append("**未识别到任何文字**\n")
            else:
                results_lines.append(f"识别到 {len(rec_texts)} 行文字:\n")
                results_lines.append("| # | 文字 | 置信度 | 坐标(左上→右下) |")
                results_lines.append("|---|---|---|---|")
                for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    if i < len(rec_boxes):
                        box = rec_boxes[i]
                        # box format: [x_min, y_min, x_max, y_max]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        coord = f"({x1},{y1})→({x2},{y2})"
                    else:
                        coord = ""
                    text_escaped = text.replace("|", "\\|")
                    results_lines.append(
                        f"| {i+1} | {text_escaped} | {float(score):.4f} | {coord} |"
                    )

                # 纯文本拼接
                results_lines.append("\n**识别全文（按顺序拼接）**:\n")
                results_lines.append("```")
                for text in rec_texts:
                    results_lines.append(text)
                results_lines.append("```\n")

        results_lines.append("---\n")
        n = len(rec_texts) if result and rec_texts else 0
        print(f"  完成: {n} 行, {elapsed:.3f}s")

    results_lines.append(f"\n## 汇总\n")
    results_lines.append(f"- 总耗时: {total_time:.3f}s")
    results_lines.append(f"- 平均每帧: {total_time/len(TEST_IMAGES):.3f}s")

    OUTPUT_FILE.write_text("\n".join(results_lines), encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_FILE}")
    print(f"总耗时: {total_time:.3f}s, 平均: {total_time/len(TEST_IMAGES):.3f}s/帧")


if __name__ == "__main__":
    main()
