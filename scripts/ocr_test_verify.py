"""
OCR 识别效果验证脚本

用通用模型和手写体模型分别识别 tests 目录下的7张测试图片，
对比两种模型对印刷体和手写体的识别效果。

结果保存到 ocr_test_result.json
"""

import os
import json
import time
from paddleocr import PaddleOCR

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_DIR = os.path.join(PROJECT_ROOT, "tests")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "ocr_test_result.json")

IMAGE_FILES = [
    "印刷体01.png",
    "印刷体2.png",
    "印刷体3数学题目.png",
    "印刷体+手写体.png",
    "印刷体+手写体2.png",
    "印刷体+手写体3.png",
    "印刷体+手写体4.png",
]


def init_ocr():
    """初始化通用模型和手写体模型两个实例"""
    print("初始化通用模型（ch_PP-OCRv4_rec_infer）...")
    ocr_general = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=True,
    )

    print("初始化手写体模型（ch_PP-OCRv4_handwritten_rec_infer）...")
    handwritten_model_dir = os.path.join(
        os.path.expanduser("~"),
        ".paddleocr", "whl", "rec", "ch",
        "ch_PP-OCRv4_handwritten_rec_infer"
    )
    ocr_handwritten = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=True,
        rec_model_dir=handwritten_model_dir,
    )

    return ocr_general, ocr_handwritten


def recognize(ocr, image_path):
    """
    识别单张图片，返回结构化结果列表

    Returns:
        list[dict]: 每个元素包含 text, confidence, box
    """
    result = ocr.ocr(image_path, cls=True)
    lines = []

    if result and result[0]:
        for line in result[0]:
            box = line[0]  # 四个角坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1][0]  # 识别文字
            confidence = line[1][1]  # 置信度

            # 计算文本框的中心点y坐标，用于按行排序
            center_y = sum(p[1] for p in box) / 4
            center_x = sum(p[0] for p in box) / 4

            lines.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "center_x": round(center_x, 1),
                "center_y": round(center_y, 1),
                "box": [[int(p[0]), int(p[1])] for p in box],
            })

    # 按y坐标排序（从上到下），同一行内按x排序（从左到右）
    lines.sort(key=lambda x: (x["center_y"], x["center_x"]))

    return lines


def print_result(img_file, general_lines, handwritten_lines):
    """打印单张图片的识别结果"""
    print(f"\n{'='*80}")
    print(f"图片: {img_file}")
    print(f"{'='*80}")

    print(f"\n--- 通用模型识别结果（{len(general_lines)}行）---")
    for i, line in enumerate(general_lines):
        print(f"  [{i+1}] {line['text']}  (置信度: {line['confidence']})")

    print(f"\n--- 手写体模型识别结果（{len(handwritten_lines)}行）---")
    for i, line in enumerate(handwritten_lines):
        print(f"  [{i+1}] {line['text']}  (置信度: {line['confidence']})")


def main():
    start_time = time.time()

    # 初始化模型
    ocr_general, ocr_handwritten = init_ocr()

    results = {}

    for img_file in IMAGE_FILES:
        img_path = os.path.join(TEST_DIR, img_file)

        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue

        print(f"\n处理: {img_file}")

        # 通用模型识别
        t1 = time.time()
        general_lines = recognize(ocr_general, img_path)
        t_general = time.time() - t1

        # 手写体模型识别
        t2 = time.time()
        handwritten_lines = recognize(ocr_handwritten, img_path)
        t_handwritten = time.time() - t2

        # 打印结果
        print_result(img_file, general_lines, handwritten_lines)

        # 保存结构化结果
        results[img_file] = {
            "general": {
                "line_count": len(general_lines),
                "avg_confidence": round(
                    sum(l["confidence"] for l in general_lines) / len(general_lines), 4
                ) if general_lines else 0,
                "time_seconds": round(t_general, 2),
                "lines": general_lines,
            },
            "handwritten": {
                "line_count": len(handwritten_lines),
                "avg_confidence": round(
                    sum(l["confidence"] for l in handwritten_lines) / len(handwritten_lines), 4
                ) if handwritten_lines else 0,
                "time_seconds": round(t_handwritten, 2),
                "lines": handwritten_lines,
            },
        }

    # 保存结果到JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"全部完成！总耗时: {total_time:.1f}秒")
    print(f"结果已保存到: {OUTPUT_FILE}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
