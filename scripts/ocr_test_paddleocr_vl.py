"""
PaddleOCR-VL (0.9B) 识别测试脚本
对 tests/ 目录下的测试图片进行 OCR 识别，重点验证公式 LaTeX 输出。
使用独立 venv: venv_paddle3
结果保存到 scripts/ocr_results_paddleocr_vl.md
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT / "tests"
OUTPUT_FILE = Path(__file__).resolve().parent / "ocr_results_paddleocr_vl.md"

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
    from paddleocr import PaddleOCRVL
    from PIL import Image

    print("初始化 PaddleOCR-VL v1 (0.9B)...", flush=True)
    ocr = PaddleOCRVL(
        pipeline_version="v1",
        device="gpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    lines = []
    lines.append("# PaddleOCR-VL (0.9B) 识别结果\n")
    lines.append(f"> 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    import paddle
    lines.append(f"> PaddlePaddle: {paddle.__version__}")
    lines.append(f"> 模型: PaddleOCR-VL-0.9B + PP-DocLayoutV2\n")
    lines.append("---\n")

    total_time = 0.0

    for img_name in TEST_IMAGES:
        img_path = TEST_DIR / img_name
        if not img_path.exists():
            lines.append(f"## {img_name}\n\n**文件不存在**\n\n---\n")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        print(f"识别: {img_name} ({w}x{h})...", flush=True)

        start = time.perf_counter()
        result = ocr.predict(str(img_path))
        elapsed = time.perf_counter() - start
        total_time += elapsed

        lines.append(f"## {img_name}\n")
        lines.append(f"- 图片尺寸: {w}x{h}")
        lines.append(f"- 耗时: {elapsed:.3f}s\n")

        if not result:
            lines.append("**无结果**\n")
        else:
            res = result[0]
            # PaddleOCR-VL 输出: markdown 文本 + parsing_res_list
            res_dict = res.json if hasattr(res, "json") else {}

            # 1. Markdown 全文
            markdown_text = ""
            if isinstance(res_dict, dict):
                # 尝试多种可能的 key
                markdown_text = res_dict.get("markdown", "") or res_dict.get("res", {}).get("markdown", "")

            if markdown_text:
                lines.append("### Markdown 输出\n")
                lines.append("```markdown")
                lines.append(markdown_text.strip())
                lines.append("```\n")

            # 2. 逐块解析结果
            parsing_list = []
            if isinstance(res_dict, dict):
                parsing_list = res_dict.get("parsing_res_list", []) or res_dict.get("res", {}).get("parsing_res_list", [])

            if parsing_list:
                lines.append(f"### 逐块解析（{len(parsing_list)} 块）\n")
                lines.append("| # | 类型 | 内容 | 坐标 |")
                lines.append("|---|---|---|---|")
                for i, block in enumerate(parsing_list):
                    label = block.get("block_label", "?")
                    content = block.get("block_content", "")
                    bbox = block.get("block_bbox", [])
                    if bbox and len(bbox) >= 4:
                        coord = f"({int(bbox[0])},{int(bbox[1])})→({int(bbox[2])},{int(bbox[3])})"
                    else:
                        coord = ""
                    content_escaped = content.replace("|", "\\|").replace("\n", " ")[:120]
                    lines.append(f"| {i+1} | {label} | {content_escaped} | {coord} |")
                lines.append("")

                # 3. 单独列出所有公式块
                formula_blocks = [b for b in parsing_list if b.get("block_label") == "formula"]
                if formula_blocks:
                    lines.append(f"### 公式 LaTeX（{len(formula_blocks)} 个）\n")
                    for i, fb in enumerate(formula_blocks):
                        lines.append(f"**公式 {i+1}**:")
                        lines.append("```latex")
                        lines.append(fb.get("block_content", ""))
                        lines.append("```\n")

            # 如果没有 parsing_res_list，尝试打印整个 json 的 key
            if not markdown_text and not parsing_list:
                lines.append("### 原始输出 keys\n")
                lines.append(f"```\n{list(res_dict.keys()) if isinstance(res_dict, dict) else type(res_dict)}\n```\n")

        lines.append("---\n")
        n = len(parsing_list) if parsing_list else 0
        print(f"  完成: {n} 块, {elapsed:.3f}s", flush=True)

    lines.append(f"\n## 汇总\n")
    lines.append(f"- 总耗时: {total_time:.3f}s")
    lines.append(f"- 平均每帧: {total_time/len(TEST_IMAGES):.3f}s")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_FILE}", flush=True)
    print(f"总耗时: {total_time:.3f}s, 平均: {total_time/len(TEST_IMAGES):.3f}s/帧", flush=True)


if __name__ == "__main__":
    main()
