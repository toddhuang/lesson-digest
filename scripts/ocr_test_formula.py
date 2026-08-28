"""公式专家模型（PP-FormulaNet）对 7 张测试图的识别测试

用法（PowerShell，使用 paddle3 环境）：
    venv_paddle3\Scripts\python.exe scripts\ocr_test_formula.py

说明：
- 使用 PaddleOCR 3.7 公式识别产线（版面检测定位公式区域 + PP-FormulaNet 识别输出 LaTeX）
- 首次运行自动下载模型权重（约 85MB，PP-FormulaNet_plus-M）
- 输出：scripts/ocr_results_formula.md

官方文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/formula_recognition.html
"""

import json
import time
from pathlib import Path

from paddleocr import FormulaRecognitionPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR.parent / "tests"
OUTPUT_MD = SCRIPT_DIR / "ocr_results_formula.md"

FORMULA_MODEL = "PP-FormulaNet_plus-M"
IMAGES = [
    "印刷体01.png",
    "印刷体2.png",
    "印刷体3数学题目.png",
    "印刷体+手写体.png",
    "印刷体+手写体2.png",
    "印刷体+手写体3.png",
    "印刷体+手写体4.png",
]


def create_pipeline():
    return FormulaRecognitionPipeline(
        formula_recognition_model_name=FORMULA_MODEL,
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def extract_result(res):
    """从产线结果提取版面框和公式 LaTeX（结构来自 ocr_formula_debug.json 实测）"""
    regions = []
    layout = res.get("layout_det_res", None)
    if layout:
        for box in layout.get("boxes", []):
            regions.append(
                {
                    "label": box["label"],
                    "box": [round(float(v), 1) for v in box["coordinate"]],
                    "score": round(float(box["score"]), 3),
                }
            )
    formulas = []
    for item in res.get("formula_res_list", []):
        polys = item.get("dt_polys", [])
        flat = None
        try:
            if len(polys) and hasattr(polys[0], "__len__"):
                flat = list(polys[0])
            elif len(polys):
                flat = list(polys)
        except TypeError:
            flat = None
        formulas.append(
            {
                "box": [round(float(v), 1) for v in flat] if flat and len(flat) >= 4 else None,
                "latex": item["rec_formula"],
            }
        )
    return regions, formulas


def format_section(name, elapsed, regions, formulas):
    lines = [
        f"## {name}",
        "",
        f"- 耗时：{elapsed:.3f}s",
        f"- 版面检测区域数：{len(regions)}",
        f"- 公式区域数：{len(formulas)}",
        "",
    ]
    if formulas:
        lines += ["### 识别到的公式（LaTeX）", ""]
        for i, f in enumerate(formulas, 1):
            box = f"（box={f['box']}）" if f["box"] else ""
            lines.append(f"{i}. `{f['latex']}` {box}")
        lines.append("")
    if regions:
        lines += ["### 版面检测明细", "", "| 标签 | 置信度 | box |", "|---|---|---|"]
        for r in regions:
            lines.append(f"| {r['label']} | {r['score']} | {r['box']} |")
        lines.append("")
    if not formulas and not regions:
        lines += ["（无公式区域检出）", ""]
    return "\n".join(lines)


def main():
    pipeline = create_pipeline()
    header = [
        "# 公式专家模型识别结果（PaddleOCR 公式识别产线）",
        "",
        f"- 公式模型：{FORMULA_MODEL}",
        "- 产线：版面检测（PP-DocLayout）+ 公式识别，方向/摆正模块已关闭",
        "- 运行环境：venv_paddle3（PaddlePaddle 3.2.2 + CUDA 12.6）",
        "",
    ]
    sections, summary = [], []
    for name in IMAGES:
        path = IMAGE_DIR / name
        if not path.exists():
            sections.append(f"## {name}\n\n- 文件不存在，跳过\n")
            summary.append((name, 0, None))
            continue
        try:
            started = time.perf_counter()
            results = pipeline.predict(str(path))
            elapsed = time.perf_counter() - started
            res = results[0]
            regions, formulas = extract_result(res)
        except RuntimeError as exc:
            sections.append(f"## {name}\n\n- 推理失败：{exc}\n")
            summary.append((name, 0, None))
            continue
        except OSError as exc:
            sections.append(f"## {name}\n\n- 文件/模型读取失败：{exc}\n")
            summary.append((name, 0, None))
            continue
        sections.append(format_section(name, elapsed, regions, formulas))
        summary.append((name, len(formulas), elapsed))
        print(f"{name}: {len(formulas)} 个公式, {elapsed:.3f}s")

    table = ["## 汇总", "", "| 图片 | 公式数 | 耗时s |", "|---|---|---|"]
    for name, n, elapsed in summary:
        cost = f"{elapsed:.3f}" if elapsed is not None else "-"
        table.append(f"| {name} | {n} | {cost} |")
    table.append("")

    OUTPUT_MD.write_text("\n".join(header + sections + table), encoding="utf-8")
    print(f"结果已写入 {OUTPUT_MD}")


if __name__ == "__main__":
    main()
