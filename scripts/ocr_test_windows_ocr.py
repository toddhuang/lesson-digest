"""Windows 自带 OCR（Windows.Media.Ocr）对 7 张测试图的识别测试

用法（PowerShell，任意 Python 3.11 环境即可，与其他依赖无冲突）：
    pip install winsdk
    python scripts/ocr_test_windows_ocr.py

前提（若引擎创建失败按此操作）：
    设置 > 时间和语言 > 语言和区域 > 添加语言：中文(简体)
    设置 > 应用 > 可选功能 > 添加可选功能 > 搜索：文本识别(OCR)

输出：scripts/ocr_results_windows_ocr.md（格式与 ocr_results_*.md 系列一致）
"""

import asyncio
import time
from pathlib import Path

from winsdk.windows.globalization import Language
from winsdk.windows.graphics.imaging import BitmapDecoder
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.storage import FileAccessMode, StorageFile

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR.parent / "tests"
OUTPUT_MD = SCRIPT_DIR / "ocr_results_windows_ocr.md"

TARGET_LANG = "zh-Hans"
IMAGES = [
    "印刷体01.png",
    "印刷体2.png",
    "印刷体3数学题目.png",
    "印刷体+手写体.png",
    "印刷体+手写体2.png",
    "印刷体+手写体3.png",
    "印刷体+手写体4.png",
]


def create_engine():
    available = [str(lang) for lang in OcrEngine.available_recognizer_languages]
    engine = OcrEngine.try_create_from_language(Language(TARGET_LANG))
    if engine is None:
        engine = OcrEngine.try_create_from_user_profile_languages()
    return engine, available


async def recognize_image(engine, image_path):
    file = await StorageFile.get_file_from_path_async(str(image_path))
    stream = await file.open_async(FileAccessMode.READ)
    try:
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        started = time.perf_counter()
        result = await engine.recognize_async(bitmap)
        elapsed = time.perf_counter() - started
    finally:
        stream.close()
    return result, elapsed


def line_rect(line):
    """OcrLine 无 bounding_rect，用行内词坐标合并出范围；个别词矩形不可访问时跳过"""
    bounds = None
    for word in line.words:
        try:
            r = word.bounding_rect
        except OSError:
            continue
        wx0, wy0, wx1, wy1 = r.x, r.y, r.x + r.width, r.y + r.height
        if bounds is None:
            bounds = [wx0, wy0, wx1, wy1]
        else:
            bounds[0] = min(bounds[0], wx0)
            bounds[1] = min(bounds[1], wy0)
            bounds[2] = max(bounds[2], wx1)
            bounds[3] = max(bounds[3], wy1)
    if bounds is None:
        return None
    return int(bounds[0]), int(bounds[1]), int(bounds[2] - bounds[0]), int(bounds[3] - bounds[1])


def format_result(image_name, result, elapsed):
    lines = result.lines
    section = [
        f"## {image_name}",
        "",
        f"- 识别行数：{len(lines)}",
        f"- 耗时：{elapsed:.3f}s",
        "",
        "### 全文",
        "",
        " ".join(line.text for line in lines),
        "",
        "### 行明细（含坐标）",
        "",
        "| # | 文本 | x,y,w,h |",
        "|---|---|---|",
    ]
    for idx, line in enumerate(lines, 1):
        text = line.text.replace("|", "\\|")
        rect = line_rect(line)
        rect_desc = f"{rect[0]},{rect[1]},{rect[2]},{rect[3]}" if rect else "-"
        section.append(f"| {idx} | {text} | {rect_desc} |")
    section.append("")
    return "\n".join(section), sum(len(line.text) for line in lines)


async def main():
    engine, available = create_engine()
    if engine is None:
        msg = [
            "# Windows 自带 OCR 测试结果",
            "",
            f"**引擎创建失败**。系统可用识别语言：{available or '无'}",
            "",
            "请先安装中文 OCR 语言包：",
            "设置 > 时间和语言 > 语言和区域 > 添加语言：中文(简体)；",
            "设置 > 应用 > 可选功能 > 添加可选功能 > 搜索：文本识别(OCR)。",
        ]
        OUTPUT_MD.write_text("\n".join(msg), encoding="utf-8")
        print(f"引擎创建失败，诊断信息已写入 {OUTPUT_MD}")
        return

    header = [
        "# Windows 自带 OCR（Windows.Media.Ocr）识别结果",
        "",
        f"- 识别语言：{str(engine.recognizer_language)}",
        f"- 系统可用识别语言：{available}",
        f"- 引擎最大图像边长限制：{OcrEngine.max_image_dimension}px（测试图 2560x1440，未超限）",
        "",
    ]

    sections, summary = [], []
    for name in IMAGES:
        path = IMAGE_DIR / name
        if not path.exists():
            sections.append(f"## {name}\n\n- 文件不存在，跳过\n")
            summary.append((name, 0, 0, None))
            continue
        try:
            result, elapsed = await recognize_image(engine, path)
        except OSError as exc:
            sections.append(f"## {name}\n\n- 文件读取/解码失败：{exc}\n")
            summary.append((name, 0, 0, None))
            continue
        except RuntimeError as exc:
            sections.append(f"## {name}\n\n- OCR 引擎调用失败：{exc}\n")
            summary.append((name, 0, 0, None))
            continue
        section, char_count = format_result(name, result, elapsed)
        sections.append(section)
        summary.append((name, len(result.lines), char_count, elapsed))
        print(f"{name}: {len(result.lines)} 行, {elapsed:.3f}s")

    table = ["## 汇总", "", "| 图片 | 行数 | 字符数 | 耗时s |", "|---|---|---|---|"]
    for name, line_count, char_count, elapsed in summary:
        cost = f"{elapsed:.3f}" if elapsed is not None else "-"
        table.append(f"| {name} | {line_count} | {char_count} | {cost} |")
    table.append("")

    OUTPUT_MD.write_text("\n".join(header + sections + table), encoding="utf-8")
    print(f"结果已写入 {OUTPUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
