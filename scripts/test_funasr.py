"""
FunASR Paraformer 中文识别测试
完全参照项目 adapters/asr/funasr.py 的写法：
  paraformer-zh + fsmn-vad + ct-punc，model_revision=v2.0.4
结果保存到 tests/FunASR.txt
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from funasr import AutoModel


def main():
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "test.wav")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "FunASR.txt")

    print("加载 FunASR 模型 (paraformer-zh + fsmn-vad + ct-punc)...")
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
    )

    print(f"开始识别: {wav_path}")
    start = time.time()
    res = model.generate(
        input=wav_path,
        batch_size_s=300,
    )
    elapsed = time.time() - start

    text = res[0].get("text", "") if res else ""
    print(f"识别完成, 耗时 {elapsed:.1f}s, 文本长度 {len(text)} 字符")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# FunASR Paraformer 识别结果\n")
        f.write(f"# 模型: paraformer-zh + fsmn-vad + ct-punc (v2.0.4)\n")
        f.write(f"# 耗时: {elapsed:.1f}s\n")
        f.write(f"# 文本长度: {len(text)} 字符\n\n")
        f.write(text)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
