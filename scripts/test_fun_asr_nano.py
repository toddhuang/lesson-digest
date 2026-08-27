"""
Fun-ASR-Nano 中文识别测试
严格参照 FunASR 官方教程文档写法：
  https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md
  - Speech Recognition (Fun-ASR-Nano) 章节
模型权重为 Apache 2.0 许可
结果保存到 tests/Fun-ASR-Nano.txt
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from funasr import AutoModel


def main():
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "test.wav")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "Fun-ASR-Nano.txt")

    print("加载 Fun-ASR-Nano 模型（首次运行需从 HuggingFace 下载约 1.6GB 模型）...")
    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        hub="hf",
        disable_update=True,
    )

    print(f"开始识别: {wav_path}")
    start = time.time()
    res = model.generate(
        input=wav_path,
        cache={},
        batch_size=1,
        language="中文",
    )
    elapsed = time.time() - start

    text = res[0].get("text", "") if res else ""
    timestamps = res[0].get("timestamps", []) if res else []
    print(f"识别完成, 耗时 {elapsed:.1f}s, 文本长度 {len(text)} 字符, 时间戳 {len(timestamps)} 个")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Fun-ASR-Nano 识别结果\n")
        f.write(f"# 模型: FunAudioLLM/Fun-ASR-Nano-2512\n")
        f.write(f"# License: Apache 2.0\n")
        f.write(f"# 耗时: {elapsed:.1f}s\n")
        f.write(f"# 文本长度: {len(text)} 字符\n")
        f.write(f"# 时间戳数量: {len(timestamps)}\n\n")
        f.write(text)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
