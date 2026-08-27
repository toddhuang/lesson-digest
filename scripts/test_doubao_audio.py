"""
豆包音频理解识别测试
通过火山方舟 Responses API + Files API 上传音频并识别
使用 doubao-seed-2-0-lite 模型的音频理解能力
结果保存到 tests/豆包.txt
"""

import sys
import os
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

API_KEY = "ark-4611bf4a-ec7d-4a24-ab2b-cbc8aeb1abaf-15268"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-lite-260428"


def upload_file(file_path: str) -> str:
    """通过 Files API 上传音频文件，返回 file_id"""
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with open(file_path, "rb") as f:
        files = {
            "purpose": (None, "user_data"),
            "file": (os.path.basename(file_path), f, "audio/wav"),
        }
        resp = requests.post(url, headers=headers, files=files, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"文件上传失败: {resp.status_code} {resp.text}")
    data = resp.json()
    file_id = data.get("id", "")
    print(f"文件上传成功: {file_id}, status={data.get('status')}")
    return file_id


def transcribe_audio(file_id: str) -> str:
    """通过 Responses API 识别音频"""
    url = f"{BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "file_id": file_id},
                    {"type": "input_text", "text": "请完整识别这段音频中的中文语音内容，逐字转写，不要遗漏，不要总结，不要添加任何解释。"},
                ],
            }
        ],
    }

    print("正在调用豆包音频理解 API（25分钟音频可能需要几分钟）...")
    resp = requests.post(url, headers=headers, json=payload, timeout=900)
    if resp.status_code != 200:
        raise RuntimeError(f"识别失败: {resp.status_code} {resp.text}")

    data = resp.json()
    # Responses API: output 数组中 type=message 的 content 里找 output_text
    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")
    return output_text


def main():
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "test.wav")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "豆包.txt")

    file_size_mb = os.path.getsize(wav_path) / 1024 / 1024
    print(f"音频文件: {wav_path} ({file_size_mb:.1f} MB)")

    start = time.time()

    # 上传文件
    file_id = upload_file(wav_path)

    # 识别
    text = transcribe_audio(file_id)
    elapsed = time.time() - start

    print(f"识别完成, 耗时 {elapsed:.1f}s, 文本长度 {len(text)} 字符")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 豆包音频理解识别结果\n")
        f.write(f"# 模型: {MODEL}\n")
        f.write(f"# API: 火山方舟 Responses API + Files API\n")
        f.write(f"# 耗时: {elapsed:.1f}s\n")
        f.write(f"# 文本长度: {len(text)} 字符\n\n")
        f.write(text)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
