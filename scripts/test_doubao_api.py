"""
豆包（火山引擎）API 验证脚本
验证 R-001 报告中的关键结论：
1. LiteLLM 走 Chat Completions 格式的基本调用
2. LiteLLM 是否能通过 extra_body 开启 thinking/reasoning
3. 直连 Responses API 的流式响应格式（reasoning 事件、output_text 事件）
4. 多模态输入格式（input_text / input_image）

输出结果保存到 scripts/test_doubao_api_result.txt
"""

import sys
import os
import json
import time
import requests

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager

# 测试用的问题（需要推理的数学题）
TEST_QUESTION = "已知集合 A={1,2,3}, B={2,3,4}，求 A∩B 和 A∪B。"
MODEL = "doubao-seed-2-1-pro-260628"

result_lines = []


def log(msg=""):
    print(msg)
    result_lines.append(str(msg))


def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def load_config():
    """从 config.yaml 加载豆包配置"""
    mgr = ConfigManager("config.yaml")
    cfg = mgr.load()
    model_cfg = cfg.llm.models[MODEL]
    provider_cfg = cfg.llm.providers[model_cfg.provider]
    return model_cfg, provider_cfg


def test_litellm_basic(model_cfg, provider_cfg):
    """测试1: LiteLLM 基本文本生成（当前代码路径）"""
    section("测试1: LiteLLM Chat Completions 基本调用")
    try:
        import litellm

        litellm_model = f"{provider_cfg.litellm_prefix}/{model_cfg.name}"
        log(f"模型: {litellm_model}")
        log(f"base_url: {provider_cfg.base_url}")

        start = time.time()
        response = litellm.completion(
            model=litellm_model,
            api_base=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            messages=[
                {"role": "user", "content": TEST_QUESTION},
            ],
            temperature=0.1,
            max_tokens=2048,
            timeout=60,
        )
        elapsed = time.time() - start

        log(f"耗时: {elapsed:.1f}s")
        log(f"finish_reason: {response.choices[0].finish_reason}")
        log(f"content:\n{response.choices[0].message.content}")
        if response.usage:
            log(f"usage: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}, "
                f"total={response.usage.total_tokens}")

        # 检查是否有 reasoning 相关字段
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            log(f"\n[发现] reasoning_content 字段存在，长度={len(reasoning)}")
            log(f"reasoning_content 前200字: {reasoning[:200]}")
        else:
            log("\n[发现] 无 reasoning_content 字段（Chat Completions 格式不返回思维链）")

        # 检查 message 的所有属性
        msg_attrs = [a for a in dir(msg) if not a.startswith("_") and a not in ("content", "role")]
        log(f"message 其他属性: {msg_attrs}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_litellm_thinking_extra_body(model_cfg, provider_cfg):
    """测试2: LiteLLM 通过 extra_body 开启 thinking"""
    section("测试2: LiteLLM 通过 extra_body 开启 thinking")
    try:
        import litellm

        litellm_model = f"{provider_cfg.litellm_prefix}/{model_cfg.name}"

        # 尝试 Responses API 风格的 thinking 参数
        log("尝试 extra_body={'thinking': {'type': 'enabled'}, 'reasoning': {'effort': 'high'}}")
        start = time.time()
        try:
            response = litellm.completion(
                model=litellm_model,
                api_base=provider_cfg.base_url,
                api_key=provider_cfg.api_key,
                messages=[
                    {"role": "user", "content": TEST_QUESTION},
                ],
                temperature=0.1,
                max_tokens=4096,
                timeout=120,
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning": {"effort": "high"},
                },
            )
            elapsed = time.time() - start
            log(f"耗时: {elapsed:.1f}s")
            log(f"finish_reason: {response.choices[0].finish_reason}")
            content = response.choices[0].message.content
            log(f"content 长度: {len(content) if content else 0}")
            log(f"content 前300字:\n{content[:300] if content else '(空)'}")

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                log(f"\n[发现] reasoning_content 存在，长度={len(reasoning)}")
                log(f"reasoning_content 前300字: {reasoning[:300]}")
            else:
                log("\n[发现] 仍无 reasoning_content 字段")

        except Exception as e:
            elapsed = time.time() - start
            log(f"[错误] {type(e).__name__}: {e}")
            log(f"(耗时 {elapsed:.1f}s)")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_responses_api_stream(model_cfg, provider_cfg):
    """测试3: 直连 Responses API，流式，开启 thinking"""
    section("测试3: 直连 Responses API（流式 + thinking）")
    try:
        url = f"{provider_cfg.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {provider_cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg.name,
            "input": TEST_QUESTION,
            "stream": True,
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high"},
            "max_output_tokens": 8192,
        }

        log(f"POST {url}")
        log(f"payload: model={model_cfg.name}, stream=True, thinking=enabled, effort=high")

        start = time.time()
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        log(f"HTTP status: {response.status_code}")

        if response.status_code != 200:
            log(f"[错误] 响应: {response.text[:500]}")
            return

        # 收集所有事件类型和内容
        event_types = {}
        reasoning_parts = []
        output_parts = []
        usage_info = None
        response_id = None

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                event_type = line[7:]
                event_types[event_type] = event_types.get(event_type, 0) + 1
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 记录 response.id
                if "response" in data and isinstance(data["response"], dict):
                    response_id = data["response"].get("id", response_id)
                    if "usage" in data["response"]:
                        usage_info = data["response"]["usage"]

                # 记录事件类型（从 type 字段）
                evt_type = data.get("type", "")
                if evt_type:
                    event_types[evt_type] = event_types.get(evt_type, 0) + 1

                # 收集 reasoning 文本
                if evt_type == "response.reasoning_summary_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        reasoning_parts.append(delta)

                # 收集 output 文本
                if evt_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        output_parts.append(delta)

                # 记录 completed 事件中的 usage
                if evt_type == "response.completed":
                    resp = data.get("response", {})
                    if "usage" in resp:
                        usage_info = resp["usage"]

        elapsed = time.time() - start

        log(f"\n耗时: {elapsed:.1f}s")
        log(f"response_id: {response_id}")
        log(f"\n收到的事件类型统计:")
        for evt, count in sorted(event_types.items()):
            log(f"  {evt}: {count}")

        reasoning_text = "".join(reasoning_parts)
        output_text = "".join(output_parts)

        log(f"\nreasoning_summary 长度: {len(reasoning_text)}")
        if reasoning_text:
            log(f"reasoning_summary 前500字:\n{reasoning_text[:500]}")

        log(f"\noutput_text 长度: {len(output_text)}")
        if output_text:
            log(f"output_text 前500字:\n{output_text[:500]}")

        if usage_info:
            log(f"\nusage: {json.dumps(usage_info, ensure_ascii=False, indent=2)}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_responses_api_nonstream(model_cfg, provider_cfg):
    """测试4: 直连 Responses API，非流式，开启 thinking（验证超时风险）"""
    section("测试4: 直连 Responses API（非流式 + thinking）")
    try:
        url = f"{provider_cfg.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {provider_cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg.name,
            "input": TEST_QUESTION,
            "stream": False,
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high"},
            "max_output_tokens": 8192,
        }

        log(f"POST {url} (stream=False)")
        start = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        elapsed = time.time() - start

        log(f"HTTP status: {response.status_code}")
        log(f"耗时: {elapsed:.1f}s")

        if response.status_code != 200:
            log(f"[错误] 响应: {response.text[:500]}")
            return

        data = response.json()
        log(f"response.id: {data.get('id')}")
        log(f"status: {data.get('status')}")

        # 输出结构
        output = data.get("output", [])
        log(f"\noutput 条目数: {len(output)}")
        for i, item in enumerate(output):
            log(f"  output[{i}].type: {item.get('type')}")
            if item.get("type") == "reasoning":
                summary = item.get("summary", [])
                log(f"    reasoning summary 条目数: {len(summary)}")
                for s in summary[:2]:
                    log(f"    summary type: {s.get('type')}, 文本前200字: {s.get('text', '')[:200]}")
            if item.get("type") == "message":
                content = item.get("content", [])
                for c in content:
                    log(f"    content type: {c.get('type')}, 文本前200字: {c.get('text', '')[:200]}")

        usage = data.get("usage", {})
        if usage:
            log(f"\nusage: {json.dumps(usage, ensure_ascii=False, indent=2)}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_responses_api_multimodal(model_cfg, provider_cfg):
    """测试5: 直连 Responses API 多模态输入（找一张测试图片）"""
    section("测试5: Responses API 多模态输入（图片+文本）")

    # 找一张测试图片
    test_image = None
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
    for candidate in ["印刷体01.png", "印刷体+手写体.png"]:
        path = os.path.join(tests_dir, candidate)
        if os.path.exists(path):
            test_image = path
            break

    if not test_image:
        log("[跳过] 未找到测试图片")
        return

    log(f"测试图片: {test_image}")

    try:
        import base64

        with open(test_image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        url = f"{provider_cfg.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {provider_cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg.name,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": "这张图片里有什么文字和数学公式？请识别出来。"},
                ],
            }],
            "stream": False,
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high"},
            "max_output_tokens": 4096,
        }

        log(f"POST {url} (multimodal, stream=False)")
        start = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        elapsed = time.time() - start

        log(f"HTTP status: {response.status_code}")
        log(f"耗时: {elapsed:.1f}s")

        if response.status_code != 200:
            log(f"[错误] 响应: {response.text[:500]}")
            return

        data = response.json()
        output_text = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        output_text += c.get("text", "")

        log(f"\n模型识别结果:\n{output_text[:800]}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def main():
    log("=" * 70)
    log("  豆包（火山引擎）API 验证报告")
    log(f"  模型: {MODEL}")
    log(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    model_cfg, provider_cfg = load_config()
    log(f"上下文长度: {model_cfg.context_length}")
    log(f"最大输出: {model_cfg.max_output}")

    test_litellm_basic(model_cfg, provider_cfg)
    test_litellm_thinking_extra_body(model_cfg, provider_cfg)
    test_responses_api_stream(model_cfg, provider_cfg)
    test_responses_api_nonstream(model_cfg, provider_cfg)
    test_responses_api_multimodal(model_cfg, provider_cfg)

    # 保存结果
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_doubao_api_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
