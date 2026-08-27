"""
DeepSeek API 验证脚本
验证 deepseek-v4-pro 通过 LiteLLM 的调用情况：
1. 基本文本生成
2. 是否支持 reasoning/thinking
3. 流式响应格式
4. 异常处理验证（无效模型名、错误 key 等）

输出结果保存到 scripts/test_deepseek_api_result.txt
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager

TEST_QUESTION = "已知集合 A={1,2,3}, B={2,3,4}，求 A∩B 和 A∪B。"
MODEL = "deepseek-v4-pro"

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
    mgr = ConfigManager("config.yaml")
    cfg = mgr.load()
    model_cfg = cfg.llm.models[MODEL]
    provider_cfg = cfg.llm.providers[model_cfg.provider]
    return model_cfg, provider_cfg


def test_litellm_basic(model_cfg, provider_cfg):
    """测试1: LiteLLM 基本文本生成"""
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

        # 检查 reasoning_content（DeepSeek 特有字段）
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            log(f"\n[发现] reasoning_content 字段存在，长度={len(reasoning)}")
            log(f"reasoning_content 前300字: {reasoning[:300]}")
        else:
            log("\n[发现] 无 reasoning_content 字段")

        msg_attrs = [a for a in dir(msg) if not a.startswith("_") and a not in ("content", "role")]
        log(f"message 其他属性: {msg_attrs}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_litellm_stream(model_cfg, provider_cfg):
    """测试2: LiteLLM 流式调用，检查是否有 reasoning 增量"""
    section("测试2: LiteLLM 流式调用")
    try:
        import litellm

        litellm_model = f"{provider_cfg.litellm_prefix}/{model_cfg.name}"

        start = time.time()
        stream = litellm.completion(
            model=litellm_model,
            api_base=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            messages=[
                {"role": "user", "content": TEST_QUESTION},
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=True,
            stream_options={"include_usage": True},
            timeout=120,
        )

        content_parts = []
        reasoning_parts = []
        usage = None
        finish_reason = None
        delta_attrs_seen = set()

        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    # 记录 delta 的所有属性
                    for attr in dir(delta):
                        if not attr.startswith("_") and attr not in ("content", "role", "model_dump"):
                            delta_attrs_seen.add(attr)

                    dc = getattr(delta, "content", None)
                    if dc:
                        content_parts.append(dc)

                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_parts.append(rc)

                if choices[0].finish_reason:
                    finish_reason = choices[0].finish_reason

            cu = getattr(chunk, "usage", None)
            if cu:
                usage = cu

        elapsed = time.time() - start

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)

        log(f"耗时: {elapsed:.1f}s")
        log(f"finish_reason: {finish_reason}")
        log(f"content 长度: {len(content)}")
        log(f"content 前300字:\n{content[:300]}")
        log(f"\nreasoning_content 长度: {len(reasoning)}")
        if reasoning:
            log(f"reasoning_content 前300字: {reasoning[:300]}")
        log(f"\ndelta 属性: {delta_attrs_seen}")
        if usage:
            log(f"usage: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}")

    except Exception as e:
        log(f"[错误] {type(e).__name__}: {e}")


def test_invalid_model(model_cfg, provider_cfg):
    """测试3: 无效模型名的错误响应"""
    section("测试3: 异常处理 - 无效模型名")
    try:
        import litellm

        litellm_model = f"{provider_cfg.litellm_prefix}/deepseek-nonexistent-model"

        response = litellm.completion(
            model=litellm_model,
            api_base=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            timeout=30,
        )
        log(f"[意外] 请求成功了: {response.choices[0].message.content}")

    except Exception as e:
        log(f"异常类型: {type(e).__name__}")
        log(f"异常消息: {str(e)[:300]}")
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
        if status:
            log(f"HTTP status: {status}")


def main():
    log("=" * 70)
    log("  DeepSeek API 验证报告")
    log(f"  模型: {MODEL}")
    log(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    model_cfg, provider_cfg = load_config()
    log(f"上下文长度: {model_cfg.context_length}")
    log(f"最大输出: {model_cfg.max_output}")

    test_litellm_basic(model_cfg, provider_cfg)
    test_litellm_stream(model_cfg, provider_cfg)
    test_invalid_model(model_cfg, provider_cfg)

    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_deepseek_api_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
