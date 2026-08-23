"""
DeepSeek API 连通性测试脚本
测试云端 DeepSeek API 是否正常可用。

用法：
    python test_deepseek.py [--model deepseek-chat]

从 .env 或环境变量读取 DEEPSEEK_API_KEY、DEEPSEEK_MODEL、DEEPSEEK_BASE_URL。
"""

import os
import sys
import json
import argparse


def load_env():
    """加载 .env 文件中的环境变量（从项目根目录加载）"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


def test_chat(api_key: str, base_url: str, model: str) -> bool:
    """测试 DeepSeek 聊天补全接口

    Args:
        api_key: DeepSeek API Key
        base_url: API 地址
        model: 模型名称

    Returns:
        True 表示接口正常
    """
    chat_url = base_url.rstrip("/") + "/chat/completions"
    print(f"  端点: {chat_url}")
    print(f"  模型: {model}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手，用一句话回答问题。"},
            {"role": "user", "content": "用一句话回答：1+1等于几？"},
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "stream": False,
    }

    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            chat_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
        result = json.loads(body)

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        print(f"  ✓ 响应: {content.strip()}")
        print(f"  Token: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        return True
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"  ✗ HTTP 错误 {e.code}: {e.reason}")
        if error_body:
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get("error", {}).get("message", error_body)
                print(f"    详情: {err_msg}")
            except Exception:
                print(f"    详情: {error_body[:200]}")
        if e.code == 401:
            print("    → API Key 无效，请检查 DEEPSEEK_API_KEY")
        elif e.code == 404:
            print("    → 模型不存在或地址错误，请检查 model 和 base_url")
        elif e.code == 429:
            print("    → 请求频率超限或余额不足")
        return False
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        return False


def test_models(api_key: str, base_url: str) -> list:
    """测试模型列表端点（DeepSeek 可能不支持此端点）

    Args:
        api_key: DeepSeek API Key
        base_url: API 地址

    Returns:
        模型名称列表
    """
    models_url = base_url.rstrip("/") + "/models"
    print(f"[2/2] 模型列表: {models_url}")
    try:
        import urllib.request
        req = urllib.request.Request(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"  ✓ 可用模型: {', '.join(models)}")
        else:
            print(f"  ⚠ 未发现模型")
        return models
    except Exception as e:
        print(f"  ⚠ 模型列表端点不可用（DeepSeek 可能不支持）: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="DeepSeek API 连通性测试")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称（默认从环境变量 DEEPSEEK_MODEL 读取）")
    parser.add_argument("--url", type=str, default=None,
                        help="API 地址（默认从环境变量 DEEPSEEK_BASE_URL 读取）")
    args = parser.parse_args()

    # 加载 .env
    load_env()

    # 获取配置
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = args.url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    print("=" * 60)
    print("DeepSeek API 连通性测试")
    print(f"API 地址: {base_url}")
    print(f"模型: {model}")
    if api_key:
        masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"API Key: {masked_key}")
    else:
        print(f"API Key: (未设置)")
    print("=" * 60)
    print()

    if not api_key:
        print("错误: 未设置 DEEPSEEK_API_KEY")
        print()
        print("请在项目根目录创建 .env 文件，添加：")
        print("  DEEPSEEK_API_KEY=sk-你的API_Key")
        print()
        print("获取地址: https://platform.deepseek.com/")
        sys.exit(1)

    # 测试1: 聊天补全
    print("[1/2] 聊天补全测试")
    chat_ok = test_chat(api_key, base_url, model)
    print()

    # 测试2: 模型列表（可选，DeepSeek 可能不支持）
    test_models(api_key, base_url)
    print()

    # 总结
    print("=" * 60)
    if chat_ok:
        print("✓ 聊天接口测试通过，DeepSeek API 正常可用")
    else:
        print("✗ 聊天接口测试失败，请检查上述错误信息")
    print("=" * 60)

    sys.exit(0 if chat_ok else 1)


if __name__ == "__main__":
    main()
