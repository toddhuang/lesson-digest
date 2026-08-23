"""
vLLM 服务连通性测试脚本
测试 4090 工作站上的 vLLM 服务是否正常运行。

用法：
    python test_vllm.py [--url http://192.168.x.x:8000/v1]

从 .env 或环境变量读取 VLLM_BASE_URL，也可通过 --url 参数覆盖。
"""

import os
import sys
import json
import argparse
import subprocess


def load_env():
    """加载 .env 文件中的环境变量"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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


def test_health(base_url: str) -> bool:
    """测试 vLLM 健康检查端点

    Args:
        base_url: vLLM 服务地址（如 http://192.168.x.x:8000）

    Returns:
        True 表示服务健康
    """
    # 去掉末尾的 /v1，健康检查端点在根路径
    health_url = base_url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url = health_url.rstrip("/") + "/health"

    print(f"[1/3] 健康检查: {health_url}")
    try:
        import urllib.request
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
        if status == 200:
            print(f"      ✓ 服务健康 (HTTP {status})")
            return True
        else:
            print(f"      ✗ 服务异常 (HTTP {status}): {body}")
            return False
    except Exception as e:
        print(f"      ✗ 连接失败: {e}")
        return False


def test_models(base_url: str) -> list:
    """测试模型列表端点

    Args:
        base_url: vLLM 服务地址

    Returns:
        模型名称列表
    """
    models_url = base_url.rstrip("/") + "/models"
    print(f"[2/3] 模型列表: {models_url}")
    try:
        import urllib.request
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"      ✓ 可用模型: {', '.join(models)}")
        else:
            print(f"      ⚠ 未发现模型")
        return models
    except Exception as e:
        print(f"      ✗ 获取模型列表失败: {e}")
        return []


def test_chat(base_url: str, model: str = None) -> bool:
    """测试聊天补全端点

    Args:
        base_url: vLLM 服务地址
        model: 模型名称（不指定则用第一个可用模型）

    Returns:
        True 表示聊天接口正常
    """
    print(f"[3/3] 聊天补全测试")

    # 如果没有指定模型，尝试从模型列表获取
    if not model:
        models_url = base_url.rstrip("/") + "/models"
        try:
            import urllib.request
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
            if models:
                model = models[0]
        except Exception:
            pass

    if not model:
        model = "default"
        print(f"      ⚠ 无法获取模型名，使用默认值: {model}")

    chat_url = base_url.rstrip("/") + "/chat/completions"
    print(f"      端点: {chat_url}")
    print(f"      模型: {model}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "用一句话回答：1+1等于几？"},
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False,
    }

    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            chat_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
        result = json.loads(body)

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        print(f"      ✓ 响应: {content.strip()}")
        print(f"      Token: prompt={prompt_tokens}, completion={completion_tokens}")
        return True
    except Exception as e:
        print(f"      ✗ 聊天测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="vLLM 服务连通性测试")
    parser.add_argument("--url", type=str, default=None,
                        help="vLLM 服务地址（默认从环境变量 VLLM_BASE_URL 读取）")
    args = parser.parse_args()

    # 加载 .env
    load_env()

    # 获取 base_url
    base_url = args.url or os.environ.get("VLLM_BASE_URL", "")
    if not base_url:
        print("错误: 未指定 vLLM 服务地址")
        print("请设置环境变量 VLLM_BASE_URL 或使用 --url 参数")
        print("示例: python test_vllm.py --url http://192.168.88.16:8000/v1")
        sys.exit(1)

    print("=" * 60)
    print("vLLM 服务连通性测试")
    print(f"目标地址: {base_url}")
    print("=" * 60)
    print()

    # 测试1: 健康检查
    health_ok = test_health(base_url)
    print()

    if not health_ok:
        print("=" * 60)
        print("✗ 服务不可达，请检查：")
        print("  1. 4090 工作站是否开机")
        print("  2. vLLM 服务是否已启动")
        print("  3. 网络是否连通（ping 192.168.88.16）")
        print("  4. 防火墙是否放行 8000 端口")
        print("=" * 60)
        sys.exit(1)

    # 测试2: 模型列表
    models = test_models(base_url)
    print()

    # 测试3: 聊天补全
    chat_ok = test_chat(base_url, model=models[0] if models else None)
    print()

    # 总结
    print("=" * 60)
    if health_ok and chat_ok:
        print("✓ 全部测试通过，vLLM 服务正常运行")
    else:
        print("⚠ 部分测试未通过，请检查上述错误信息")
    print("=" * 60)

    sys.exit(0 if (health_ok and chat_ok) else 1)


if __name__ == "__main__":
    main()
