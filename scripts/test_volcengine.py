"""
火山引擎豆包 API 连通性测试脚本
测试云端豆包大模型 API 是否正常可用。

用法：
    python test_volcengine.py [--model doubao-seed-2.0-pro]

从 .env 或环境变量读取 VOLCENGINE_API_KEY、VOLCENGINE_MODEL、VOLCENGINE_BASE_URL。
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
    """测试豆包聊天接口（使用 responses.create，seed 系列新模型专用）

    Args:
        api_key: 火山引擎 API Key
        base_url: API 地址
        model: 模型名称

    Returns:
        True 表示接口正常
    """
    print(f"  端点: {base_url.rstrip('/')}/responses")
    print(f"  模型: {model}")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "用一句话回答：1+1等于几？"}],
                }
            ],
            max_output_tokens=2000,
            temperature=0.1,
            # 尝试禁用思考模式（火山引擎 seed 系列可能需要此参数）
            reasoning={"effort": "none"},
        )

        # 提取输出文本
        content = getattr(response, "output_text", "")
        if not content:
            output = getattr(response, "output", [])
            if output:
                for item in output:
                    summary = getattr(item, "summary", None)
                    if summary and isinstance(summary, list):
                        for s in summary:
                            text = getattr(s, "text", "")
                            if text:
                                content += text

        # 提取 token 使用
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else 0

        print(f"  ✓ 响应: {content.strip()}")
        print(f"  Token: input={input_tokens}, output={output_tokens}, total={total_tokens}")
        return True
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        if "404" in str(e) or "Not Found" in str(e):
            print("    → 模型不存在或接口不支持，请检查模型名")
        elif "401" in str(e) or "Unauthorized" in str(e):
            print("    → API Key 无效，请检查 VOLCENGINE_API_KEY")
        return False


def test_models(api_key: str, base_url: str) -> list:
    """测试模型列表端点

    Args:
        api_key: 火山引擎 API Key
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
        print(f"  ⚠ 模型列表端点不可用: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="火山引擎豆包 API 连通性测试")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称（默认从环境变量 VOLCENGINE_MODEL 读取）")
    parser.add_argument("--url", type=str, default=None,
                        help="API 地址（默认从环境变量 VOLCENGINE_BASE_URL 读取）")
    args = parser.parse_args()

    # 加载 .env
    load_env()

    # 获取配置
    api_key = os.environ.get("VOLCENGINE_API_KEY", "")
    base_url = args.url or os.environ.get("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = args.model or os.environ.get("VOLCENGINE_MODEL", "doubao-seed-2-1-pro-260628")

    print("=" * 60)
    print("火山引擎豆包 API 连通性测试")
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
        print("错误: 未设置 VOLCENGINE_API_KEY")
        print()
        print("请在项目根目录创建 .env 文件，添加：")
        print("  VOLCENGINE_API_KEY=你的火山引擎API_Key")
        print()
        print("获取地址: https://ai.volcengine.com/ （火山方舟控制台）")
        sys.exit(1)

    # 测试1: 聊天补全
    print("[1/2] 聊天补全测试")
    chat_ok = test_chat(api_key, base_url, model)
    print()

    # 测试2: 模型列表
    test_models(api_key, base_url)
    print()

    # 总结
    print("=" * 60)
    if chat_ok:
        print("✓ 聊天接口测试通过，豆包 API 正常可用")
    else:
        print("✗ 聊天接口测试失败，请检查上述错误信息")
    print("=" * 60)

    sys.exit(0 if chat_ok else 1)


if __name__ == "__main__":
    main()
