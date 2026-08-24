"""
云端 DeepSeek API 适配器
deepseek-chat，128K 上下文。
"""

import os

from adapters.llm.openai_compatible import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """云端 DeepSeek API 适配器（deepseek-chat，128K 上下文）"""

    def __init__(self, config: dict):
        default_config = {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "context_length": 131072,
            "timeout": 120,
        }
        default_config.update(config)
        super().__init__(default_config)
