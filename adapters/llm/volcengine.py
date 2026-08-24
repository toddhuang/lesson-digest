"""
火山引擎豆包 API 适配器
OpenAI 兼容，中文教育场景优化。使用 responses.create 接口（seed 系列新模型专用）。
"""

import os

from utils.models import LLMResponse, TokenUsage
from adapters.llm.openai_compatible import OpenAICompatibleAdapter


class VolcengineAdapter(OpenAICompatibleAdapter):
    """火山引擎豆包 API 适配器（OpenAI 兼容，中文教育场景优化）

    支持模型：doubao-seed-2.0-pro / doubao-1.5-pro / doubao-1.5-lite 等
    API 地址：https://ark.cn-beijing.volces.com/api/v3
    """

    def __init__(self, config: dict):
        default_config = {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-seed-2-1-pro-260628",
            "api_key": os.environ.get("VOLCENGINE_API_KEY", ""),
            "context_length": 131072,
            "timeout": 120,
        }
        default_config.update(config)
        super().__init__(default_config)

    def _convert_messages_to_responses_input(self, messages: list) -> list:
        """将传统 chat.completions 的 messages 格式转换为 responses.create 的 input 格式

        responses.create 的 input 格式：
        [
            {"role": "system", "content": [{"type": "input_text", "text": "..."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "..."}]},
        ]
        """
        input_list = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                input_list.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                })
            elif isinstance(content, list):
                converted_content = []
                for item in content:
                    if item.get("type") == "text":
                        converted_content.append({"type": "input_text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        converted_content.append({"type": "input_image", "image_url": item.get("image_url", "")})
                    else:
                        converted_content.append(item)
                input_list.append({"role": role, "content": converted_content})
        return input_list

    def _extract_responses_output(self, response) -> str:
        """从 responses.create 响应中提取输出文本

        兼容多种返回格式：
        - response.output_text
        - response.output[0].content[0].text
        - response.output[*].content[*].text
        """
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        output = getattr(response, "output", None)
        if output and isinstance(output, list):
            texts = []
            for item in output:
                content = getattr(item, "content", None)
                if content and isinstance(content, list):
                    for c in content:
                        text = getattr(c, "text", None)
                        if text:
                            texts.append(text)
            if texts:
                return "".join(texts)

        if isinstance(response, dict):
            if "output_text" in response:
                return response["output_text"]
            if "output" in response and isinstance(response["output"], list):
                texts = []
                for item in response["output"]:
                    if "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if "text" in c:
                                texts.append(c["text"])
                if texts:
                    return "".join(texts)

        return ""

    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse:
        """非流式对话（使用 responses.create 接口，seed 系列新模型专用）

        注意：seed 系列模型默认开启深度思考，会消耗大量 output token。
        通过 reasoning={"effort": "none"} 禁用思考过程，直接输出答案。
        """
        from utils.logger import setup_logger
        logger = setup_logger("LLM")

        responses_input = self._convert_messages_to_responses_input(messages)

        try:
            response = self._client.responses.create(
                model=self.model,
                input=responses_input,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning={"effort": "none"},
            )
        except Exception as e:
            self._handle_openai_error(e, context="responses.create调用")

        content = getattr(response, "output_text", "")
        if not content:
            content = self._extract_responses_output(response)

        usage = getattr(response, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
            completion_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        if not content:
            logger.warning(f"火山引擎豆包返回空内容，status={getattr(response, 'status', 'unknown')}")

        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            finish_reason="stop",
        )
