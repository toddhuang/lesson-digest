# M17 LLM 适配层 — 详细设计

> 接口文档：`03_接口设计/M17_LLM适配层接口.md`
> 模块编号：M17

---

## 一、模块结构

```
adapters/
├── __init__.py
└── llm_adapter.py    # LLM 适配层（抽象基类 + Volcengine + DeepSeek + Mock 实现）
```

---

## 二、抽象基类 LLMAdapter

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Iterator, Optional

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str

@dataclass
class LLMChunk:
    delta_content: str
    finish_reason: Optional[str]
    usage: Optional[TokenUsage]

class LLMAdapter(ABC):
    @abstractmethod
    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse: ...

    @abstractmethod
    def chat_stream(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> Iterator[LLMChunk]: ...

    @abstractmethod
    def get_context_length(self) -> int: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def get_model_name(self) -> str: ...
```

---

## 三、VolcengineAdapter 实现（豆包，默认服务商）

### 3.1 初始化

```python
class VolcengineAdapter(LLMAdapter):
    def __init__(self, config: dict):
        self.base_url = config["base_url"]  # https://ark.cn-beijing.volces.com/api/v3
        self.api_key = config["api_key"]
        self.model = config["model"]  # doubao-seed-2-1-pro-260628
        self._context_length = config.get("context_length", 131072)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client
```

### 3.2 chat 实现（responses.create 接口）

> **重要**：豆包 seed 系列模型**不支持** `chat.completions.create` 接口（返回 404），必须使用 `responses.create` 接口。且必须**禁用思考过程**（`reasoning={"effort": "none"}`），否则 100 token 全在思考过程中，`status=incomplete`，`output_text` 为空。

```python
def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs):
    client = self._get_client()
    
    # 转换 messages 格式为 responses.create 的 input 格式
    input_messages = []
    for msg in messages:
        input_messages.append({
            "role": msg["role"],
            "content": [{"type": "input_text", "text": msg["content"]}]
        })
    
    response = client.responses.create(
        model=self.model,
        input=input_messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        reasoning={"effort": "none"},  # 禁用思考过程，必须设置
    )
    
    # 解析响应
    content = response.output_text or ""
    usage = TokenUsage(
        prompt_tokens=response.usage.input_tokens if response.usage else 0,
        completion_tokens=response.usage.output_tokens if response.usage else 0,
        total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
    )
    
    return LLMResponse(
        content=content,
        model=self.model,
        usage=usage,
        finish_reason="stop",
    )
```

### 3.3 其他方法

```python
def get_context_length(self) -> int:
    return self._context_length

def count_tokens(self, text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_model_name(self) -> str:
    return self.model
```

---

## 四、DeepSeekAdapter 实现（备选服务商）

### 4.1 初始化

```python
class DeepSeekAdapter(LLMAdapter):
    def __init__(self, config: dict):
        self.base_url = config["base_url"]  # https://api.deepseek.com
        self.api_key = config["api_key"]
        self.model = config["model"]  # deepseek-v4-flash
        self._context_length = config.get("context_length", 131072)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client
```

### 4.2 chat 实现（chat.completions.create 接口）

> DeepSeek 支持标准 `chat.completions.create` 接口。必须**禁用思考过程**（`extra_body={"thinking": {"type": "disabled"}}`）。

```python
def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs):
    client = self._get_client()
    
    response = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},  # 禁用思考过程
    )
    
    content = response.choices[0].message.content or ""
    usage = TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
    
    return LLMResponse(
        content=content,
        model=self.model,
        usage=usage,
        finish_reason=response.choices[0].finish_reason or "stop",
    )
```

---

## 五、MockAdapter 实现（开发测试用）

```python
class MockAdapter(LLMAdapter):
    """Mock 适配器，返回假数据，用于开发测试和链路验证"""
    
    def __init__(self, config: dict = None):
        self.model = "mock-model"
        self._context_length = 131072

    def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs):
        # 根据消息内容返回不同的假数据
        content = "这是 Mock 适配器返回的假数据。"
        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
        )

    def get_context_length(self) -> int:
        return self._context_length

    def count_tokens(self, text: str) -> int:
        return len(text) // 2  # 粗略估算

    def get_model_name(self) -> str:
        return self.model
```

---

## 六、适配器工厂

```python
def create_adapter(provider_name: str, config: dict) -> LLMAdapter:
    """根据服务商名称创建对应适配器"""
    adapters = {
        "volcengine": VolcengineAdapter,
        "deepseek": DeepSeekAdapter,
        "mock": MockAdapter,
    }
    if provider_name not in adapters:
        raise ValueError(f"不支持的服务商: {provider_name}，支持: {list(adapters.keys())}")
    return adapters[provider_name](config)
```

---

## 七、关键踩坑记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 豆包 seed 系列 404 | 不支持 chat.completions.create 接口 | 使用 responses.create 接口 |
| 豆包 output_text 为空 | 思考过程占用全部 token，status=incomplete | 设置 reasoning={"effort": "none"} 禁用思考 |
| DeepSeek 思考过程占用 token | 默认启用思考 | 设置 extra_body={"thinking": {"type": "disabled"}} |
| 豆包 responses.create 输入格式不同 | 需要 input 数组而非 messages | 转换为 [{"role":..., "content":[{"type":"input_text","text":...}]}] |

---

## 八、依赖关系

- **被依赖**：M11 LLM 客户端模块
- **依赖**：无（底层适配层）
- **第三方依赖**：openai（SDK，兼容豆包和 DeepSeek 的 OpenAI 兼容接口）、tiktoken（token 计数）
