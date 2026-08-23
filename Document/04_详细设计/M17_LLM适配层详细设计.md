# M17 LLM 适配层 — 详细设计

> 接口文档：`03_接口设计/M17_LLM适配层接口.md`
> 模块编号：M17

---

## 一、模块结构

```
adapters/
├── __init__.py
└── llm_adapter.py    # LLM 适配层（抽象基类 + VLLM + DeepSeek 实现）
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

## 三、VLLMAdapter 实现（本地 vLLM）

### 3.1 初始化

```python
class VLLMAdapter(LLMAdapter):
    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self._context_length = config.get("context_length", 8192)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="EMPTY",  # vLLM 不需要 API Key
            )
        return self._client
```

### 3.2 chat（非流式）

```python
def chat(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> LLMResponse:
    client = self._get_client()
    response = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        **kwargs
    )
    return LLMResponse(
        content=response.choices[0].message.content,
        model=response.model,
        usage=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        finish_reason=response.choices[0].finish_reason,
    )
```

### 3.3 chat_stream（流式）

```python
def chat_stream(self, messages, temperature=0.3, top_p=0.9, max_tokens=2000, **kwargs) -> Iterator[LLMChunk]:
    client = self._get_client()
    stream = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
        **kwargs
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        finish_reason = chunk.choices[0].finish_reason
        usage = None
        if chunk.usage:
            usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        yield LLMChunk(
            delta_content=delta,
            finish_reason=finish_reason,
            usage=usage,
        )
```

### 3.4 count_tokens

```python
def count_tokens(self, text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
```

### 3.5 其他方法

```python
def get_context_length(self) -> int:
    return self._context_length

def get_model_name(self) -> str:
    return self.model
```

---

## 四、DeepSeekAdapter 实现（云端 DeepSeek API）

### 4.1 初始化

```python
class DeepSeekAdapter(LLMAdapter):
    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.api_key = config["api_key"]
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

### 4.2 chat / chat_stream / count_tokens

与 VLLMAdapter 完全相同（都使用 OpenAI SDK），区别仅在于：
- `base_url`：DeepSeek API 地址
- `api_key`：DeepSeek API Key
- `_context_length`：131072（128K）

为避免代码重复，可提取公共基类 `OpenAICompatibleAdapter`：

```python
class OpenAICompatibleAdapter(LLMAdapter):
    """OpenAI 兼容 API 的公共实现（vLLM 和 DeepSeek 都兼容）"""
    def __init__(self, base_url, model, api_key, context_length):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._context_length = context_length
        self._client = None

    # chat / chat_stream / count_tokens 公共实现
    ...

class VLLMAdapter(OpenAICompatibleAdapter):
    def __init__(self, config):
        super().__init__(
            base_url=config["base_url"],
            model=config["model"],
            api_key="EMPTY",
            context_length=config.get("context_length", 8192),
        )

class DeepSeekAdapter(OpenAICompatibleAdapter):
    def __init__(self, config):
        super().__init__(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
            context_length=config.get("context_length", 131072),
        )
```

---

## 五、适配器工厂

```python
def create_adapter(backend: str, config: dict) -> LLMAdapter:
    adapters = {
        "vllm": VLLMAdapter,
        "deepseek": DeepSeekAdapter,
        # 后续新增适配器在此注册
    }
    if backend not in adapters:
        raise ValueError(f"不支持的 LLM 后端: {backend}")
    return adapters[backend](config)
```

---

## 六、异常映射

OpenAI SDK 抛出的异常需要映射为项目自定义异常：

```python
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

def _map_openai_error(e: Exception) -> Exception:
    if isinstance(e, RateLimitError):
        return LLMRateLimitError(str(e))
    elif isinstance(e, APITimeoutError):
        return LLMTimeoutError(str(e))
    elif isinstance(e, APIConnectionError):
        return LLMError(f"连接失败: {e}")
    elif isinstance(e, APIError):
        status_code = getattr(e, "status_code", 500)
        if status_code >= 500:
            return LLMServerError(str(e))
        else:
            return LLMClientError(str(e))
    else:
        return LLMError(str(e))
```

在 chat 方法中捕获并映射：
```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    raise _map_openai_error(e) from e
```

---

## 七、可扩展适配器

| 适配器 | 服务 | 上下文 | API 兼容 |
|---|---|---|---|
| VLLMAdapter | 本地 vLLM | 8K | OpenAI 兼容 |
| DeepSeekAdapter | DeepSeek API | 128K | OpenAI 兼容 |
| OpenAIAdapter | OpenAI API | 128K | OpenAI 原生 |
| QwenCloudAdapter | 通义千问 | 32K+ | OpenAI 兼容 |

所有 OpenAI 兼容的服务都可以继承 `OpenAICompatibleAdapter`，只需配置不同的 base_url 和 api_key。

---

## 八、测试要点

1. VLLM 非流式调用
2. VLLM 流式调用
3. DeepSeek 非流式调用
4. DeepSeek 流式调用
5. token 计数
6. 上下文长度获取
7. 异常映射（超时/429/5xx/4xx）
8. 适配器工厂创建
9. 无效后端的错误处理
10. 流式响应的增量内容拼接
11. API Key 从环境变量读取
12. 连接失败的错误处理
