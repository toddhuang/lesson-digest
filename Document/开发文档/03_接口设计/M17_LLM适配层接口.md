# M17 LLM 适配层 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M17
> 模块职责：定义统一 LLM 接口，封装具体 LLM 服务，隔离第三方库变化

---

## 一、模块概述

LLM 适配层采用**适配器模式（Adapter Pattern）**，定义统一的 LLM 调用接口，封装具体 LLM 服务的 API 差异。当前实现为 VLLMAdapter（本地 vLLM）和 DeepSeekAdapter（云端 DeepSeek API），将来新增 LLM 服务（如 OpenAI、Claude、通义千问）时只需新增适配器，核心业务代码（M11 LLM 客户端模块）零修改。

---

## 二、统一接口定义

### 2.1 LLMAdapter（抽象基类）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Iterator, Optional

@dataclass
class TokenUsage:
    prompt_tokens: int       # 输入 token 数
    completion_tokens: int   # 输出 token 数
    total_tokens: int        # 总 token 数

@dataclass
class LLMResponse:
    content: str             # 生成的文本内容
    model: str               # 使用的模型名
    usage: TokenUsage        # token 使用统计
    finish_reason: str       # 结束原因（stop/length/content_filter）

@dataclass
class LLMChunk:
    delta_content: str       # 增量文本内容
    finish_reason: Optional[str]  # 结束原因（最后一个 chunk 有值）
    usage: Optional[TokenUsage]   # token 使用统计（最后一个 chunk 有值）

class LLMAdapter(ABC):
    """LLM 适配器抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """非流式对话"""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        **kwargs
    ) -> Iterator[LLMChunk]:
        """流式对话，返回响应块迭代器"""
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """返回模型上下文长度（token）"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """统计文本的 token 数"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名"""
        pass

    @abstractmethod
    def rebuild_client(self) -> None:
        """重建底层 HTTP 客户端（用于断线重连，销毁旧连接并建立新连接）"""
        pass
```

---

## 三、当前实现一：VLLMAdapter（本地 vLLM）

### 3.1 类定义

```python
class VLLMAdapter(LLMAdapter):
    """本地 vLLM 推理服务适配器"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self._context_length = config.get("context_length", 8192)
        self._client = None
```

### 3.2 初始化客户端

使用 OpenAI Python SDK（vLLM 兼容 OpenAI API 格式）：
```python
from openai import OpenAI

self._client = OpenAI(
    base_url=self.base_url,
    api_key="EMPTY",  # vLLM 不需要 API Key
)
```

### 3.3 chat

**功能**：非流式对话。

**调用方式**：
```python
response = self._client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    stream=False,
)
```

**返回转换**：
```python
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

### 3.4 chat_stream

**功能**：流式对话。

**调用方式**：
```python
stream = self._client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    stream=True,
)
```

**返回转换**：
```python
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    finish_reason = chunk.choices[0].finish_reason
    usage = chunk.usage  # 流式响应最后一个 chunk 有 usage
    yield LLMChunk(
        delta_content=delta,
        finish_reason=finish_reason,
        usage=TokenUsage(...) if usage else None,
    )
```

### 3.5 count_tokens

**功能**：统计 token 数（使用 tiktoken 近似）。

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
return len(enc.encode(text))
```

### 3.6 rebuild_client

**功能**：销毁旧的 OpenAI 客户端，重新创建新客户端（用于断线重连）。

```python
def rebuild_client(self) -> None:
    del self._client  # 销毁旧连接
    self._client = OpenAI(
        base_url=self.base_url,
        api_key="EMPTY",
    )
```

---

## 四、当前实现二：DeepSeekAdapter（云端 DeepSeek API）

### 4.1 类定义

```python
class DeepSeekAdapter(LLMAdapter):
    """云端 DeepSeek API 适配器"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.api_key = config["api_key"]
        self._context_length = config.get("context_length", 131072)
        self._client = None
```

### 4.2 初始化客户端

```python
from openai import OpenAI

self._client = OpenAI(
    base_url=self.base_url,
    api_key=self.api_key,
)
```

### 4.3 chat / chat_stream

与 VLLMAdapter 完全相同（都使用 OpenAI SDK），区别仅在于 base_url 和 api_key。

### 4.4 count_tokens

DeepSeek 使用与 GPT 相同的 tokenizer，使用 tiktoken cl100k_base。

### 4.5 rebuild_client

与 VLLMAdapter 相同，销毁旧客户端并重新创建：

```python
def rebuild_client(self) -> None:
    del self._client
    self._client = OpenAI(
        base_url=self.base_url,
        api_key=self.api_key,
    )
```

---

## 五、可扩展实现

| 适配器 | 服务 | 上下文 | 说明 |
|---|---|---|---|
| VLLMAdapter | 本地 vLLM | 8K | 当前实现（本地免费推理） |
| DeepSeekAdapter | DeepSeek API | 128K | 当前实现（云端长上下文） |
| OpenAIAdapter | OpenAI API | 128K | GPT-4o，质量最高，需代理 |
| ClaudeAdapter | Anthropic API | 200K | Claude 3 Opus，长上下文最强 |
| QwenCloudAdapter | 通义千问 API | 32K+ | 阿里云端，国内访问快 |
| ZhipuAdapter | 智谱 GLM API | 128K | 智谱云端，国内 |

---

## 六、适配器工厂

### 6.1 create_adapter

**功能**：根据配置创建对应的 LLM 适配器。

```python
def create_adapter(backend: str, config: dict) -> LLMAdapter:
    """
    创建 LLM 适配器

    Args:
        backend: 后端类型（"vllm"/"deepseek"/"openai"/...）
        config: 后端配置

    Returns:
        LLMAdapter 实例
    """
    adapters = {
        "vllm": VLLMAdapter,
        "deepseek": DeepSeekAdapter,
        # 后续新增适配器在此注册
    }
    if backend not in adapters:
        raise ValueError(f"不支持的后端: {backend}")
    return adapters[backend](config)
```

---

## 七、依赖关系

- **被依赖**：M11 LLM 客户端模块（通过 LLMAdapter 接口调用）
- **依赖**：无（隔离第三方库）
- **第三方依赖**：openai（OpenAI Python SDK，vLLM 和 DeepSeek 都兼容）、tiktoken（token 计数）
