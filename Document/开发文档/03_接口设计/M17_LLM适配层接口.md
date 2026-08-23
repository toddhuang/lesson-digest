# M17 LLM 适配层 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M17
> 模块职责：定义统一 LLM 接口，封装具体 LLM 服务，隔离第三方库变化

---

## 一、模块概述

LLM 适配层采用**适配器模式（Adapter Pattern）**，定义统一的 LLM 调用接口，封装具体 LLM 服务的 API 差异。当前实现为 VolcengineAdapter（火山引擎豆包 API）、DeepSeekAdapter（云端 DeepSeek API）和 MockAdapter（测试用假数据），将来新增 LLM 服务（如 OpenAI、Claude、通义千问、智谱GLM）时只需新增适配器，核心业务代码（M11 LLM 客户端模块）零修改。

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
```

---

## 三、当前实现一：VolcengineAdapter（火山引擎豆包 API，默认推荐）

### 3.1 类定义

```python
class VolcengineAdapter(LLMAdapter):
    """火山引擎豆包 API 适配器（默认推荐，中文同音词纠错效果好）"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.api_key = config["api_key"]
        self._context_length = config.get("context_length", 131072)
        self._client = None
```

### 3.2 初始化客户端

使用 OpenAI Python SDK（豆包兼容 OpenAI API 格式，但使用 `responses.create` 接口）：

```python
from openai import OpenAI

self._client = OpenAI(
    base_url=self.base_url,
    api_key=self.api_key,
)
```

### 3.3 chat（重要：使用 responses.create 接口）

**功能**：非流式对话。

> **重要**：豆包 seed 系列模型**不支持** `chat.completions.create` 接口（会返回 404），必须使用 `responses.create` 接口。且必须禁用思考过程 `reasoning={"effort": "none"}`，否则 token 会被思考过程占满，status=incomplete，output_text 为空。

**调用方式**：
```python
response = self._client.responses.create(
    model=self.model,
    input=messages,
    temperature=temperature,
    top_p=top_p,
    max_output_tokens=max_tokens,
    reasoning={"effort": "none"},  # 必须禁用思考过程
)
```

**返回转换**：
```python
return LLMResponse(
    content=response.output_text,
    model=response.model,
    usage=TokenUsage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
    ),
    finish_reason="stop" if response.status == "completed" else response.status,
)
```

### 3.4 chat_stream

**功能**：流式对话。

使用 `responses.create` 接口的流式模式：
```python
stream = self._client.responses.create(
    model=self.model,
    input=messages,
    temperature=temperature,
    top_p=top_p,
    max_output_tokens=max_tokens,
    reasoning={"effort": "none"},
    stream=True,
)
```

### 3.5 count_tokens

豆包使用与 GPT 相同的 tokenizer，使用 tiktoken cl100k_base。

---

## 四、当前实现二：DeepSeekAdapter（云端 DeepSeek API，备选）

### 4.1 类定义

```python
class DeepSeekAdapter(LLMAdapter):
    """云端 DeepSeek API 适配器（备选服务商）"""

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

### 4.3 chat（重要：禁用思考过程）

**功能**：非流式对话。

> **重要**：DeepSeek 推理模型必须禁用思考过程 `extra_body={"thinking": {"type": "disabled"}}`，否则 token 会被思考过程占满。

**调用方式**：
```python
response = self._client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    stream=False,
    extra_body={"thinking": {"type": "disabled"}},  # 必须禁用思考过程
)
```

### 4.4 chat_stream

与 chat 类似，设置 `stream=True`，同样需要禁用思考过程。

### 4.5 count_tokens

DeepSeek 使用与 GPT 相同的 tokenizer，使用 tiktoken cl100k_base。

---

## 五、当前实现三：MockAdapter（测试用假数据）

### 5.1 类定义

```python
class MockAdapter(LLMAdapter):
    """Mock 适配器，返回假数据，用于单元测试，不依赖真实 API"""

    def __init__(self, config: dict = None):
        self.model = "mock-model"
        self._context_length = 8192
```

### 5.2 chat

返回预设的假数据，根据 messages 中的关键词返回不同的模拟响应。

---

## 六、可扩展实现

| 适配器 | 服务 | 上下文 | 说明 |
|---|---|---|---|
| VolcengineAdapter | 火山引擎豆包 API | 128K+ | **当前默认**，中文同音词纠错效果好 |
| DeepSeekAdapter | DeepSeek API | 128K | 当前备选，长上下文 |
| MockAdapter | 假数据 | 8K | 单元测试用 |
| OpenAIAdapter | OpenAI API | 128K | GPT-4o，质量最高，需代理 |
| ClaudeAdapter | Anthropic API | 200K | Claude 3 Opus，长上下文最强 |
| QwenCloudAdapter | 通义千问 API | 32K+ | 阿里云端，国内访问快 |
| ZhipuAdapter | 智谱 GLM API | 128K | 智谱云端，国内 |

---

## 七、适配器工厂

### 7.1 create_adapter

**功能**：根据服务商名称创建对应的 LLM 适配器。

```python
def create_adapter(provider: str, config: dict) -> LLMAdapter:
    """
    创建 LLM 适配器

    Args:
        provider: 服务商名称（"volcengine"/"deepseek"/"mock"/...）
        config: 服务商配置

    Returns:
        LLMAdapter 实例
    """
    adapters = {
        "volcengine": VolcengineAdapter,
        "deepseek": DeepSeekAdapter,
        "mock": MockAdapter,
        # 后续新增适配器在此注册
    }
    if provider not in adapters:
        raise ValueError(f"不支持的服务商: {provider}")
    return adapters[provider](config)
```

---

## 八、依赖关系

- **被依赖**：M11 LLM 客户端模块（通过 LLMAdapter 接口调用）
- **依赖**：无（隔离第三方库）
- **第三方依赖**：openai（OpenAI Python SDK，豆包和 DeepSeek 都兼容）、tiktoken（token 计数）
