# M17 LLM 适配层 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M17
> 模块职责：定义统一 LLM 接口，封装具体 LLM 服务，隔离第三方库变化

---

## 一、模块概述

LLM 适配层采用**适配器模式（Adapter Pattern）**，定义统一的 LLM 调用接口，封装具体 LLM 服务的 API 差异。M11-M17 重构后当前实现为 **LiteLLMAdapter**（通过 LiteLLM 统一调用所有 OpenAI 兼容服务商，如豆包、DeepSeek）和 **MockLLMAdapter**（测试用假数据）。将来新增 LLM 服务时只需在配置注册表中添加模型+服务商，无需新增适配器类。

> **重构变化**：旧版 `VolcengineAdapter` + `DeepSeekAdapter` 两个独立类合并为 `LiteLLMAdapter` 统一处理；旧接口 `chat` / `chat_stream` / `get_context_length` / `count_tokens` / `get_model_name` 全部废弃，统一为 `generate(prompt, payload, temperature)`。

---

## 二、统一接口定义

### 2.1 LLMAdapter（抽象基类）

```python
from abc import ABC, abstractmethod
from config import ModelConfig, ProviderConfig
from utils.models import LLMResponse


class LLMAdapter(ABC):
    """LLM 适配器抽象基类

    每个适配器实例绑定一个具体模型（ModelConfig）和服务商（ProviderConfig）。
    适配器负责将 prompt + payload 转换为底层 API 调用，
    内部处理流式接收、分块、异常映射等细节。
    """

    @abstractmethod
    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        pass

    @abstractmethod
    def generate(self, prompt: str, payload: str, temperature: float) -> LLMResponse:
        """生成 LLM 响应

        Args:
            prompt: 系统提示词（任务指令）
            payload: 待处理内容（数据）
            temperature: 温度参数（由 LLMSession 从任务配置传入）

        Returns:
            LLMResponse 对象

        Raises:
            LLMClientError: 参数错误或认证失败（不重试）
            LLMRateLimitError: 速率限制（可重试）
            LLMTimeoutError: 超时（可重试）
            LLMConnectionError: 连接失败（可重试）
            LLMServerError: 服务端错误（可重试）
            LLMContextOverflowError: 输入超过上下文限制
        """
        pass
```

> 共享数据结构 `LLMResponse` / `TokenUsage` / `LLMChunk` 统一定义见 `00_数据模型.md` 第四节。

---

## 三、当前实现一：LiteLLMAdapter（统一处理多服务商，默认推荐）

### 3.1 类定义

```python
class LiteLLMAdapter(LLMAdapter):
    """基于 LiteLLM 的 LLM 适配器，统一调用所有 OpenAI 兼容服务商"""

    def __init__(self, model_config: ModelConfig, provider_config: ProviderConfig,
                 max_retries: int = 3, timeout: int = 120):
        self.model_config = model_config
        self.provider_config = provider_config
        self.max_retries = max_retries
        self.timeout = timeout
        self._splitter = RecursiveTextSplitter()
```

### 3.2 generate（自动分块 + 流式接收）

**功能**：生成 LLM 响应，自动处理 payload 超限分块。

**流程**：
1. `count_tokens(prompt)` 计算 prompt token 数
2. 计算可用空间 = `context_length * 0.9 - prompt_tokens - max_output`
3. `count_tokens(payload)` 计算 payload token 数
4. payload 未超限 → 单次调用 `_call_single`
5. payload 超限 → `RecursiveTextSplitter.split()` 分块，逐块调用并拼接 content、累加 usage

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| prompt | str | 是 | 系统提示词（任务指令） |
| payload | str | 是 | 待处理内容 |
| temperature | float | 是 | 温度参数（由 LLMSession 传入） |

**输出**：LLMResponse（分块调用时 content 为各块 `"\n".join`，usage 为累加值）

**异常**：见 3.3 异常映射表

### 3.3 _call_single（单次 API 调用 + 异常映射）

使用 `litellm.completion(stream=True)` 流式接收，model 名拼为 `{litellm_prefix}/{model_name}`（如 `openai/doubao-seed-2-1-pro-260628`）。

```python
stream = litellm.completion(
    model=f"{provider_config.litellm_prefix}/{model_config.name}",
    api_base=provider_config.base_url,
    api_key=provider_config.api_key,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ],
    temperature=temperature,
    max_tokens=model_config.max_output,
    stream=True,
    stream_options={"include_usage": True},
    num_retries=max_retries,
    timeout=timeout,
)
```

**异常映射表**（按 AGENTS.md 规范分类处理，禁止裸 `except Exception`）：

| LiteLLM 异常 | 项目异常 | 处理策略 |
|---|---|---|
| `BadRequestError` (400) | `LLMClientError` | 参数错误，不重试 |
| `AuthenticationError` (401) | `LLMClientError` | 认证失败，不重试 |
| `RateLimitError` (429) | `LLMRateLimitError` | 限流，指数退避重试 |
| `InternalServerError` (5xx) | `LLMServerError` | 服务端错误，延迟重试 |
| `Timeout` | `LLMTimeoutError` | 超时，改流式重试 |
| `APIConnectionError` | `LLMConnectionError` | 连接失败，重试 |
| `APIError` (status≥500) | `LLMServerError` | 服务端错误，重试 |
| `APIError` (其他) | `LLMError` | 兜底 |

### 3.4 _collect_stream（流式响应收集）

遍历流式 chunk，收集 `delta.content` 拼接为完整 content，记录 `finish_reason`，从最后一个含 `usage` 的 chunk 提取 token 统计。

---

## 四、当前实现二：MockLLMAdapter（测试用假数据）

### 4.1 类定义

```python
class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器，根据 prompt 内容返回假数据，用于链路测试"""

    def __init__(self, model_config: ModelConfig = None, provider_config: ProviderConfig = None,
                 max_retries: int = 3, timeout: int = 120):
        if model_config is None:
            model_config = ModelConfig(name="mock-model", provider="mock", ...)
        self.model_config = model_config
```

### 4.2 generate

根据 prompt 关键词返回不同假数据：
- 含 `corrected_text` / `一次完成`：返回合并调用 JSON `{corrected_text, knowledge_segments, problem_segments}`
- 含 `思维导图` / `OPML`：返回 OPML XML
- 含 `知识点`：返回知识点 JSON 数组
- 含 `题目` / `习题`：返回题目 JSON 数组
- 其他：返回纯文本假数据

不依赖真实 API，用于 Pipeline 链路测试。

---

## 五、可扩展实现

| 适配器 | 服务 | 说明 |
|---|---|---|
| LiteLLMAdapter | 豆包 / DeepSeek 等 OpenAI 兼容服务商 | **当前默认**，通过 LiteLLM 统一调用，配置注册表添加模型即可 |
| MockLLMAdapter | 假数据 | **当前实现**，单元测试用 |
| OpenAIAdapter | OpenAI API | GPT-4o，LiteLLM 已支持，配置即可启用 |
| ClaudeAdapter | Anthropic API | Claude 3 Opus，LiteLLM 已支持 |
| QwenCloudAdapter | 通义千问 API | 阿里云端，LiteLLM 已支持 |
| ZhipuAdapter | 智谱 GLM API | 智谱云端，LiteLLM 已支持 |

> LiteLLM 已覆盖主流服务商，新增服务商只需在 `config.yaml` 的 `llm.models[]` 和 `llm.providers` 注册，无需新增适配器类。

---

## 六、适配器工厂

### 6.1 create_llm_adapter

**功能**：根据模型和服务商配置创建 LLM 适配器实例。

```python
def create_llm_adapter(
    model_config: ModelConfig,
    provider_config: ProviderConfig,
    max_retries: int = 3,
    timeout: int = 120,
    mock: bool = False,
) -> LLMAdapter:
    if mock:
        return MockLLMAdapter(model_config, provider_config, max_retries, timeout)
    return LiteLLMAdapter(model_config, provider_config, max_retries, timeout)
```

适配器实例由 `LLMClient._get_adapter(model_name)` 懒加载并按模型名缓存，同一模型复用一个适配器。

---

## 七、依赖关系

- **被依赖**：M11 LLM 客户端模块（通过 LLMAdapter 接口调用，由 LLMSession 桥接 2 参数 `generate(prompt, payload)` 到 3 参数 `generate(prompt, payload, temperature)`）
- **依赖**：`utils.token_counter`（token 计数）、`utils.exceptions`（异常映射）、`core.llm.text_splitter`（超限分块）
- **第三方依赖**：litellm（统一 LLM 调用）、tiktoken（token 计数）
