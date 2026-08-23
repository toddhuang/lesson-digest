# M11 LLM 客户端模块 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M11
> 模块职责：LLM 调用业务逻辑（双后端管理、重试、缓存、流式），通过适配层调用具体 LLM 服务

---

## 一、模块概述

LLM 客户端模块负责统一管理大模型调用。包含双后端管理（本地 vLLM + 云端 DeepSeek）、重试机制、结果缓存、流式响应处理等业务逻辑。**不直接依赖 vLLM 或 DeepSeek 的 API**，而是通过 M17 LLM 适配层调用具体 LLM 服务。

---

## 二、接口列表

### 11.1 chat

**功能**：非流式对话，返回完整响应。

**签名**：
```python
def chat(
    messages: List[dict],
    backend: str = "auto",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2000,
    use_cache: bool = True
) -> LLMResponse
```

**输入参数**：
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| messages | List[dict] | 是 | — | 对话消息列表，格式：[{"role": "system"/"user"/"assistant", "content": "..."}] |
| backend | str | 否 | "auto" | 后端选择："auto"（自动）/"local"（本地）/"cloud"（云端） |
| temperature | float | 否 | 0.3 | 温度参数（0-2） |
| top_p | float | 否 | 0.9 | top_p 参数（0-1） |
| max_tokens | int | 否 | 2000 | 最大生成 token 数 |
| use_cache | bool | 否 | True | 是否使用缓存 |

**输出**：LLMResponse（响应对象）

**异常**：
| 异常类型 | 触发条件 |
|---|---|
| LLMError | LLM 调用失败（含重试耗尽） |
| LLMRateLimitError | 速率限制 |
| LLMTimeoutError | 超时 |
| InvalidBackendError | 无效的后端选择 |

---

### 11.2 chat_stream

**功能**：流式对话，返回响应块迭代器。

**签名**：
```python
def chat_stream(
    messages: List[dict],
    backend: str = "auto",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2000
) -> Iterator[LLMChunk]
```

**输入参数**：同 chat（无 use_cache，流式不缓存）

**输出**：Iterator[LLMChunk]（响应块迭代器）

**异常**：同 chat

---

### 11.3 select_backend

**功能**：根据任务类型和输入长度自动选择后端。

**签名**：
```python
def select_backend(
    messages: List[dict],
    task_type: str = "general"
) -> str
```

**输入参数**：
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| messages | List[dict] | 是 | — | 对话消息列表（用于估算 token 数） |
| task_type | str | 否 | "general" | 任务类型："general"/"knowledge_extraction"/"problem_extraction"/"mindmap" |

**输出**：str（"local" 或 "cloud"）

**选择规则**：
| 条件 | 后端 |
|---|---|
| 任务类型为 knowledge_extraction/problem_extraction/mindmap | cloud |
| 输入 token > 6000（接近本地 8K 上限） | cloud |
| 其他 | local |

---

### 11.4 count_tokens

**功能**：统计文本的 token 数。

**签名**：
```python
def count_tokens(text: str, backend: str = "local") -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| text | str | 是 | — | 待统计文本 |
| backend | str | 否 | "local" | 后端（不同模型 tokenizer 不同） |

**输出**：int（token 数）

---

### 11.5 get_context_length

**功能**：获取指定后端的上下文长度。

**签名**：
```python
def get_context_length(backend: str) -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| backend | str | 是 | 后端："local"/"cloud" |

**输出**：int（上下文长度，local=8192，cloud=131072）

---

## 三、数据结构

### 3.1 LLMResponse（dataclass）

```python
@dataclass
class LLMResponse:
    content: str           # 生成的文本内容
    model: str             # 使用的模型名
    usage: TokenUsage      # token 使用统计
    finish_reason: str     # 结束原因："stop"/"length"/"content_filter"
    backend: str           # 使用的后端："local"/"cloud"
    latency: float         # 响应延迟（秒）
```

### 3.2 LLMChunk（dataclass）

```python
@dataclass
class LLMChunk:
    delta_content: str     # 增量文本内容
    finish_reason: str     # 结束原因（最后一个 chunk 有值）
    usage: TokenUsage     # token 使用统计（最后一个 chunk 有值）
```

### 3.3 TokenUsage（dataclass）

```python
@dataclass
class TokenUsage:
    prompt_tokens: int     # 输入 token 数
    completion_tokens: int # 输出 token 数
    total_tokens: int      # 总 token 数
```

---

## 四、重试机制

### 4.1 可重试异常

| 异常 | 重试 | 退避策略 |
|---|---|---|
| 网络超时 / 连接重置 | ✅ | 指数退避 5s→10s→20s |
| HTTP 5xx（服务器错误） | ✅ | 指数退避 |
| 速率限制（429） | ✅ | 指数退避 + 读取 Retry-After 头 |
| HTTP 4xx（客户端错误） | ❌ | 直接抛出 |
| 内容过滤 | ❌ | 直接抛出 |

### 4.2 最大重试次数

默认 5 次（MVP 质量优先，失败则中断而非跳过）。

### 4.3 重试日志

每次重试记录日志：重试次数、异常类型、等待时间、最终结果。

---

## 五、缓存机制

### 5.1 缓存键

```
cache_key = md5(messages_json + backend + model + temperature + top_p + max_tokens)
```

### 5.2 缓存格式

JSON 文件，存储于 `cache/llm/{cache_key}.json`：
```json
{
  "response": {...LLMResponse...},
  "created_at": "2026-08-23T10:00:00"
}
```

### 5.3 缓存有效期

默认永久有效（相同输入应返回相同或相似输出）。可配置 `cache_ttl` 设置有效期。

---

## 六、第三方库调用（通过适配层）

本模块**不直接调用** vLLM 或 DeepSeek API，而是通过 M17 LLM 适配层的统一接口调用：

```python
from adapters.llm_adapter import LLMAdapter, VLLMAdapter, DeepSeekAdapter

local_adapter: LLMAdapter = VLLMAdapter(config.local)
cloud_adapter: LLMAdapter = DeepSeekAdapter(config.cloud)

response = local_adapter.chat(messages, **kwargs)
```

适配层统一接口详见：`M17_LLM适配层接口.md`

---

## 七、依赖关系

- **被依赖**：M7 知识点提取模块、M8 题目提取模块、M10 思维导图生成模块
- **依赖**：M17 LLM 适配层（隔离 vLLM/DeepSeek）、M1 配置管理、M14 工具集（token 计数）
- **第三方依赖**：通过适配层间接依赖 vLLM、DeepSeek API
