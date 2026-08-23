# M11 LLM 客户端模块 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M11
> 模块职责：LLM 调用业务逻辑（双后端管理、重试、缓存、健康检查、断线重连），通过适配层调用具体 LLM 服务

---

## 一、模块概述

LLM 客户端模块负责统一管理大模型调用。包含双后端管理（本地 vLLM + 云端 DeepSeek）、重试机制、结果缓存、健康检查、断线重连等业务逻辑。**不直接依赖 vLLM 或 DeepSeek 的 API**，而是通过 M17 LLM 适配层调用具体 LLM 服务。

> **MVP 设计原则**：质量优先，不考虑成本。三个核心 LLM 任务（知识点提取、题目提取、思维导图生成）全部使用云端 DeepSeek（128K 上下文，强模型智力），不做成本优化（如合并调用、减少输入）。本地 vLLM 保留用于短文本任务和云端故障时的手动降级。

---

## 二、接口列表

### 2.1 chat

**功能**：非流式对话，返回完整响应。

**签名**：
```python
def chat(
    messages: List[dict],
    backend: str,
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
| backend | str | 是 | — | 后端选择："local"（本地 vLLM）/ "cloud"（云端 DeepSeek），**不支持 auto** |
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
| InvalidBackendError | 无效的后端选择（非 "local"/"cloud"） |
| LLMConnectionError | 连接失败（健康检查未通过或重连失败） |

---

### 2.2 chat_stream

**功能**：流式对话，返回响应块迭代器。

**签名**：
```python
def chat_stream(
    messages: List[dict],
    backend: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2000
) -> Iterator[LLMChunk]
```

**输入参数**：同 chat（无 use_cache，流式不缓存）

**输出**：Iterator[LLMChunk]（响应块迭代器）

**异常**：同 chat

---

### 2.3 health_check

**功能**：检查指定后端的连接健康状态。

**签名**：
```python
def health_check(self, backend: str) -> bool
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| backend | str | 是 | 后端："local"/"cloud" |

**输出**：bool（健康返回 True，不可达返回 False）

**实现方式**：发送一个极简的 LLM 请求（如 `messages=[{"role":"user","content":"hi"}]`, `max_tokens=1`），3秒内返回则视为健康。

---

### 2.4 reconnect

**功能**：重新建立与指定后端的连接（适配层客户端重建）。

**签名**：
```python
def reconnect(self, backend: str) -> bool
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| backend | str | 是 | 后端："local"/"cloud" |

**输出**：bool（重连成功返回 True）

---

### 2.5 count_tokens

**功能**：统计文本的 token 数。

**签名**：
```python
def count_tokens(self, text: str, backend: str = "local") -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| text | str | 是 | — | 待统计文本 |
| backend | str | 否 | "local" | 后端（不同模型 tokenizer 不同） |

**输出**：int（token 数）

---

### 2.6 get_context_length

**功能**：获取指定后端的上下文长度。

**签名**：
```python
def get_context_length(self, backend: str) -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| backend | str | 是 | 后端："local"/"cloud" |

**输出**：int（上下文长度，local=8192，cloud=131072）

---

## 三、数据结构

> 共享数据结构（LLMResponse、LLMChunk、TokenUsage）统一定义见 `00_数据模型.md`。

---

## 四、第三方库调用（通过适配层）

本模块**不直接调用** vLLM 或 DeepSeek API，而是通过 M17 LLM 适配层的统一接口调用：

```python
from adapters.llm_adapter import LLMAdapter, VLLMAdapter, DeepSeekAdapter

local_adapter: LLMAdapter = VLLMAdapter(config.local)
cloud_adapter: LLMAdapter = DeepSeekAdapter(config.cloud)

response = local_adapter.chat(messages, temperature, top_p, max_tokens)
```

适配层统一接口详见：`M17_LLM适配层接口.md`

---

## 五、后端使用策略

| 任务 | 后端 | 理由 |
|---|---|---|
| 知识点提取 | cloud | 全文本一次性语义判断，需要长上下文+强模型智力 |
| 题目提取 | cloud | 全文本一次性语义判断，需要理解各种讲题过渡语 |
| 思维导图生成 | cloud | 基于知识点列表+摘要生成结构，云端模型结构能力更强 |
| 短文本润色/格式化 | local | 输入短，本地免费，减少云端调用 |
| 健康检查 | local/cloud | 极简请求，快速检测连通性 |

> **MVP 阶段不做自动后端选择**：所有调用方明确指定 backend，避免自动选择逻辑带来的不确定性。云端 API 成本极低（DeepSeek 约 ¥0.001/千 token 输入），MVP 阶段优先保证质量。

---

## 六、依赖关系

- **被依赖**：M7 知识点提取模块、M8 题目提取模块、M10 思维导图生成模块
- **依赖**：M17 LLM 适配层（隔离 vLLM/DeepSeek）、M1 配置管理、M14 工具集（token 计数）
- **第三方依赖**：通过适配层间接依赖 vLLM、DeepSeek API
