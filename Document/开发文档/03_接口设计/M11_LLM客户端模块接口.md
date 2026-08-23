# M11 LLM 客户端模块 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M11
> 模块职责：LLM 调用业务逻辑（多服务商管理、重试、缓存），通过适配层调用具体 LLM 服务

---

## 一、模块概述

LLM 客户端模块负责统一管理大模型调用。包含多服务商管理（默认豆包，备选 DeepSeek）、重试机制、结果缓存等业务逻辑。**不直接依赖豆包或 DeepSeek 的 API**，而是通过 M17 LLM 适配层调用具体 LLM 服务。

> **MVP 设计原则**：质量优先，不考虑成本。三个核心 LLM 任务（知识点提取、题目提取、思维导图生成）+ ASR 纠错全部使用云端 LLM（默认豆包），不做成本优化（如合并调用、减少输入）。

> **放弃本地 4090 的原因**：4090 工作站开一天的电费 + 硬件折旧 > 云端 API token 费用，且本地 27B 模型推理质量不如云端。

---

## 二、接口列表

### 2.1 chat

**功能**：非流式对话，返回完整响应。

**签名**：
```python
def chat(
    messages: List[dict],
    backend: str = "cloud",
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
| backend | str | 否 | "cloud" | 后端选择："cloud"（映射到 default_provider，默认豆包）/ "mock"（测试用假数据） |
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

### 2.2 chat_stream

**功能**：流式对话，返回响应块迭代器。

**签名**：
```python
def chat_stream(
    messages: List[dict],
    backend: str = "cloud",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2000
) -> Iterator[LLMChunk]
```

**输入参数**：同 chat（无 use_cache，流式不缓存）

**输出**：Iterator[LLMChunk]（响应块迭代器）

**异常**：同 chat

---

### 2.3 count_tokens

**功能**：统计文本的 token 数。

**签名**：
```python
def count_tokens(self, text: str, backend: str = "cloud") -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| text | str | 是 | — | 待统计文本 |
| backend | str | 否 | "cloud" | 后端（不同模型 tokenizer 不同） |

**输出**：int（token 数）

---

### 2.4 get_context_length

**功能**：获取指定后端的上下文长度。

**签名**：
```python
def get_context_length(self, backend: str = "cloud") -> int
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| backend | str | 否 | 后端："cloud" |

**输出**：int（上下文长度，云端模型 128K+）

---

## 三、数据结构

> 共享数据结构（LLMResponse、LLMChunk、TokenUsage）统一定义见 `00_数据模型.md`。

---

## 四、第三方库调用（通过适配层）

本模块**不直接调用**豆包或 DeepSeek API，而是通过 M17 LLM 适配层的统一接口调用：

```python
from adapters.llm_adapter import create_adapter

# 根据 default_provider 创建适配器（volcengine=豆包, deepseek=DeepSeek）
adapter = create_adapter(provider_name, config)

response = adapter.chat(messages, temperature, top_p, max_tokens)
```

适配层统一接口详见：`M17_LLM适配层接口.md`

---

## 五、后端使用策略

| 任务 | 后端 | 理由 |
|---|---|---|
| 知识点提取 | cloud | 全文本一次性语义判断，需要长上下文+强模型智力 |
| 题目提取 | cloud | 全文本一次性语义判断，需要理解各种讲题过渡语 |
| 思维导图生成 | cloud | 基于知识点列表+摘要生成结构，云端模型结构能力更强 |
| ASR 纠错 | cloud（豆包） | 豆包对中文同音词纠错效果优于 DeepSeek |
| 单元测试 | mock | 返回假数据，不依赖真实 API |

> **MVP 阶段不做自动后端选择**：所有调用方明确指定 backend="cloud"，映射到配置中的 default_provider。云端 API 成本极低，MVP 阶段优先保证质量。

---

## 六、多服务商配置

### 6.1 配置结构

```yaml
llm:
  default_provider: volcengine  # volcengine=豆包, deepseek=DeepSeek
  providers:
    volcengine:
      base_url: https://ark.cn-beijing.volces.com/api/v3
      model: doubao-seed-2-1-pro-260628
      api_key: ${VOLCENGINE_API_KEY}
    deepseek:
      base_url: https://api.deepseek.com
      model: deepseek-v4-flash
      api_key: ${DEEPSEEK_API_KEY}
```

### 6.2 切换默认服务商

修改 `config.yaml` 中的 `default_provider` 即可，无需修改代码。

---

## 七、依赖关系

- **被依赖**：M7 知识点提取模块、M8 题目提取模块、M10 思维导图生成模块、M12 输出组装模块（ASR 纠错）
- **依赖**：M17 LLM 适配层（隔离豆包/DeepSeek）、M1 配置管理、M14 工具集（token 计数）
- **第三方依赖**：通过适配层间接依赖豆包 API、DeepSeek API
