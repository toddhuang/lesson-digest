# R-005 LLM 应用框架选型调研

> 调研目标：调研国内外主流 LLM 应用框架/库，逐一对照本项目需求，给出具体匹配/不匹配的判断依据。
> 调研时间：2026-08-26
> 信息来源：各框架官网文档和 GitHub 官方仓库

---

## 一、本项目需求清单

LLM 层需要解决的 5 个核心问题：

| # | 需求 | 具体说明 |
|---|---|---|
| D1 | 统一多模型调用 | 豆包（火山引擎，OpenAI 兼容接口）、DeepSeek（OpenAI 兼容接口），未来可能加模型；切换模型不改业务代码 |
| D2 | 长文本分块处理 | prompt + payload 超 context_length 时，对 payload 分块、逐块调用、拼接结果文本返回。分块策略参考 R-004（递归分隔符、token 估算） |
| D3 | 生产级调用能力 | 重试（限流指数退避、服务端错误延迟重试）、超时处理、流式接收并拼接完整结果、错误分类（400 不重试、401 不重试、429 退避、5xx 重试） |
| D4 | 输出解析 | JSON 解析（知识点/题目提取）、OPML/XML 解析（思维导图）、纯文本（纠错）；解析失败能拿到错误信息 |
| D5 | 轻量 | 不需要 Agent、工具调用、RAG、向量数据库、可视化工作流、Memory 等功能；不引入不必要的重型依赖 |

---

## 二、国外框架详细分析

### 1. LiteLLM

| 项目 | 信息 |
|---|---|
| 官网 | https://docs.litellm.ai |
| GitHub | https://github.com/BerriAI/litellm |
| Stars | 57,276 |
| 语言 | Python（Rust 核心） |
| 定位 | 统一 LLM API 网关 / SDK |

**核心思路**：一个 `completion()` 函数用 OpenAI 格式调用 100+ LLM 提供商。通过 `model="provider/model"` 指定模型，通过 `api_base` 指定自定义端点。所有提供商的异常统一映射为 OpenAI 异常类型。另有 Router 类提供多部署负载均衡、fallback、冷却。

**逐需求对照：**

**D1 统一多模型调用：✅ 完全匹配**

豆包和 DeepSeek 都是 OpenAI 兼容接口，LiteLLM 原生支持 `openai/` provider 传自定义 `api_base`：

```python
from litellm import completion

# 豆包
response = completion(
    model="openai/doubao-seed-2-1-pro-260628",
    api_base="https://ark.cn-beijing.volces.com/api/v3",
    api_key="xxx",
    messages=[{"role": "user", "content": "..."}]
)

# DeepSeek
response = completion(
    model="deepseek/deepseek-chat",
    api_key="xxx",
    messages=[{"role": "user", "content": "..."}]
)
```

切换模型只改 model 名和 api_base，业务代码不变。

**D2 长文本分块：❌ 不提供**

LiteLLM 只做单次 API 调用，不提供文本分块功能。但它定义了专门的 `ContextWindowExceededError` 异常（继承自 BadRequestError），可以捕获后自行分块重试。分块逻辑需要自己实现（参考 R-004 的递归分隔符方案）。

**D3 生产级调用：✅ 匹配**

- **重试**：内置 `num_retries` 参数，429/5xx 自动重试，支持指数退避（`litellm.max_retries = 3`）
- **超时**：支持 `timeout` 参数，超时抛 `openai.APITimeoutError`
- **流式**：`stream=True` 返回迭代器，逐 chunk 读取，拼接需自己做
- **错误分类**：所有 provider 的异常统一映射为 OpenAI 异常类型（BadRequestError 400、AuthenticationError 401、RateLimitError 429、APITimeoutError 408、InternalServerError 5xx），还有 `_should_retry(status_code)` 方法判断是否该重试
- **Router**：支持 fallback 模型链、负载均衡（simple-shuffle/latency-based/cost-based）、失败冷却（allowed_fails/cooldown_time）、限流感知

**D4 输出解析：❌ 不提供**

返回 OpenAI 格式的 response 对象，`response.choices[0].message.content` 是原始文本。JSON/XML 解析需要自己用标准库做。

**D5 轻量：✅ 完全匹配**

核心就是一个 `completion()` 函数 + Router 类，没有 Agent/RAG/向量库等包袱。Rust 核心性能好。

**总结**：D1✅ D2❌ D3✅ D4❌ D5✅。精确解决多模型统一调用和生产级调用，分块和输出解析需要自己实现。

---

### 2. LangChain

| 项目 | 信息 |
|---|---|
| 官网 | https://python.langchain.com |
| GitHub | https://github.com/langchain-ai/langchain |
| Stars | 145,001 |
| 语言 | Python / JS |
| 定位 | Agent 工程平台，全能型框架 |

**核心思路**：提供 LLM 应用开发全套组件——ChatModel（统一模型接口）、PromptTemplate、OutputParser、TextSplitter、Chain/LCEL（链式编排）、Agent、Memory、Callback、Cache。组件可组合。

**逐需求对照：**

**D1 统一多模型调用：✅ 匹配**

`ChatOpenAI` 支持传 `base_url` 对接 OpenAI 兼容接口，豆包/DeepSeek 都可以：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="doubao-seed-2-1-pro-260628",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="xxx"
)
```

`init_chat_model("provider:model")` 支持一行初始化任意 provider。

**D2 长文本分块：✅ 分块器匹配，但 Map-Reduce 链不匹配**

- `RecursiveCharacterTextSplitter` 是业界最成熟的分块器：递归按分隔符优先级（段落→行→句子→字符）切分，支持 chunk_size/chunk_overlap/length_function（可传 token 计数器），这正是 R-004 建议参考的实现
- 但 LangChain 官方的 Map-Reduce chain（MapReduceDocumentsChain）的 reduce 阶段是**再调一次 LLM 合并摘要**，不是代码拼接文本，和我们的需求不符
- 可以用 splitter 分块 + 自己写循环逐块调用 + 自己拼接，等于只用 splitter 不用 chain

**D3 生产级调用：✅ 匹配**

- 重试：模型有 `.with_retry()` 方法，支持配置重试次数和退避
- 超时：支持 timeout 参数
- 流式：`.stream()` 返回分块响应
- 回调：Callback 体系支持日志、追踪、token 用量统计
- 缓存：内置 Cache 层（内存/SQLite/Redis）
- 异常：有自己的异常体系，但不如 LiteLLM 的跨 provider 异常映射统一

**D4 输出解析：✅ 匹配**

- `JsonOutputParser`：解析 JSON 输出
- `PydanticOutputParser`：用 Pydantic 模型约束输出结构，自动生成格式说明注入 prompt
- `OutputFixingParser`：解析失败时自动调 LLM 修复格式
- `XMLOutputParser`：解析 XML（OPML 是 XML 子集，理论上可用）
- `RetryOutputParser`：解析失败时带错误信息重试

**D5 轻量：❌ 不匹配**

LangChain 已拆分为多个包（langchain-core、langchain、langchain-openai、langchain-text-splitters 等），即使只装核心包也有大量依赖。Agent、Chain、Tool、Memory 等概念是框架核心，不用也会引入。版本迭代快，API 历史上多次大改（v0.1→v0.2→v0.3→v1.0），升级有风险。

**总结**：D1✅ D2⚠️（分块器可用但需自己组合） D3✅ D4✅ D5❌。功能最全但最重，且我们需要的分块+拼接模式和它的 Map-Reduce chain 设计不匹配，只能用零件不能用整体。

---

### 3. Pydantic AI

| 项目 | 信息 |
|---|---|
| 官网 | https://ai.pydantic.dev |
| GitHub | https://github.com/pydantic/pydantic-ai |
| Stars | 19,502 |
| 语言 | Python |
| 定位 | Pydantic 团队出品的类型安全 Agent 框架 |

**核心思路**：用 Agent 抽象封装模型调用、指令、工具、依赖注入。输出用 Pydantic 模型声明，自动验证，验证失败自动反思重试。全链路类型标注，IDE 补全友好。支持 OpenAI、Anthropic、Gemini、DeepSeek、Groq、Ollama 等，也可通过 LiteLLM 接入更多模型。

**逐需求对照：**

**D1 统一多模型调用：⚠️ 部分匹配**

- 原生支持 DeepSeek（`deepseek:deepseek-chat`）
- 原生支持 OpenAI 兼容接口（`openai:model` + base_url），豆包理论上可以
- 也支持通过 LiteLLM 接入（`litellm:model`）
- 模型在 Agent 初始化时指定，切换模型改配置即可

**D2 长文本分块：❌ 不提供**

没有文本分块功能，需要自己实现。

**D3 生产级调用：✅ 匹配**

- 流式：支持 `model.stream()` 和 `model.run_stream()`，流式输出结构化数据
- 重试：输出验证失败时自动重试（reflection），LLM 会收到验证错误信息后重新生成
- 超时/错误：依赖底层模型 SDK 的异常
- 可观测性：集成 Pydantic Logfire（OpenTelemetry），可追踪每次调用

**D4 输出解析：✅✅ 最强**

这是 Pydantic AI 的核心卖点：

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class KnowledgePoint(BaseModel):
    name: str
    timestamp: float

agent = Agent(
    'openai:doubao-seed-2-1-pro-260628',
    output_type=list[KnowledgePoint],  # 直接声明输出类型
    system_prompt="..."
)
result = agent.run_sync(text)
points = result.output  # 直接拿到 List[KnowledgePoint]，已验证
```

框架自动生成 JSON Schema 注入 prompt、解析响应、Pydantic 验证、失败自动重试。OPML/XML 没有原生支持，需要自己解析文本。

**D5 轻量：⚠️ 部分匹配**

比 LangChain 轻，但核心抽象是 Agent（含工具调用、依赖注入、Graph、MCP、A2A 等），我们不需要工具调用和 Agent 能力。如果只用 `Agent` + `output_type` 做单次调用，概念上有点重但实际依赖不算太多。

**总结**：D1⚠️ D2❌ D3✅ D4✅✅ D5⚠️。结构化输出最强，但分块没有，Agent 抽象对我们来说偏重，豆包接入需验证。

---

### 4. Instructor

| 项目 | 信息 |
|---|---|
| 官网 | https://python.useinstructor.com |
| GitHub | https://github.com/jxnl/instructor |
| Stars | 13,780 |
| 语言 | Python |
| 定位 | LLM 结构化输出库 |

**核心思路**：用 Pydantic 模型定义期望的输出结构，Instructor 通过 function calling 或 JSON mode 引导 LLM 返回符合 schema 的数据，自动处理验证、重试、流式结构化输出。支持 OpenAI、Anthropic、Google、LiteLLM 等多种后端。

**逐需求对照：**

**D1 统一多模型调用：⚠️ 部分匹配**

Instructor 是在现有 LLM 客户端之上的封装层，支持多种后端（OpenAI、Anthropic、Google、LiteLLM 等）。通过 `instructor.from_openai()` 等方式 patch 客户端。豆包/DeepSeek 用 OpenAI 兼容客户端可以接入。但它不是统一调用层，需要配合底层 SDK 使用。

**D2 长文本分块：❌ 不提供**

**D3 生产级调用：⚠️ 部分匹配**

- 重试：验证失败时自动重试（带错误信息回传 LLM），可配置重试次数
- 流式：支持流式结构化输出（边生成边验证）
- 但限流退避、超时等依赖底层 SDK

**D4 输出解析：✅✅ 强**

和 Pydantic AI 类似，用 Pydantic 模型声明输出，自动验证+重试。支持多态模型、嵌套模型、部分 JSON 提取。比 Pydantic AI 更专注于结构化输出这一件事。

**D5 轻量：✅ 匹配**

只做结构化输出一件事，非常轻量。

**总结**：D1⚠️ D2❌ D3⚠️ D4✅✅ D5✅。是优秀的结构化输出辅助库，但不能独立作为 LLM 层，需要配合其他调用库使用。

---

### 5. DSPy

| 项目 | 信息 |
|---|---|
| 官网 | https://dspy.ai |
| GitHub | https://github.com/stanfordnlp/dspy |
| Stars | 37,596 |
| 语言 | Python |
| 定位 | "编程而非提示词"的 LLM 框架 |

**核心思路**：不手写 prompt，而是用 Signature（声明输入输出）+ Module（推理模块）+ Optimizer（自动搜索最优 prompt）。框架根据评估指标自动优化 prompt。

**逐需求对照：**

D1：✅ 支持多模型（通过 LM 抽象）
D2：❌ 不提供分块
D3：⚠️ 有基础重试但不是重点
D4：✅ 通过 Signature 声明输入输出类型
D5：❌ 理念差异大——我们需要自己控制 prompt（教育场景的提示词需要精心设计），DSPy 的自动优化 prompt 方式不适合我们的场景；学习和调试成本高

**结论**：不匹配。

---

### 6. LlamaIndex

| 项目 | 信息 |
|---|---|
| 官网 | https://www.llamaindex.ai |
| GitHub | https://github.com/run-llama/llama_index |
| Stars | 51,875 |
| 定位 | 文档 Agent 和 RAG 数据框架 |

**核心思路**：文档加载→分块→向量化→索引→检索→生成的完整 RAG 管线。

**逐需求对照：**

D1：✅ 支持多模型
D2：✅ 有分块器（SentenceSplitter 等），但为 RAG 检索设计
D3：✅ 有重试/流式
D4：✅ 有输出解析
D5：❌ 核心是 RAG（向量检索、文档索引），我们不需要

**结论**：不匹配。我们是全文处理不是检索增强。

---

### 7. Haystack

| 项目 | 信息 |
|---|---|
| 官网 | https://haystack.deepset.ai |
| GitHub | https://github.com/deepset-ai/haystack |
| Stars | 26,317 |
| 定位 | 生产级 AI 编排框架 |

**核心思路**：以 Pipeline DAG 为核心，组装检索、路由、生成组件。

**逐需求对照：**

D1：✅ 支持多模型
D2：✅ 有分块器
D3：✅ 生产级 Pipeline
D4：✅ 有输出解析
D5：❌ Pipeline 编排、RAG 组件是核心，我们的流程是线性的不需要 DAG

**结论**：不匹配。

---

### 8. Semantic Kernel

| 项目 | 信息 |
|---|---|
| 官网 | https://learn.microsoft.com/en-us/semantic-kernel/ |
| GitHub | https://github.com/microsoft/semantic-kernel |
| Stars | 28,500 |
| 定位 | 微软的 LLM 集成框架 |

**核心思路**：LLM 函数封装为 Plugin，Planner 自动编排，深度集成 Azure。

**逐需求对照：**

D1：✅ 支持多模型
D2：❌ 不提供分块
D3：✅ 有重试/流式
D4：⚠️ 有基础结构化输出
D5：❌ 以 C# 为主，强绑定 Azure 生态，Python 版本功能弱于 C# 版

**结论**：不匹配。

---

## 三、国内平台分析

### 9. Dify

| 项目 | 信息 |
|---|---|
| 官网 | https://dify.ai |
| GitHub | https://github.com/langgenius/dify |
| Stars | 153,523 |
| 技术栈 | TypeScript + Python |

**核心思路**：可视化 LLMOps 平台，拖拽式工作流编排 + RAG Pipeline + Agent + 模型管理。

**不匹配原因**：是独立运行的平台服务（需要部署 Docker），不是可以 `pip install` 的 Python 库。我们是代码驱动的批处理工具，无法把核心 LLM 调用逻辑交给一个外部平台。

---

### 10. RAGFlow

| 项目 | 信息 |
|---|---|
| 官网 | https://ragflow.io |
| GitHub | https://github.com/infiniflow/ragflow |
| Stars | 89,252 |
| 技术栈 | Python + Go |

**核心思路**：深度文档理解 + RAG 引擎，强调文档解析和检索精度。

**不匹配原因**：纯 RAG 场景，独立平台，不是 Python 库。

---

### 11. FastGPT

| 项目 | 信息 |
|---|---|
| 官网 | https://fastgpt.in |
| GitHub | https://github.com/labring/FastGPT |
| Stars | 29,458 |
| 技术栈 | TypeScript |

**核心思路**：知识库平台 + 可视化工作流。

**不匹配原因**：TypeScript 技术栈，独立平台，不是 Python 库。

---

## 四、对比总表

| 框架 | D1 多模型 | D2 分块 | D3 生产级 | D4 输出解析 | D5 轻量 | 匹配需求数 |
|---|---|---|---|---|---|---|
| **LiteLLM** | ✅ 100+模型，OpenAI格式 | ❌ 有超限异常但无分块 | ✅ 重试/超时/流式/异常映射/Router | ❌ 返回原始文本 | ✅ 极轻量 | 3/5 |
| **LangChain** | ✅ ChatOpenAI+init_chat_model | ⚠️ Splitter可用，Map-Reduce链不匹配 | ✅ with_retry/流式/回调/缓存 | ✅ JSON/Pydantic/XML/自动修复 | ❌ 重，Agent/Chain包袱 | 3.5/5 |
| **Pydantic AI** | ⚠️ DeepSeek原生，豆包需验证 | ❌ | ✅ 流式/验证重试/Logfire | ✅✅ Pydantic类型声明+自动验证 | ⚠️ Agent抽象偏重 | 3/5 |
| **Instructor** | ⚠️ 需配合底层SDK | ❌ | ⚠️ 仅验证重试 | ✅✅ Pydantic+function calling | ✅ 极轻量 | 2.5/5 |
| DSPy | ✅ | ❌ | ⚠️ | ✅ Signature | ❌ 理念不匹配 | 2/5 |
| LlamaIndex | ✅ | ✅ RAG分块器 | ✅ | ✅ | ❌ RAG核心 | 3.5/5 但场景不匹配 |
| Haystack | ✅ | ✅ | ✅ | ✅ | ❌ Pipeline/RAG核心 | 3.5/5 但场景不匹配 |
| Semantic Kernel | ✅ | ❌ | ✅ | ⚠️ | ❌ C#/Azure绑定 | 2/5 |
| Dify/RAGFlow/FastGPT | ✅ | ✅ | ✅ | ✅ | ❌ 独立平台非库 | 不适用 |

---

## 五、方案组合分析

没有任何一个框架 5 个需求全满足。D2（分块）所有框架都不直接提供我们需要的"prompt+payload分块+代码拼接"模式，D4（输出解析）只有 LangChain 和 Pydantic AI/Instructor 提供。

### 方案 A：LiteLLM + 自实现分块 + 自实现解析

- D1：LiteLLM 统一调用
- D2：参考 R-004 自己实现递归分隔符分块（约 100-150 行代码）
- D3：LiteLLM 重试/超时/异常映射 + 自己拼接流式
- D4：Python 标准库 json / xml.etree 解析
- D5：最轻量，只加 litellm 一个依赖

**优点**：依赖最少、完全可控、每个组件只做一件事
**缺点**：分块和解析需要自己写和维护

### 方案 B：LangChain（只用零件）

- D1：ChatOpenAI
- D2：RecursiveCharacterTextSplitter（自己写循环和拼接，不用 Map-Reduce chain）
- D3：with_retry / 流式 / 回调
- D4：JsonOutputParser / PydanticOutputParser
- D5：接受 LangChain 的重量

**优点**：分块器和解析器成熟，开发量最小
**缺点**：依赖重、版本升级风险、Agent/Chain 等不用的概念也会引入、Map-Reduce chain 不匹配需要绕开

### 方案 C：LiteLLM + Instructor + 自实现分块

- D1：LiteLLM 统一调用
- D2：自己实现分块
- D3：LiteLLM + Instructor 双重重试（可能重叠）
- D4：Instructor 处理 JSON 结构化输出和验证重试
- D5：轻量，两个专注的库

**优点**：结构化输出有保障，仍然轻量
**缺点**：多一个依赖；Instructor 的重试和 LiteLLM 的重试可能冲突需要配置；OPML/XML 仍需自己解析

### 方案 D：Pydantic AI + 自实现分块

- D1：Pydantic AI 模型抽象（豆包需验证）
- D2：自己实现分块
- D3：Pydantic AI 流式/重试
- D4：Pydantic output_type 自动验证
- D5：接受 Agent 抽象

**优点**：类型安全最好，结构化输出最强
**缺点**：Agent 框架对我们的简单调用场景偏重；豆包兼容性未验证；分块仍需自己写

---

## 六、参考来源

- LiteLLM 官网：https://docs.litellm.ai
- LiteLLM 异常映射：https://docs.litellm.ai/docs/exception_mapping
- LiteLLM Router：https://docs.litellm.ai/docs/routing
- LiteLLM GitHub：https://github.com/BerriAI/litellm
- LangChain 官网：https://python.langchain.com
- LangChain Text Splitters：https://python.langchain.com/docs/concepts/text_splitters/
- LangChain Summarize（Map-Reduce/Refine）：https://python.langchain.com/v0.2/docs/tutorials/summarization/
- LangChain GitHub：https://github.com/langchain-ai/langchain
- Pydantic AI 官网：https://ai.pydantic.dev
- Pydantic AI GitHub：https://github.com/pydantic/pydantic-ai
- Instructor 官网：https://python.useinstructor.com
- Instructor GitHub：https://github.com/jxnl/instructor
- DSPy 官网：https://dspy.ai
- DSPy GitHub：https://github.com/stanfordnlp/dspy
- LlamaIndex 官网：https://www.llamaindex.ai
- LlamaIndex GitHub：https://github.com/run-llama/llama_index
- Haystack 官网：https://haystack.deepset.ai
- Haystack GitHub：https://github.com/deepset-ai/haystack
- Semantic Kernel 官网：https://learn.microsoft.com/en-us/semantic-kernel/
- Semantic Kernel GitHub：https://github.com/microsoft/semantic-kernel
- Dify 官网：https://dify.ai / GitHub：https://github.com/langgenius/dify
- RAGFlow 官网：https://ragflow.io / GitHub：https://github.com/infiniflow/ragflow
- FastGPT 官网：https://fastgpt.in / GitHub：https://github.com/labring/FastGPT
