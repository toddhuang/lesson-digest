# M11 LLM 客户端模块 — 接口设计

> 所属文档：02_架构设计.md
> 模块编号：M11
> 模块职责：LLM 调用业务逻辑（模型注册表管理、任务-模型映射、会话创建），通过适配层调用具体 LLM 服务

---

## 一、模块概述

LLM 客户端模块负责管理模型注册表和服务商配置，按任务名创建 LLM 会话。**不直接依赖豆包或 DeepSeek 的 API**，而是通过 M17 LLM 适配层的 `LiteLLMAdapter` 统一调用具体 LLM 服务。

> **M11-M17 重构变化**：
> - 旧接口 `chat` / `chat_stream` / `count_tokens` / `get_context_length` 全部废弃，改为 `get_session(task_name)` 返回 `LLMSession`
> - 删除多服务商 `backend` 别名（`"cloud"` / `"deepseek"` / `"volcengine"`），改为模型注册表 + 任务映射
> - 删除 LLM 响应缓存（`use_cache`），缓存由 pipeline 层统一管理（断点续传）
> - 重试 / 超时 / 流式接收下沉到适配器层（`LiteLLMAdapter` 内部处理）

> **AGENTS.md 约定**：一次 LLM 调用返回三样东西（纠错全文 + 知识点段 + 题目段），不做三次独立调用。每个 LLM 调用点在配置中独立指定 provider，默认豆包，DeepSeek 适配器保留不删除。

---

## 二、接口列表

### 2.1 get_session

**功能**：根据任务名获取 LLM 会话，会话绑定具体模型和 temperature。

**签名**：
```python
def get_session(self, task_name: str) -> LLMSession
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| task_name | str | 是 | 任务名（对应 `config.yaml` 中 `tasks` 节的键，如 `asr_correct_and_extract`） |

**输出**：`LLMSession`（绑定了适配器实例和 temperature 的会话，实现 `LLMGenerator` 协议）

**流程**：
1. 从 `tasks[task_name]` 取 `TaskConfig`（含 model 名 + temperature）
2. 从 `llm_config.models[model]` 取 `ModelConfig`（含 provider / capabilities / context_length / max_output）
3. `_get_adapter(model_name)` 懒加载并缓存适配器实例（按模型名缓存，同一模型复用一个适配器）
4. 创建 `LLMSession(adapter, temperature, model_name)`

**异常**：
| 异常类型 | 触发条件 |
|---|---|
| ConfigError | 任务未配置，或任务引用了未注册的模型，或模型引用了未配置的服务商 |

---

### 2.2 health_check

**功能**：健康检查，发送一个极短请求验证连通性。

**签名**：
```python
def health_check(self, task_name: str) -> bool
```

**输入参数**：
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| task_name | str | 是 | 任务名（使用该任务配置的模型） |

**输出**：`bool`（True 表示健康，False 表示异常）

**实现**：调用 `get_session(task_name)` 后发送 `session.generate(prompt="你是一个健康检查助手。", payload="请回复：ok")`，捕获 `LLMError` 返回 False。

---

## 三、数据结构

> 共享数据结构（`LLMResponse`、`TokenUsage`）统一定义见 `00_数据模型.md` 第四节。
> `LLMSession` 实现 `core.llm.protocol.LLMGenerator` 协议，详见 `M11_M17_LLM接口重构设计.md`。

---

## 四、第三方库调用（通过适配层）

本模块**不直接调用**豆包或 DeepSeek API，而是通过 M17 LLM 适配层的统一接口调用：

```python
from adapters.llm.factory import create_llm_adapter

# 根据模型+服务商配置创建适配器（LiteLLMAdapter 统一处理多服务商）
adapter = create_llm_adapter(model_config, provider_config, max_retries, timeout, mock=False)

# 业务模块通过 LLMSession 调用（2 参数，temperature 由会话注入）
session = llm_client.get_session("asr_correct_and_extract")
response = session.generate(prompt=SYSTEM_PROMPT, payload=text)
```

适配层统一接口详见：`M17_LLM适配层接口.md`

---

## 五、任务使用策略

每个 LLM 调用点独立配置模型和 temperature，在 `config.yaml` 的 `tasks` 节映射：

| 任务 | temperature | 说明 |
|---|---|---|
| `asr_correct_and_extract` | 0.1 | ASR 纠错 + 知识点段 + 题目段一次性提取（AGENTS.md 约定的合并调用） |
| `problem_extraction` | 0.0 | 题目原题提取（基于题目段 + OCR），输出最确定 |
| `knowledge_summary` | 0.3 | 知识点深度整理，需要一定创造性 |
| `solution_summary` | 0.2 | 解题过程整理，适度灵活 |
| `mindmap_generation` | 0.3 | 思维导图生成，需要一定创造性 |

> Mock 适配器用于链路测试（`Pipeline(mock_llm=True)`），不依赖真实 API。

---

## 六、多服务商配置

### 6.1 配置结构

模型注册表 + 服务商配置 + 任务映射三段式，API Key 直接写在 `config.yaml`（已在 `.gitignore`，不会提交）：

```yaml
llm:
  max_retries: 3
  timeout: 120

  # 模型注册表：所有可用的模型在此注册
  # capabilities 可选值：text / reasoning / vision
  models:
    - name: doubao-seed-2-1-pro-260628
      provider: volcengine
      capabilities: [text, reasoning]
      context_length: 256000
      max_output: 16384
    - name: deepseek-chat
      provider: deepseek
      capabilities: [text]
      context_length: 131072
      max_output: 8192

  # 服务商配置
  # litellm_prefix: LiteLLM 提供商前缀，OpenAI 兼容接口统一用 "openai"
  providers:
    volcengine:
      base_url: "https://ark.cn-beijing.volces.com/api/v3"
      api_key: ""  # 在火山引擎方舟控制台创建，填入 config.yaml
      litellm_prefix: "openai"
    deepseek:
      base_url: "https://api.deepseek.com"
      api_key: ""  # 在 DeepSeek 平台创建，填入 config.yaml
      litellm_prefix: "openai"

# 任务-模型映射：每个任务指定使用的模型和 temperature
tasks:
  asr_correct_and_extract:
    model: doubao-seed-2-1-pro-260628
    temperature: 0.1
  problem_extraction:
    model: doubao-seed-2-1-pro-260628
    temperature: 0.0
  mindmap_generation:
    model: doubao-seed-2-1-pro-260628
    temperature: 0.3
```

### 6.2 切换服务商 / 模型

修改 `config.yaml` 中 `tasks.{task_name}.model` 指向另一注册模型即可，无需修改代码。每个调用点可独立配置，满足 AGENTS.md "每个 LLM 调用点可独立配置服务商" 约定。

> **已废弃**：`default_provider`、`.env` 环境变量方案（`${VOLCENGINE_API_KEY}`）。改为 API Key 直接写 `config.yaml`（模板文件 `config.example.yaml` 中 api_key 必须为空字符串）。

---

## 七、依赖关系

- **被依赖**：`ContentExtractor`（`asr_correct_and_extract`）、`ProblemExtractor`（`problem_extraction`）、`MindmapGenerator`（`mindmap_generation`）等业务模块，通过 `LLMSession` 调用
- **依赖**：M17 LLM 适配层（`LiteLLMAdapter` / `MockLLMAdapter`，隔离豆包/DeepSeek）、M1 配置管理（`ModelConfig` / `ProviderConfig` / `TaskConfig`）
- **第三方依赖**：通过适配层间接依赖 litellm、豆包 API、DeepSeek API
