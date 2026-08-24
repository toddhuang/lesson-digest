# R-001 豆包（火山引擎）API 调研报告

> 调研日期：2026-08-24
> 调研人：AI Assistant
> 文档版本：v1.0

---

## 调研目标

1. 确认火山引擎豆包系列模型的当前可用列表、各模型能力与限制（上下文长度、思维链长度、多模态、工具调用等）
2. 确认 Responses API 与 Chat API 的差异，选定重构后统一使用的接口
3. 确认深度思考（thinking）机制的开启方式、响应格式、超时风险与应对策略
4. 确认多模态输入的消息格式（图片+文本混合输入的正确写法）
5. 确认 API 错误码体系，为异常分类处理提供依据
6. 发现当前代码中 `VolcengineAdapter` 的实现问题，为重构提供准确的修改依据

---

## 调研结果

### 一、当前可用模型列表

以下为 2026-08 可用且非"即将下线"状态的模型：

| 模型 ID | 定位 | 上下文窗口 | 最大输入 | 最大回答 | 最大思维链 | 深度思考默认 | 多模态 | 工具调用 | 结构化输出 |
|---|---|---|---|---|---|---|---|---|---|
| `doubao-seed-evolving` | 快速迭代旗舰 | 1024k | 1024k | 256k | 256k | enabled(high) | 是 | 是 | 是(json_schema) |
| `doubao-seed-2-1-pro-260628` | Seed 2.1 旗舰 | 256k | 256k | 256k | 256k | enabled(high) | 是 | 是 | 是(json_schema) |
| `doubao-seed-2-1-turbo-260628` | Seed 2.1 高速 | 256k | 256k | 256k | 256k | enabled(high) | 是 | 是 | 是 |
| `doubao-seed-2-0-lite-260428` | Seed 2.0 轻量 | 256k | 224k | 128k | 128k | enabled | 是 | 是 | 是(json_schema) |
| `doubao-seed-2-0-mini-260428` | Seed 2.0 迷你 | 256k | 224k | 128k | 128k | enabled | 是 | 是 | 是 |
| `doubao-seed-character-260628` | 角色模型 | 128k | 96k | 32k | 128k | enabled | 是 | 是 | 是(json_schema) |

同平台可调用的第三方模型：`deepseek-v4-pro-ga-260813`、`deepseek-v4-flash-ga-260731`、`glm-5-2-260617`。

> 数据来源：火山引擎官方模型列表文档（2026-08-17 更新）

### 二、API 接口选择

| 接口 | 端点 | 特点 | 推荐场景 |
|---|---|---|---|
| **Responses API** | `POST /api/v3/responses` | 新一代 API，原生支持深度思考、多模态、工具调用、结构化输出、上下文缓存 | **推荐**，所有新功能使用此接口 |
| Chat API | `POST /api/v3/chat/completions` | 兼容 OpenAI 格式，部分老模型仅支持此接口 | 仅用于兼容老代码 |

**结论：重构后统一使用 Responses API。** 当前代码中 `VolcengineAdapter` 已使用 `client.responses.create()`，方向正确，但需完善深度思考和流式处理。

### 三、深度思考（Thinking）机制

#### 3.1 控制参数

```json
{
  "thinking": {"type": "enabled"},
  "reasoning": {"effort": "high"}
}
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `thinking.type` | `enabled` | 强制开启深度思考（2.1 系列默认值） |
| | `disabled` | 关闭深度思考，直接回答 |
| | `auto` | 模型自主判断（仅部分老模型支持，2.1 系列不支持） |
| `reasoning.effort` | `minimal` | 关闭思考，直接回答 |
| | `low` | 轻量思考，快速响应 |
| | `medium` | 均衡模式（通用默认值，但 2.1 系列默认 high） |
| | `high` | 深度分析（2.1 系列默认值） |
| | `xhigh` / `max` | 更高程度思考（仅部分模型支持） |

#### 3.2 响应结构

深度思考模型的输出包含**两部分**：

1. **思维链摘要（reasoning summary）**：模型思考过程的摘要，非原始思考内容
2. **最终回答（output_text）**：模型的最终答案

**非流式响应**：一次性返回完整结果，但深度思考模型**容易超时**。

**流式响应（官方推荐）**：通过 SSE 事件流返回，关键事件类型：

| 事件类型 | 说明 |
|---|---|
| `response.reasoning_summary_text.delta` | 思维链摘要增量文本 |
| `response.output_text.delta` | 最终回答增量文本 |
| `response.output_text.done` | 最终回答完成（含聚合文本） |
| `response.completed` | 全部完成（含 usage 统计） |

#### 3.3 超时处理（关键风险）

> **官方明确警告**：深度思考模型在非流式输出场景中容易因超时导致任务失败，**推荐使用流式输出**。

实践要求：
1. 所有深度思考调用**必须使用流式模式**（`stream=True`）
2. 如业务需要非流式结果，先以流式获取完整内容后再聚合
3. `timeout` 设置要足够大（深度思考可能需要 30-120 秒）
4. `max_output_tokens` = 思维链长度 + 回答长度，需设置足够大（建议至少 8192，复杂任务 16384+）

### 四、多模态输入格式

Responses API 的 `input` 字段支持两种格式：

**纯文本（string）**：
```json
{"model": "...", "input": "请分析这个数学问题..."}
```

**多模态（array）**：
```json
{
  "model": "...",
  "input": [{
    "role": "user",
    "content": [
      {"type": "input_image", "image_url": "https://.../image.png"},
      {"type": "input_text", "text": "你看见了什么？"}
    ]
  }]
}
```

**关键注意事项**：
- `content.type` 必须是 `input_text`，**不能写成 `text`**（会报 400 参数错误）
- 图片支持 `url`（http/https/tos/s3）、`base64`、`binary`
- 最多支持 10 张图片

### 五、错误码与异常分类

| HTTP 状态码 | 错误码 | 含义 | 处理策略 |
|---|---|---|---|
| 400 | `InvalidParameter.*` | 参数错误 | 检查请求参数，不重试 |
| 401 | - | 认证失败 | 检查 API Key，不重试 |
| 403 | - | 权限不足 | 检查模型开通状态，不重试 |
| 429 | `RateLimitExceeded.EndpointRPMExceeded` | 请求频率超限 (RPM) | 指数退避重试 |
| 429 | `RateLimitExceeded.EndpointTPMExceeded` | Token 流量超限 (TPM) | 指数退避重试，考虑降低并发 |
| 500 | - | 服务端内部错误 | 延迟重试（偶发错误） |
| 504 / 超时 | - | 上游拥塞 / 请求过长 | 检查输入长度，改用流式，重试 |

### 六、当前代码 `VolcengineAdapter` 的问题清单

1. **非流式调用**：`client.responses.create(stream=False)`，深度思考模型容易超时
2. **未处理思维链**：直接取 `output_text`，忽略了 `reasoning` 部分
3. **`thinking` 硬编码 disabled**：`extra_body={"thinking": {"type": "disabled"}}`，关闭了深度思考
4. **`max_tokens` 命名错误**：Responses API 用 `max_output_tokens`，当前传 `max_tokens` 可能不生效
5. **裸 `except Exception`**：未分类处理异常（对应代码问题追踪 C-002）
6. **硬编码默认配置**：`default_config` 中写死 model 名称（对应代码问题追踪 C-003）

---

## 额外建议

### 建议 1：每个 LLM 任务独立配置

在 `config.yaml` 的 `llm.tasks` 下为每个任务独立配置 provider、model、thinking、reasoning_effort、max_output_tokens、timeout、temperature：

| 任务 | 推荐 model | thinking | effort | max_output_tokens | timeout |
|---|---|---|---|---|---|
| ASR 纠错 | doubao-seed-2-1-pro | enabled | medium | 8192 | 120s |
| 知识点+题目粗提取（一次调用） | doubao-seed-2-1-pro | enabled | high | 16384 | 180s |
| 知识点深度整理 | doubao-seed-2-1-pro | enabled | high | 8192 | 120s |
| 题目原题/解题分离 | doubao-seed-2-1-pro | enabled | high | 8192 | 120s |
| 思维导图生成 | doubao-seed-2-1-turbo | disabled | - | 4096 | 60s |

### 建议 2：流式调用封装

在 LLM 适配层统一封装流式调用逻辑，对外暴露同步接口（内部流式聚合），避免每个业务模块都处理 SSE 事件。思维链摘要可保存到 debug，最终回答返回给业务层。

### 建议 3：异常分类重试机制

实现统一的重试装饰器：
- 429 限流：指数退避（初始 1s，最大 30s，最多 5 次）
- 500 服务端错误：固定延迟重试（最多 3 次）
- 超时：检查是否非流式调用深度思考模型，自动降级为流式后重试
- 400/401/403：不重试，立即抛出明确异常

### 建议 4：思维链 debug 输出

深度思考模型的 reasoning 摘要可保存到 `debug/{视频名}/llm_reasoning/` 下，用于分析模型推理过程，排查输出质量问题。

---

*文档结束*
