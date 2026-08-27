# R-001 豆包（火山引擎）API 调研报告

> 调研日期：2026-08-24（文档调研），2026-08-27（实测验证）
> 文档版本：v2.0（含实测验证）
> 实测模型：doubao-seed-2-1-pro-260628、deepseek-v4-pro
> 实测脚本：scripts/test_doubao_api.py、scripts/test_deepseek_api.py

---

## 总结论

**Chat Completions API 已能满足需求，无需切换到 Responses API。**

实测发现豆包和 DeepSeek 在 Chat Completions 格式下均通过 `reasoning_content` 字段返回思维链内容，LiteLLM 可直接接收。当前 LiteLLM 适配器只需补充收集 `delta.reasoning_content` 即可支持深度思考，无需重写为 Responses API。

| 验证项 | 结论 |
|---|---|
| 豆包 Chat Completions 返回 reasoning_content | **已验证**，非流式和流式均有 |
| DeepSeek Chat Completions 返回 reasoning_content | **已验证**，非流式和流式均有 |
| 是否必须用 Responses API | **否**，Chat Completions 已返回思维链 |
| thinking 参数通过 LiteLLM extra_body 传递 | **已验证**，参数被接受 |
| 非流式调用是否超时 | 简单问题 20s 正常返回；长任务仍建议流式 |
| 多模态输入 | **已验证**，Responses API 的 input_image + input_text 格式可用 |
| 错误码分类 | **已验证**，无效模型返回 400 BadRequestError |

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
| Responses API | `POST /api/v3/responses` | 新一代 API，原生支持深度思考、多模态、工具调用、结构化输出、上下文缓存 | 多模态输入（图片+文本） |
| Chat API | `POST /api/v3/chat/completions` | 兼容 OpenAI 格式，LiteLLM 统一调用 | **文本生成主力接口** |

**实测结论（2026-08-27 修正）**：

原报告认为"必须使用 Responses API 才能获取思维链"，**实测验证此结论不成立**。豆包 doubao-seed-2-1-pro 在 Chat Completions 格式下，响应消息中已包含 `reasoning_content` 字段（非流式 237 字，流式通过 `delta.reasoning_content` 增量返回）。DeepSeek-v4-pro 同样如此。

因此：
- **文本生成**：继续使用 Chat Completions（LiteLLM），只需在适配器中补充收集 `reasoning_content`
- **多模态输入**：使用 Responses API（Chat Completions 的多模态格式未在本次验证中测试，Responses API 已验证可用）
- 当前代码通过 LiteLLM 走 Chat Completions 的方向是正确的，不需要重写

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

#### 3.3 超时处理

> **官方警告**：深度思考模型在非流式输出场景中容易因超时导致任务失败，推荐使用流式输出。

**实测验证（2026-08-27）**：
- 简单数学题（集合交集并集）非流式 + thinking 开启：20.8 秒正常返回，未超时
- 流式 + thinking：25.7 秒完成
- 对于长文本处理（如 ASR 纠错 8000 字），非流式仍有超时风险，**建议统一使用流式**

实践要求：
1. 所有 LLM 调用**内部统一使用流式模式**（`stream=True`），聚合后返回上层（当前 LiteLLM 适配器已实现）
2. `timeout` 设置要足够大（深度思考可能需要 30-120 秒）
3. `max_output_tokens` / `max_tokens` 需设置足够大（建议至少 8192，复杂任务 16384+）

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

> 2026-08-27 更新：VolcengineAdapter 已被 LiteLLMAdapter 替代（Issues #2-#5），以下问题状态已更新。

1. ~~**非流式调用**~~：**已修复**，LiteLLMAdapter 内部统一 `stream=True`
2. **未处理思维链**：**待修复**，Chat Completions 响应中 `reasoning_content` 字段已确认存在，但 `_collect_stream()` 只收集 `delta.content`，需补充收集 `delta.reasoning_content`
3. ~~**thinking 硬编码 disabled**~~：**已移除**，当前代码不设置 thinking 参数（模型默认开启）
4. ~~**max_tokens 命名错误**~~：**不适用**，LiteLLM 走 Chat Completions 格式，`max_tokens` 是正确参数名
5. ~~**裸 except Exception**~~：**已修复**，LiteLLMAdapter 分类捕获各异常类型
6. ~~**硬编码默认配置**~~：**已修复**，模型名从 ModelConfig 读取

---

## 七、实测验证详情（2026-08-27）

### 7.1 豆包 doubao-seed-2-1-pro-260628

**测试1：LiteLLM Chat Completions 基本调用**
- 耗时 23.7s，HTTP 200
- `reasoning_content` 字段存在（237 字），通过 `message.reasoning_content` 获取
- 内容质量好，正确使用 LaTeX 格式
- usage: prompt=75, completion=1042, total=1117

**测试2：LiteLLM 通过 extra_body 传 thinking 参数**
- 耗时 21.8s，参数被接受
- `reasoning_content` 存在（219 字）
- 模型即使不显式传 thinking 参数也会返回 reasoning_content（2.1 pro 默认开启深度思考）

**测试3：直连 Responses API 流式**
- 耗时 25.7s，HTTP 200
- 事件类型：`response.reasoning_summary_text.delta`（307 条）、`response.output_text.delta`（411 条）
- reasoning_summary 长度 597 字，output_text 长度 574 字
- 注意：`requests.iter_lines(decode_unicode=True)` 对 SSE 流的中文解码有乱码问题，非流式无此问题
- 结论：如需直连 Responses API 流式，需正确处理编码；通过 LiteLLM 则无此问题

**测试4：直连 Responses API 非流式**
- 耗时 20.8s，HTTP 200
- output 结构：`output[0].type="reasoning"`（含 summary_text）+ `output[1].type="message"`（含 output_text）
- usage 包含 `output_tokens_details.reasoning_tokens`: 782
- 中文正常，无编码问题

**测试5：Responses API 多模态输入**
- 耗时 10.6s，HTTP 200
- 使用 `input_image`（base64）+ `input_text` 格式
- 正确识别图片中的文字内容和颜色

### 7.2 DeepSeek deepseek-v4-pro

**测试1：LiteLLM Chat Completions 基本调用**
- 耗时 5.7s，HTTP 200
- `reasoning_content` 字段存在（94 字），通过 `message.reasoning_content` 获取
- usage: prompt=112, completion=95, total=207
- 回答简洁，直接给出结果

**测试2：LiteLLM 流式调用**
- 耗时 2.6s，HTTP 200
- `delta.reasoning_content` 增量返回（66 字）
- `delta.content` 增量返回（85 字）
- usage 正常返回

**测试3：异常处理**
- 无效模型名返回 400 BadRequestError
- 错误消息明确列出支持的模型：`deepseek-v4-pro, deepseek-v4-flash, deepseek-v4-flash-vision-exp`
- 注意：DeepSeek 有视觉模型 `deepseek-v4-flash-vision-exp`（实验性）

### 7.3 两家对比

| 维度 | 豆包 doubao-seed-2-1-pro | DeepSeek deepseek-v4-pro |
|---|---|---|
| 响应速度（简单问题） | ~22s | ~5s |
| reasoning_content 长度 | 200-600 字 | 60-100 字 |
| 回答风格 | 详细讲解，LaTeX 格式 | 简洁直接 |
| 思维链获取方式 | Chat Completions 的 `reasoning_content` | 相同 |
| 多模态 | Responses API 已验证 | 有 vision-exp 模型（未验证） |
| 适用场景 | 知识点整理、题目解析（需要详细讲解） | ASR 纠错、简单提取（速度快） |

### 7.4 对代码的影响

1. **LLMResponse 需新增 `reasoning_content` 字段**，用于保存思维链（可输出到 debug）
2. **LiteLLMAdapter._collect_stream() 需补充收集 `delta.reasoning_content`**
3. **不需要切换到 Responses API**，Chat Completions 已满足文本生成需求
4. **多模态场景**（将来 OCR+LLM）需单独处理，可通过 LiteLLM 的 OpenAI 兼容多模态格式或直连 Responses API
5. **thinking 控制**：豆包 2.1 pro 默认开启 thinking，是否需要按任务关闭以节省 token 待讨论

---

## 八、建议（按实测更新）

### 建议 1：任务-模型映射（已实现）

`config.yaml` 的 `tasks` 节已支持每个任务独立配置 model 和 temperature。thinking/reasoning 参数暂不需要在配置中暴露——豆包 2.1 pro 默认开启深度思考，通过 Chat Completions 的 `reasoning_content` 即可获取思维链。

实测速度对比：豆包 ~22s vs DeepSeek ~5s（简单问题）。DeepSeek 适合对速度敏感的任务（如 ASR 纠错），豆包适合需要详细讲解的任务（知识点整理、题目解析）。

### 建议 2：流式调用封装（已实现）

LiteLLMAdapter 已统一内部流式接收、对外同步返回。需补充：收集 `delta.reasoning_content` 并放入 LLMResponse。

### 建议 3：异常分类（已实现）

LiteLLMAdapter 已分类捕获 BadRequestError / AuthenticationError / RateLimitError / InternalServerError / Timeout / APIConnectionError，映射为项目自定义异常。

### 建议 4：思维链 debug 输出（待实现）

深度思考模型的 `reasoning_content` 可保存到 `debug/{视频名}/llm_reasoning/` 下，用于分析模型推理过程，排查输出质量问题。

### 建议 5：多模态支持（待实现）

将来 OCR+LLM 融合时，多模态输入通过 Responses API 的 `input_image` + `input_text` 格式（已验证可用），或通过 LiteLLM 的 OpenAI 兼容多模态格式。

---

*文档结束*
