# M11 LLM 客户端模块 — 详细设计

> 接口文档：`03_接口设计/M11_LLM客户端模块接口.md`
> 模块编号：M11

---

## 一、模块结构

```
core/
└── llm_client.py            # LLM 客户端业务模块
adapters/
└── llm_adapter.py           # LLM 适配层（VLLMAdapter / DeepSeekAdapter）
```

---

## 二、类设计

### 2.1 LLMClient 类

```python
class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.local_adapter = create_adapter("vllm", config.local.__dict__)
        self.cloud_adapter = create_adapter("deepseek", config.cloud.__dict__)
        self.max_retries = config.max_retries
        self._cache_dir = "./cache/llm"

    def chat(self, messages, backend="auto", temperature=0.3, top_p=0.9, max_tokens=2000, use_cache=True) -> LLMResponse
    def chat_stream(self, messages, backend="auto", temperature=0.3, top_p=0.9, max_tokens=2000) -> Iterator[LLMChunk]
    def select_backend(self, messages, task_type="general") -> str
    def count_tokens(self, text, backend="local") -> int
    def get_context_length(self, backend) -> int
```

---

## 三、核心流程

### 3.1 chat 流程

```
1. 如果 backend="auto"，调用 select_backend() 选择后端
2. 获取对应适配器（local_adapter / cloud_adapter）
3. 计算缓存键
4. use_cache=True 且缓存存在 → 加载缓存，返回
5. 执行带重试的适配器调用：
   for attempt in range(max_retries):
     try:
       response = adapter.chat(messages, temperature, top_p, max_tokens)
       break
     except (LLMTimeoutError, LLMRateLimitError, LLMServerError) as e:
       if attempt == max_retries - 1:
         raise LLMError(f"重试 {max_retries} 次后失败: {e}")
       wait_time = 5 * (2 ** attempt)  # 指数退避 5s→10s→20s→40s→80s
       if 速率限制且有 Retry-After 头:
         wait_time = max(wait_time, retry_after)
       logger.warning(f"LLM 调用失败，{wait_time}s 后重试 ({attempt+1}/{max_retries}): {e}")
       time.sleep(wait_time)
6. 写入缓存
7. 返回 LLMResponse
```

### 3.2 select_backend 逻辑

```python
def select_backend(self, messages, task_type="general"):
    # 规则1：特定任务类型强制用云端
    cloud_tasks = ["knowledge_extraction", "problem_extraction", "mindmap"]
    if task_type in cloud_tasks:
        return "cloud"

    # 规则2：输入 token 接近本地上限，用云端
    total_text = " ".join(m["content"] for m in messages)
    input_tokens = self.count_tokens(total_text, "local")
    local_context = self.get_context_length("local")  # 8192
    if input_tokens > local_context * 0.75:  # 超过 75% 上限
        return "cloud"

    # 规则3：其他用本地
    return "local"
```

### 3.3 chat_stream 流程

流式调用不使用缓存（流式响应难以缓存），直接调用适配器：

```python
def chat_stream(self, messages, backend="auto", ...):
    if backend == "auto":
        backend = self.select_backend(messages)
    adapter = self.local_adapter if backend == "local" else self.cloud_adapter
    return adapter.chat_stream(messages, temperature, top_p, max_tokens)
```

流式调用的重试较复杂（流已经开始后失败难以重试），MVP 阶段流式调用不做重试，失败直接抛出。

---

## 四、重试机制详解

### 4.1 可重试异常分类

```python
RETRYABLE_EXCEPTIONS = (
    TimeoutError,           # 网络超时
    ConnectionError,        # 连接重置/拒绝
    LLMServerError,         # HTTP 5xx
    LLMRateLimitError,      # HTTP 429
)

NON_RETRYABLE_EXCEPTIONS = (
    LLMClientError,         # HTTP 4xx（除429）
    LLMContentFilterError,  # 内容过滤
    InvalidRequestError,    # 请求参数错误
)
```

### 4.2 指数退避

```python
def get_backoff_time(attempt: int) -> float:
    base = 5  # 基础等待 5 秒
    return base * (2 ** attempt)  # 5, 10, 20, 40, 80
```

### 4.3 速率限制的 Retry-After

如果异常响应包含 `Retry-After` 头，使用该值作为等待时间：
```python
retry_after = getattr(e, "retry_after", None)
if retry_after:
    wait_time = max(wait_time, float(retry_after))
```

---

## 五、缓存机制

### 5.1 缓存键

```python
def get_cache_key(messages, backend, model, temperature, top_p, max_tokens):
    key_data = {
        "messages": messages,
        "backend": backend,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

### 5.2 缓存格式

```json
{
  "response": {
    "content": "...",
    "model": "deepseek-chat",
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    "finish_reason": "stop"
  },
  "backend": "cloud",
  "created_at": "2026-08-23T10:00:00"
}
```

---

## 六、适配器初始化

```python
from adapters.llm_adapter import create_adapter

# 本地 vLLM
self.local_adapter = create_adapter("vllm", {
    "base_url": config.local.base_url,
    "model": config.local.model,
    "context_length": 8192
})

# 云端 DeepSeek
self.cloud_adapter = create_adapter("deepseek", {
    "base_url": config.cloud.base_url,
    "model": config.cloud.model,
    "api_key": config.cloud.api_key,  # 从环境变量读取
    "context_length": 131072
})
```

---

## 七、异常处理

| 异常 | 触发条件 | 处理方式 |
|---|---|---|
| LLMError | 重试耗尽后仍失败 | 抛出，中断流水线 |
| LLMRateLimitError | 429 速率限制 | 重试（读取 Retry-After） |
| LLMTimeoutError | 超时 | 重试（指数退避） |
| InvalidBackendError | 无效后端选择 | 抛出，检查配置 |
| LLMClientError | 4xx 客户端错误 | 不重试，直接抛出 |

---

## 八、性能考虑

- 本地 vLLM：延迟取决于模型和输入长度，压测显示 TPOT ~35ms/token
- 云端 DeepSeek：网络延迟 + 推理延迟，通常 10-60 秒
- 重试机制增加了最坏情况下的总耗时（最多 5 次重试 × 80 秒退避 = 约 7 分钟）
- 缓存命中时耗时为 0（直接读取本地文件）

---

## 九、测试要点

1. 本地 vLLM 调用
2. 云端 DeepSeek 调用
3. 自动后端选择逻辑
4. 重试机制（模拟超时/5xx/429）
5. 指数退避时间计算
6. 缓存命中和失效
7. 流式调用
8. token 计数准确性
9. 上下文长度获取
10. 不可重试异常的直接抛出
