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
        self._health_status = {"local": None, "cloud": None}  # 健康状态缓存

    def chat(self, messages, backend, temperature=0.3, top_p=0.9, max_tokens=2000, use_cache=True) -> LLMResponse
    def chat_stream(self, messages, backend, temperature=0.3, top_p=0.9, max_tokens=2000) -> Iterator[LLMChunk]
    def health_check(self, backend) -> bool
    def reconnect(self, backend) -> bool
    def count_tokens(self, text, backend="local") -> int
    def get_context_length(self, backend) -> int
```

---

## 三、核心流程

### 3.1 chat 流程

```
1. 校验 backend 参数（必须是 "local" 或 "cloud"）
2. 获取对应适配器（local_adapter / cloud_adapter）
3. 计算缓存键
4. use_cache=True 且缓存存在 → 加载缓存，返回
5. 健康检查（如果距离上次检查超过 health_check_interval 秒）
   a. 健康检查失败 → 尝试重连
   b. 重连失败 → 抛出 LLMConnectionError
6. 执行带重试的适配器调用：
   for attempt in range(max_retries):
     try:
       response = adapter.chat(messages, temperature, top_p, max_tokens)
       break
     except (LLMTimeoutError, LLMRateLimitError, LLMServerError, LLMConnectionError) as e:
       if attempt == max_retries - 1:
         raise LLMError(f"重试 {max_retries} 次后失败: {e}")
       wait_time = 5 * (2 ** attempt)  # 指数退避 5s→10s→20s→40s→80s
       if 速率限制且有 Retry-After 头:
         wait_time = max(wait_time, retry_after)
       logger.warning(f"LLM 调用失败，{wait_time}s 后重试 ({attempt+1}/{max_retries}): {e}")
       time.sleep(wait_time)
7. 写入缓存
8. 返回 LLMResponse
```

### 3.2 chat_stream 流程

流式调用不使用缓存，直接调用适配器：

```python
def chat_stream(self, messages, backend, ...):
    # 校验 backend
    adapter = self.local_adapter if backend == "local" else self.cloud_adapter
    # 健康检查（同 chat）
    # 流式调用不做重试（流已经开始后失败难以重试）
    return adapter.chat_stream(messages, temperature, top_p, max_tokens)
```

流式调用的重试较复杂，MVP 阶段流式调用不做重试，失败直接抛出。

---

## 四、健康检查机制

### 4.1 health_check 实现

```python
def health_check(self, backend: str) -> bool:
    adapter = self.local_adapter if backend == "local" else self.cloud_adapter
    try:
        # 发送极简请求，3秒超时
        response = adapter.chat(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=3
        )
        self._health_status[backend] = time.time()
        return True
    except Exception:
        self._health_status[backend] = None
        return False
```

### 4.2 健康检查触发时机

| 时机 | 说明 |
|---|---|
| 流水线启动时 | 检查 local 和 cloud 两个后端，记录初始健康状态 |
| 每次 chat 调用前 | 如果距离上次检查超过 `health_check_interval`（默认30秒），重新检查 |
| 调用失败时 | 立即触发健康检查，判断是服务故障还是临时网络问题 |
| 手动调用 | `pipeline.health_check()` 可手动触发 |

### 4.3 健康状态缓存

```python
self._health_status = {
    "local": 1724563200.0,  # 上次健康检查通过的时间戳，None=不健康
    "cloud": 1724563200.0,
}
```

---

## 五、断线重连机制

### 5.1 reconnect 实现

```python
def reconnect(self, backend: str) -> bool:
    adapter = self.local_adapter if backend == "local" else self.cloud_adapter
    # 重建适配器客户端（销毁旧连接，建立新连接）
    adapter.rebuild_client()
    # 重连后立即做健康检查
    return self.health_check(backend)
```

### 5.2 重连触发条件

| 条件 | 处理 |
|---|---|
| 健康检查失败 | 自动尝试重连1次，重连成功则继续，失败则抛出 LLMConnectionError |
| 调用时连接被重置 | 计入重试，重试前自动尝试重连 |
| vLLM 服务重启 | 健康检查检测到不可达，自动重连 |
| 网络临时中断 | 重试机制的指数退避通常能覆盖，无需显式重连 |

### 5.3 重连次数限制

单次调用流程中，重连最多尝试 1 次。重连失败则抛出异常，由上层决定是否整体中断。MVP 阶段不做无限重连（避免卡死）。

---

## 六、重试机制

### 6.1 可重试异常分类

```python
RETRYABLE_EXCEPTIONS = (
    TimeoutError,           # 网络超时
    ConnectionError,        # 连接重置/拒绝
    LLMServerError,         # HTTP 5xx
    LLMRateLimitError,      # HTTP 429
    LLMConnectionError,     # 健康检查失败/重连失败
)

NON_RETRYABLE_EXCEPTIONS = (
    LLMClientError,         # HTTP 4xx（除429）
    LLMContentFilterError,  # 内容过滤
    InvalidRequestError,    # 请求参数错误
)
```

### 6.2 指数退避

```python
def get_backoff_time(attempt: int) -> float:
    base = 5  # 基础等待 5 秒
    return base * (2 ** attempt)  # 5, 10, 20, 40, 80
```

### 6.3 速率限制的 Retry-After

如果异常响应包含 `Retry-After` 头，使用该值作为等待时间：
```python
retry_after = getattr(e, "retry_after", None)
if retry_after:
    wait_time = max(wait_time, float(retry_after))
```

---

## 七、缓存机制

### 7.1 缓存键

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

### 7.2 缓存格式

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

## 八、适配器初始化

```python
from adapters.llm_adapter import create_adapter

# 本地 vLLM
self.local_adapter = create_adapter("vllm", {
    "base_url": config.local.base_url,
    "model": config.local.model,
    "context_length": config.local.context_length,
})

# 云端 DeepSeek
self.cloud_adapter = create_adapter("deepseek", {
    "base_url": config.cloud.base_url,
    "model": config.cloud.model,
    "api_key": config.cloud.api_key,  # 从环境变量读取
    "context_length": config.cloud.context_length,
})
```

适配器类型从配置的 `adapter_type` 字段读取，支持通过配置切换引擎。

---

## 九、异常处理

| 异常 | 触发条件 | 处理方式 |
|---|---|---|
| LLMError | 重试耗尽后仍失败 | 抛出，中断流水线 |
| LLMRateLimitError | 429 速率限制 | 重试（读取 Retry-After） |
| LLMTimeoutError | 超时 | 重试（指数退避） |
| LLMConnectionError | 连接失败/健康检查未通过 | 尝试重连，重连失败则重试或抛出 |
| InvalidBackendError | 无效后端选择 | 抛出，检查调用代码 |
| LLMClientError | 4xx 客户端错误 | 不重试，直接抛出 |

---

## 十、性能考虑

- 本地 vLLM：延迟取决于模型和输入长度，压测显示 TPOT ~35ms/token
- 云端 DeepSeek：网络延迟 + 推理延迟，通常 10-60 秒
- 重试机制增加了最坏情况下的总耗时（最多 5 次重试 × 80 秒退避 = 约 7 分钟）
- 缓存命中时耗时为 0（直接读取本地文件）
- 健康检查增加了每次调用前的 3 秒检测（仅在超过间隔时触发）

---

## 十一、测试要点

1. 本地 vLLM 调用
2. 云端 DeepSeek 调用
3. 重试机制（模拟超时/5xx/429）
4. 指数退避时间计算
5. 缓存命中和失效
6. 流式调用
7. token 计数准确性
8. 上下文长度获取
9. 不可重试异常的直接抛出
10. 健康检查（正常/不可达）
11. 断线重连（模拟服务重启）
12. 无效 backend 的错误处理
13. 适配器类型从配置读取
