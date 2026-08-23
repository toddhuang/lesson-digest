# M11 LLM 客户端模块 — 详细设计

> 接口文档：`03_接口设计/M11_LLM客户端模块接口.md`
> 模块编号：M11

---

## 一、模块结构

```
core/
└── llm_client.py            # LLM 客户端业务模块（多服务商管理）
adapters/
└── llm_adapter.py           # LLM 适配层（VolcengineAdapter / DeepSeekAdapter / MockAdapter）
```

---

## 二、类设计

### 2.1 LLMClient 类

```python
class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.default_provider = config.default_provider  # "volcengine" / "deepseek"
        self.providers = {}  # provider_name -> LLMAdapter
        self.max_retries = config.max_retries
        self._cache_dir = "./cache/llm"
        self._init_providers()

    def _init_providers(self):
        """根据配置初始化所有服务商适配器"""
        for name, provider_config in self.config.providers.items():
            if provider_config.enabled:
                self.providers[name] = create_adapter(name, provider_config.__dict__)

    def chat(self, messages, backend="cloud", temperature=0.3, top_p=0.9,
             max_tokens=2000, use_cache=True, provider=None) -> LLMResponse
    def chat_stream(self, messages, backend="cloud", temperature=0.3,
                    top_p=0.9, max_tokens=2000, provider=None) -> Iterator[LLMChunk]
    def count_tokens(self, text) -> int
    def get_context_length(self, provider=None) -> int
    def get_available_providers(self) -> List[str]
```

> **重要变更**：移除 `local_adapter`、`health_check`、`reconnect` 方法。纯云端架构，不再使用本地 vLLM。`backend="cloud"` 映射到 `default_provider`（默认 volcengine/豆包）。

---

## 三、核心流程

### 3.1 chat 流程

```
1. 确定服务商：
   a. 如果指定 provider 参数 → 使用该服务商
   b. 否则使用 default_provider（默认 volcengine/豆包）
2. 获取对应适配器（self.providers[provider_name]）
3. Token 超限检测：
   a. 统计 messages 总 token 数（含 system + user + 预留 max_tokens）
   b. context_length = adapter.get_context_length()（豆包=131072, DeepSeek=131072）
   c. 如果 total_tokens > context_length → 抛出 LLMContextOverflowError
4. 计算缓存键
5. use_cache=True 且缓存存在 → 加载缓存，返回
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

### 3.2 服务商选择逻辑

```python
def _get_provider(self, backend: str, provider: Optional[str] = None) -> str:
    if provider:
        if provider not in self.providers:
            raise LLMError(f"服务商 {provider} 未配置或未启用")
        return provider
    # backend="cloud" 映射到 default_provider
    if backend == "cloud":
        return self.default_provider
    raise LLMError(f"不支持的 backend: {backend}，纯云端架构仅支持 backend='cloud'")
```

---

## 四、缓存机制

### 4.1 缓存键

```
cache_key = md5(provider + model + messages_json + temperature + top_p + max_tokens)
```

### 4.2 缓存格式

JSON 文件，存储于 `cache/llm/{cache_key}.json`：

```json
{
  "provider": "volcengine",
  "model": "doubao-seed-2-1-pro-260628",
  "content": "...",
  "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
  "timestamp": "2026-08-23T10:00:00"
}
```

---

## 五、异常处理

| 异常类型 | 触发条件 | 处理方式 |
|---|---|---|
| LLMError | 通用 LLM 错误 | 抛出 |
| LLMTimeoutError | 请求超时 | 重试（最多 max_retries 次） |
| LLMRateLimitError | 速率限制（429） | 重试，等待 Retry-After |
| LLMServerError | 服务端错误（5xx） | 重试 |
| LLMConnectionError | 连接失败 | 重试 |
| LLMContextOverflowError | Token 超出上下文长度 | 抛出，不重试 |
| LLMResponseParseError | 响应解析失败 | 抛出 |

---

## 六、配置示例

```yaml
llm:
  default_provider: volcengine  # 默认服务商：豆包
  max_retries: 5
  providers:
    volcengine:
      enabled: true
      api_key: "${VOLCENGINE_API_KEY}"
      base_url: "${VOLCENGINE_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
      model: "${VOLCENGINE_MODEL:-doubao-seed-2-1-pro-260628}"
      context_length: 131072
    deepseek:
      enabled: true
      api_key: "${DEEPSEEK_API_KEY}"
      base_url: "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
      model: "${DEEPSEEK_MODEL:-deepseek-v4-flash}"
      context_length: 131072
```

---

## 七、依赖关系

- **被依赖**：M7 知识点提取、M8 题目提取、M10 思维导图生成、M12 输出组装（ASR 纠错）、M14 工具集（ASR 纠错工具）
- **依赖**：M17 LLM 适配层（VolcengineAdapter / DeepSeekAdapter / MockAdapter）、M1 配置管理、M14 工具集（token 计数、缓存）
- **第三方依赖**：通过适配层间接依赖豆包 API（默认）/ DeepSeek API（备选）
