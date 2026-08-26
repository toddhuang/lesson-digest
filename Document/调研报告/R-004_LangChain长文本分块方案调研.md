# R-004 LangChain 长文本分块方案调研

> 调研目标：调研 LangChain 官方文档中关于长文本分块和上下文超限处理的方案，评估是否适用于本项目（prompt + payload 超限时分块处理），重点关注如何降低切片引入的 bug。
> 调研时间：2026-08-26
> 信息来源：LangChain 官方文档（python.langchain.com / docs.langchain.com）

---

## 一、LangChain 文本分块方案

LangChain 官方文档将文本分块分为四种策略：

### 1. 基于长度的分块（Length-based）

- **Token-based**：按 token 数切分，使用 tiktoken 编码器精确计算
- **Character-based**：按字符数切分

```python
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

### 2. 基于文本结构的分块（Text-structured）—— RecursiveCharacterTextSplitter

**这是 LangChain 官方推荐的通用分块器**，核心思路是递归降级：

1. 首先尝试按段落分隔符（`\n\n`）切分，保持段落完整
2. 如果某个段落仍然超过 chunk_size，降级为按行（`\n`）切分
3. 如果某行仍然超长，降级为按句子边界切分
4. 最终降级到词/字符级别

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
texts = text_splitter.split_text(document)
```

**关键参数：**
- `chunk_size`：每块最大长度
- `chunk_overlap`：块间重叠长度（用于保持上下文连续）
- `separators`：分隔符优先级列表，按顺序尝试
- `length_function`：长度计算函数（默认 `len`，可传入 token 计数器）

### 3. 基于文档结构的分块（Document-structured）

按文档固有结构切分：Markdown 按标题、HTML 按标签、JSON 按对象、代码按函数。不适用于我们的纯文本 ASR 场景。

### 4. 基于语义的分块（Semantic）

通过 embedding 计算相邻句子的语义相似度，在语义变化大的地方断开。需要额外 embedding 模型调用，成本高，适合 RAG 检索场景。

---

## 二、LangChain 长文档处理三策略

LangChain 官方 summarization 教程中提出了三种处理超长文档的策略：

| 策略 | 做法 | 适用场景 |
|---|---|---|
| **Stuff** | 全部内容塞进一个 prompt | 内容不超长时 |
| **Map-Reduce** | 每块独立调 LLM（Map），再将结果合并（Reduce）；如果 Map 结果累积超限，递归 collapse | 多文档摘要、各块独立处理 |
| **Refine** | 顺序遍历每块，每块带上之前的结果迭代更新 | 需要顺序上下文的摘要 |

**Map-Reduce 的 Reduce 阶段细节**（来自官方文档）：
- `ReduceDocumentsChain` 累积所有 Map 结果
- 如果累积 token 超过 `token_max`（默认 4000），分批传入 collapse chain 压缩
- 压缩后如果仍然超限，递归压缩
- 最终一次性生成结果

---

## 三、对本项目的适用性分析

### 我们的场景

```
prompt（指令，固定） + payload（ASR全文，可能超长）
    ↓ 超限时
payload 分块 → 每块用相同 prompt 调 LLM → 结果文本拼接返回
```

### LangChain 方案的适配评估

| LangChain 组件 | 是否适用 | 理由 |
|---|---|---|
| RecursiveCharacterTextSplitter | **适用，参考其设计** | 递归分隔符策略能最大程度保持语义完整，降低切片 bug |
| Token-based 长度计算 | **适用** | 比字符估算更精确，减少超限风险 |
| chunk_overlap | **不适用** | 我们是结果拼接模式，overlap 会导致重复处理和重复输出 |
| Stuff 策略 | 适用（未超限时） | 直接调用，不分块 |
| Map-Reduce 链 | **不适用** | Reduce 阶段是再调 LLM 合并摘要，我们的 Reduce 是代码拼接文本 |
| Refine 链 | **不适用** | 顺序迭代更新模式，每块依赖前一块结果，不适合我们的独立处理+拼接 |
| 语义分块 | **不适用** | 需要 embedding 模型，成本高，ASR 文本结构简单不需要 |
| 引入 LangChain 依赖 | **不建议** | 我们只需要分块逻辑，LangChain 是重型框架（含 agent、RAG、chain 等），引入会增加大量不必要的依赖 |

---

## 四、结论与建议

### 结论

**不引入 LangChain 作为依赖**，但参考其 `RecursiveCharacterTextSplitter` 的设计思路自行实现分块逻辑。

理由：
1. 我们只需要分块功能，不需要 LangChain 的 chain/agent/RAG 等重型组件
2. LangChain 的 Map-Reduce/Refine 链是为摘要场景设计的，和我们的"分块独立处理+文本拼接"模式不同
3. 自行实现可以精确控制分块行为，降低引入框架的不确定性

### 分块实现建议（降低切片 bug 的关键措施）

**1. 采用递归分隔符策略（参考 RecursiveCharacterTextSplitter）**

按优先级尝试分隔符，尽量在自然边界断开：
```
段落边界 \n\n → 换行 \n → 中文句末标点 。！？ → 空格 → 强制切分
```
不在句子中间切断，保持语义完整。

**2. 基于 token 估算，不基于字符数**

使用 token 计数计算 chunk_size，避免中文字符估算误差导致超限。chunk_size 计算公式：
```
chunk_size = context_length × 0.9 - prompt_tokens - reserved_output_tokens
```
留 10% 余量防止估算误差。

**3. 不使用 chunk_overlap**

我们是结果拼接模式，overlap 会导致同一段内容被处理两次，输出重复。通过递归分隔符策略保证语义完整即可。

**4. 分块前校验**

- prompt + 预留输出超过 context_length 的 90% 时直接报错（prompt 本身太大，分块无法解决）
- 单块（即使按最小语义单位）仍然超限时报错，不静默截断

**5. 输出格式由 prompt 保证可拼接**

分块场景下，业务模块在 prompt 中要求输出 JSONL（每行一个对象）而非 JSON 数组，这样各块输出直接拼接后逐行解析即可，不需要 LLM 模块感知 JSON 格式。

**6. 分块信息记录到 debug**

记录每块的起止位置、token 数、分块数量，便于排查分块导致的问题。

---

## 五、参考来源

- [Text splitters 概念文档](https://python.langchain.com/docs/concepts/text_splitters/) — LangChain 官方
- [RecursiveCharacterTextSplitter 使用指南](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — LangChain 官方
- [Summarize Text 教程（Stuff / Map-Reduce / Refine）](https://python.langchain.com/v0.2/docs/tutorials/summarization/) — LangChain 官方 v0.2
- [How-to guides: Text splitters](https://python.langchain.com/docs/how_to/#text-splitters) — LangChain 官方
