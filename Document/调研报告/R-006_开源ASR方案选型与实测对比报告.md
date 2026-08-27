# R-006 开源 ASR 方案选型与实测对比报告

> 调研日期：2026-08-26
> 文档版本：v1.0
> 开发环境：Windows 11 + RTX 3080 10GB + CUDA 12.6 + cuDNN 8.9.7
> 技术栈：Python 3.11、funasr==1.4.2、torch（CUDA 版）
> 测试音频：tests/test.wav，16kHz 单声道 PCM，时长 25 分 08 秒，高中数学集合第一课

---

## 调研目标

调研免费开源、无 license 限制的 ASR 库，并通过实际测试对比各方案在中文教学视频场景下的识别能力、速度和工程可用性，确定本项目的 ASR 选型。

---

## 总结论

**现阶段唯一可行方案：FunASR Paraformer（paraformer-zh + fsmn-vad + ct-punc）。**

| 方案 | 耗时 | 文本长度 | 字级时间戳 | License | 结论 |
|---|---|---|---|---|---|
| **FunASR Paraformer** | **22.5s** | 8157 字 | **有** | MIT 代码 + 模型协议（需署名） | **采用** |
| Fun-ASR-Nano | 33.3s / 10s 音频 | 仅短音频验证 | **无**（CTC 张量缺失） | Apache 2.0 | 不采用（见下文） |
| 豆包音频理解 API | 192.5s | 8211 字 | 无 | 云端付费 API | 备选（非开源） |

**Fun-ASR-Nano 不采用的原因**：
1. 非 vLLM 模式下 RTF=3.24（比实时慢 3 倍），25 分钟音频预计需约 81 分钟，不可接受
2. checkpoint 缺少 86 个 CTC 张量，**字级时间戳不可用**——这对本项目的文字-音频定位流程是硬伤
3. 官方 340x realtime 性能需 vLLM 加速，Windows 平台 vLLM 支持有限

**豆包音频理解 API 不作为主力方案的原因**：
1. 无时间戳，无法支持字级定位
2. 需联网、按 token 付费，不适合本地批量处理
3. 识别质量（标点、断句）确实优于 FunASR，可作为未来云端备选

---

## 一、调研范围

### 1.1 候选方案

| 项目 | Code License | 模型 License | 中文 CER（官方） | Stars |
|---|---|---|---|---|
| FunASR Paraformer | MIT | FunASR 模型协议（需署名，允许商用） | 10.18% | 20K |
| Fun-ASR-Nano | Apache 2.0 | Apache 2.0 | 8.20%（vLLM） | 20K（同仓库） |
| OpenAI Whisper | MIT | MIT | ~20% | 108K |
| sherpa-onnx | Apache 2.0 | 取决于模型来源 | 同所用模型 | 14.4K |
| Vosk | Apache 2.0 | Apache 2.0 | 一般（Kaldi 老模型） | 15K |
| NVIDIA NeMo | Apache 2.0 | CC-BY-4.0 等 | 英文强，中文弱 | 18.3K |
| WeNet | Apache 2.0 | Apache 2.0 | 较好 | 5.2K |
| SenseVoice | MIT | FunASR 模型协议 | 7.81% | 9.1K |

> 数据来源：https://github.com/modelscope/FunASR

### 1.2 筛选逻辑

- **Whisper**：中文 CER 约 20%，是 FunASR 的 2 倍，且无内置 VAD/标点，排除
- **Vosk**：基于 Kaldi 老架构，中文精度不如现代模型，排除
- **NeMo**：中文支持弱，框架重，排除
- **WeNet**：Apache 2.0 且中文好，但偏训练框架，预训练模型不如 FunASR 丰富，集成成本高，暂不考虑
- **sherpa-onnx**：运行时 Apache 2.0，但 Paraformer ONNX 模型权重仍受 FunASR 模型协议约束；自身不提供模型，排除
- **SenseVoice**：中文 CER 最低（7.81%），但模型权重使用与 Paraformer 相同的 FunASR 模型协议，且本项目已验证 Paraformer 可用，暂不额外引入
- **Fun-ASR-Nano**：Apache 2.0 模型权重，进入实测
- **豆包音频理解 API**：作为云端对照进入实测

最终实测三个方案：FunASR Paraformer、Fun-ASR-Nano、豆包音频理解 API。

---

## 二、实测环境与方法

### 2.1 测试音频

- 文件：tests/test.mp4 → tests/test.wav
- 提取参数（与项目代码一致）：`ffmpeg -i test.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1`
- 时长：1508.5 秒（25 分 08 秒）
- 内容：高中数学"集合第一课"教学视频，中文讲解为主，开头有英文 "Hello everybody"

### 2.2 测试脚本

| 脚本 | 说明 |
|---|---|
| scripts/test_funasr.py | FunASR Paraformer 测试，参照项目 adapters/asr/funasr.py 写法 |
| scripts/test_fun_asr_nano.py | Fun-ASR-Nano 测试，参照 FunASR 官方教程文档写法 |
| scripts/test_doubao_audio.py | 豆包音频理解 API 测试，通过火山方舟 Responses API + Files API |

### 2.3 输出文件

| 文件 | 说明 |
|---|---|
| tests/FunASR.txt | FunASR Paraformer 完整识别结果 |
| tests/Fun-ASR-Nano.txt | Fun-ASR-Nano 短音频（10s）验证结果 |
| tests/豆包.txt | 豆包音频理解完整识别结果 |

---

## 三、实测结果

### 3.1 FunASR Paraformer

**调用方式**（严格参照项目 adapters/asr/funasr.py）：

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    punc_model="ct-punc",
    punc_model_revision="v2.0.4",
)
res = model.generate(input=wav_path, batch_size_s=300)
```

**性能数据**：

| 指标 | 值 |
|---|---|
| 总耗时 | 22.5 秒 |
| RTF（实时率） | 0.015（约 67x realtime） |
| 输出文本长度 | 8157 字 |
| VAD 分段数 | 约 10 段 |
| GPU 显存峰值 | 约 2GB |
| 字级时间戳 | 有（res[0]["timestamp"]） |

**识别质量**：
- 中文内容识别准确，专业术语（集合、元素、互异性、列举法、描述法等）全部正确
- 开头英文 "Hello, everybody" 正确识别
- 数学符号口语化表达正确（如"a方等于四"、"正负一"、"二零二二次幂"）
- 标点由 ct-punc 模型自动添加，基本合理
- 个别同音字错误（如"撒花"识别为"早啊"），不影响语义

**关键经验**：
- **VAD 模型（fsmn-vad）是必需的**。首次测试未加载 VAD 模型时，25 分钟音频作为单个 batch 处理，GPU 显存占满 9.8GB、运行 35 分钟以上无结果。加上 VAD 后 22.5 秒完成。
- VAD 将长音频切分为短段后批量推理，是 FunASR 处理长音频的标准方式。

### 3.2 Fun-ASR-Nano

**调用方式**（严格参照 FunASR 官方教程文档）：

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)
res = model.generate(input=wav_path, cache={}, batch_size=1, language="中文")
```

> 官方文档来源：https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md "Speech Recognition (Fun-ASR-Nano)" 章节

**短音频验证结果**（10 秒片段）：

| 指标 | 值 |
|---|---|
| 音频时长 | 10 秒 |
| 识别耗时 | 33.3 秒 |
| RTF | 3.24（比实时慢 3.24 倍） |
| 识别结果 | "hello everybody，我是神奇小猪，欢迎大家来到集合第一课。在今天这个视频里面，我们将给大家讲清楚什么是集合，为" |

**完整音频测试**：
- 运行 35 分钟以上未完成，手动终止
- 按 RTF=3.24 推算，25 分钟音频需约 81 分钟
- 进程占用 40GB 系统内存、9.8GB 显存

**致命问题：字级时间戳不可用**

模型加载时输出警告：

```
Warning, miss key in ckpt: ctc_decoder.linear1.weight ...（共 86 个 CTC 张量缺失）
Disabling CTC timestamps because the checkpoint did not initialize 86 of 86 required CTC tensors.
Text transcription remains available.
```

HuggingFace 上的 Fun-ASR-Nano checkpoint 未包含 CTC 解码器权重，导致 `res[0]["timestamps"]` 不可用。本项目的核心流程（LLM 返回文字段 → 与 ASR 文本模糊匹配 → 定位字级时间戳 → 截取视频帧）依赖 ASR 提供字级时间戳，此问题为硬伤。

**vLLM 加速的可行性**：
- 官方 340x realtime 性能数据基于 vLLM 推理引擎
- vLLM 在 Windows 上原生支持有限（官方推荐 Linux + WSL2）
- 本项目需考虑大部分用户为 Windows 环境，不能强制要求 WSL2
- 即使使用 vLLM，时间戳缺失问题仍然存在

### 3.3 豆包音频理解 API

**调用方式**（参照火山方舟官方文档）：

```python
# 1. 通过 Files API 上传音频（46MB WAV）
POST https://ark.cn-beijing.volces.com/api/v3/files
# 2. 通过 Responses API 识别
POST https://ark.cn-beijing.volces.com/api/v3/responses
# model: doubao-seed-2-0-lite-260428
# input: [{type: "input_audio", file_id: "..."}, {type: "input_text", text: "请完整识别..."}]
```

> 官方文档来源：https://docs.volcengine.com/docs/82379/2377589

**性能数据**：

| 指标 | 值 |
|---|---|
| 音频大小 | 46.0 MB |
| 上传耗时 | 约 3 秒 |
| 识别耗时 | 约 189 秒 |
| 总耗时 | 192.5 秒 |
| 输出文本长度 | 8211 字 |
| 字级时间戳 | 无 |

**识别质量**：
- 识别内容与 FunASR 基本一致，文本长度接近（8211 vs 8157）
- 标点和断句优于 FunASR：正确使用逗号、句号、感叹号、问号，段落分隔自然
- "π" 正确识别（FunASR 识别为"派"）
- "21世纪" 正确识别（FunASR 识别为"二十一世纪"，语义等价）
- 部分口语化填充词处理更干净

**局限性**：
- 无时间戳，无法用于字级定位
- 需联网，依赖云端服务可用性
- 按 token 计费，大批量处理有成本
- 音频文件需上传至云端，存在隐私考量

---

## 四、方案对比总结

### 4.1 核心维度对比

| 维度 | FunASR Paraformer | Fun-ASR-Nano | 豆包音频理解 |
|---|---|---|---|
| 中文识别准确率 | 好（CER 10.18%） | 好（CER 8.20%，vLLM） | 好 |
| 识别速度（3080） | **22.5s / 25min 音频** | 33.3s / 10s 音频 | 192.5s / 25min 音频 |
| 字级时间戳 | **有** | **无** | 无 |
| 本地运行 | **是** | 是 | 否（云端） |
| 成本 | **免费** | 免费 | 按 token 付费 |
| 网络依赖 | 无（模型已下载） | 无（模型已下载） | 必须联网 |
| 隐私 | **本地处理** | 本地处理 | 音频上传云端 |
| 模型 License | MIT 代码 + 模型协议（署名） | Apache 2.0 | 云端服务 |
| Windows 兼容 | **好** | 好（但慢） | N/A |
| VAD + 标点 | **内置（fsmn-vad + ct-punc）** | VAD 有，标点无 | 自动 |

### 4.2 License 说明

FunASR 源码为 MIT，但 Paraformer 模型权重使用 [FunASR Model Open Source License Agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)：
- 允许商用、修改、分发
- **要求署名**（保留模型名称和出处）
- 有"禁止诋毁"条款（违反则许可自动终止）

这不是 OSI 认证的标准开源协议，但对于本项目（教学工具，非竞品）不构成实际风险。只需在产品中注明使用了 FunASR Paraformer 模型即可。

---

## 五、字级时间戳存储与对齐方案设计

### 5.1 现有问题：Sentence 抽象多余

当前 `adapters/asr/funasr.py` 的 `_split_to_sentences()` 用正则按 `。？！` 将完整文本切成 `Sentence` 列表，只保留句子级起止时间，**丢弃了 FunASR 返回的字级时间戳**。

问题：
1. 切分逻辑是适配器自己发明的，不是 ASR 引擎的输出。FunASR 的 VAD 分段是音频分段（用于批量推理），输出的是连续文本 + 字级时间戳，没有"句子"概念
2. 标点由 ct-punc 模型预测，会出错，基于错误标点切句子会传递错误
3. 下游流程（LLM 返回文字段 → 全文匹配 → 取首尾时间戳）操作的是全文本和字符索引，不需要"第N个句子"
4. 切完句子后字级时间戳被丢弃，而后续字符级定位恰恰需要它

**结论**：重构中去掉 `Sentence` 和 `_split_to_sentences()`，ASR 适配器直接返回完整文本 + 字级时间戳。

### 5.2 数据结构设计

时间戳只存在一个地方（ASR 原始输出），纠错和匹配产生的不确定性由显式对齐结构表达。

```python
@dataclass
class CharTime:
    """单个字符的时间戳（毫秒）"""
    start_ms: int
    end_ms: int

@dataclass
class RawTranscript:
    """ASR 原始输出，时间戳的唯一权威来源"""
    text: str                                   # ASR 原始文本（含标点）
    char_timestamps: List[Optional[CharTime]]   # 与 text 等长，标点位置为 None

@dataclass
class AlignedTranscript:
    """纠错后文本 + 与原始文本的对齐映射"""
    text: str                         # LLM 纠错后的全文
    raw_align: List[Optional[int]]    # 与 text 等长，值为 RawTranscript.text 的字符索引；
                                      # None 表示 LLM 新增字（标点、修正字），无直接时间戳
    raw: RawTranscript                # 引用原始数据
```

### 5.3 两次对齐与失配处理

时间戳从 ASR 到最终文字段定位，需穿越两次对齐：

```
原始ASR文本 ──(对齐1: LLM纠错)──→ 纠错后文本 ──(对齐2: 模糊匹配)──→ LLM返回的文字段
   ↑字级时间戳                        ↑工作文本                       ↑要取时间戳的目标
```

**对齐1：LLM 纠错对齐（Myers diff 序列对齐）**

LLM 纠错会增删改字符（"派"→"π"、加标点、改断句）。用 Myers diff 算法计算原始文本到纠错文本的最小编辑序列，生成 `raw_align` 映射：
- 匹配的字：`raw_align[i]` 指向原始字索引，继承时间戳
- LLM 替换的字（如"派"→"π"）：对齐到被替换字的位置，继承时间戳（语音没变，只是文字修正）
- LLM 新增的字（标点、连词）：`raw_align[i] = None`，取时间戳时跳过

**对齐2：文字段模糊匹配**

LLM 返回的文字段是纠错后文本的"近似引用"，不会逐字相同。用编辑距离/LCS 在全文中找到最佳匹配区间，返回 `(start_idx, end_idx, confidence)`：
- 中间偶尔 1-2 个字对不上（同音不同字、漏字）：不影响首尾定位
- 相似度高于阈值：取匹配区间
- 相似度低于最低阈值：降级为关键词匹配（方案B）

**取时间戳**：

```
文字段区间 [start_idx, end_idx)
    → 遍历 raw_align[start_idx..end_idx]
    → 跳过 None，找到第一个和最后一个非 None 的 raw_index
    → 从 raw.char_timestamps[raw_index] 取时间
```

各环节失配处理汇总：

| 环节 | 失配情况 | 处理 |
|---|---|---|
| 对齐1（纠错） | LLM 改字（"派"→"π"） | Myers diff 做替换对齐，继承原字时间戳 |
| 对齐1（纠错） | LLM 加标点/连词 | raw_align 对应位置=None，取时间时跳过 |
| 对齐2（匹配） | 文字段中个别字与全文不一致 | 模糊匹配找最佳区间，不要求逐字相同 |
| 对齐2（匹配） | 整段相似度低于阈值 | 降级关键词匹配 |
| 取时间戳 | 区间内有 None（标点） | 跳过 None，取最近的有时间戳的字 |

### 5.4 对齐1算法细节：Myers diff

#### 5.4.1 算法原理

Myers diff（Eugene Myers, 1986）是 Git diff 使用的最小编辑距离算法。核心思想：**将两个文本的对齐转化为网格图上的最短路径问题**。

设原始文本 A 沿 x 轴，纠错后文本 B 沿 y 轴：

- **对角线移动（↘）**：A[i] == B[j]，匹配，无编辑代价
- **向右移动（→）**：删除 A 中的字（A 有 B 没有）
- **向下移动（↓）**：插入 B 中的字（B 有 A 没有）

目标：从 (0,0) 走到 (N,M)，走最多的对角线，最少的→↓。

定义对角线编号 `k = x - y`。算法按编辑距离 d 递增迭代：d=0 只能走对角线；d=1 允许 1 次编辑；d=2 允许 2 次……直到到达终点。每轮记录每条对角线上能到达的最远 x 坐标（贪婪策略：同样编辑次数，走得越远越好）。

#### 5.4.2 在本项目中的应用示例

输入：
- A（ASR 原始）：`"派的近似值能不能构成集合"`
- B（LLM 纠错）：`"π的近似值，能不能构成集合？"`

Myers diff 输出编辑脚本：

```
替换: A[0]"派" → B[0]"π"
匹配: A[1..4]"的近似值" == B[1..4]"的近似值"
插入: B[5]"，"
匹配: A[5..11]"能不能构成集合" == B[6..12]"能不能构成集合"
插入: B[13]"？"
```

生成 `raw_align` 映射：

```
B 的位置:  0    1    2    3    4    5     6    7    ...  12    13
B 的字符:  π    的   近   似   值   ，     能   不   ...  合    ？
raw_align: 0    1    2    3    4   None   5    6    ...  11   None
           ↑                            ↑
    替换，继承A[0]时间戳          LLM新增标点，无时间戳
```

取时间戳：
- B[0] "π" → raw_align[0]=0 → A[0] "派" 的时间戳（语音相同，仅文字修正）
- B[5] "，" → raw_align[5]=None → 跳过，取相邻字时间
- 文字段区间 [0, 6) → 第一个非 None 是 0，最后一个非 None 是 5 → 取 A[0].start_ms 到 A[5].end_ms

#### 5.4.3 替换 vs 删除+插入

标准 Myers diff 只有 match/insert/delete 三种操作。"派"→"π" 会被表示为 delete "派" + insert "π"，导致 "π" 的 raw_align=None，丢失时间戳。

**后处理规则**：回溯编辑脚本时，如果一个 insert 紧跟在同位置的 delete 后面，合并为替换，让 B 的字继承 A 的字的时间戳。语义依据：LLM 改的是字，不是语音。

#### 5.4.4 性能评估

| 文本长度 | 编辑距离 d | 时间复杂度 | 预估耗时 |
|---|---|---|---|
| 8,000 字 | ~100（加标点+少量改字） | O((N+M)×d) | <10ms |
| 40,000 字 | ~500 | O((N+M)×d) | <200ms |

LLM 纠错改动量很小（主要是加标点、改同音字），d 远小于 N，性能不是问题。

#### 5.4.5 实现选择

使用 Python 标准库 `difflib.SequenceMatcher`（Ratcliff-Obershelp 算法，非 Myers 本身，但输出格式等价），其 `get_opcodes()` 自动返回 equal/replace/delete/insert 四种操作，且**自动将相邻的 delete+insert 合并为 replace**，正好满足 5.4.3 的需求：

```python
import difflib

matcher = difflib.SequenceMatcher(None, raw_text, corrected_text)
for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == 'equal':    # 匹配，B[j1:j2] ↔ A[i1:i2]，直接继承时间戳
    elif tag == 'replace': # 替换，B[j1:j2] 继承 A[i1:i2] 时间戳
    elif tag == 'insert':  # B 新增，raw_align = None
    elif tag == 'delete':  # A 有 B 没有，丢弃
```

无需引入第三方库，无需自行实现 Myers。

### 5.5 内存与序列化评估

| 视频长度 | 字数 | Python dataclass 内存 | JSON 序列化耗时 |
|---|---|---|---|
| 25 分钟（测试视频） | 8,000 | ~1.1 MB | ~20-40ms |
| 90 分钟（典型教学） | 20,000 | ~2.8 MB | ~50-80ms |
| 3 小时（极长） | 40,000 | ~5.6 MB | ~100-150ms |

对比：FunASR 模型占 2GB 显存，LLM 上下文占几百 MB，整个流水线耗时分钟级。时间戳数据的内存开销（几 MB）和序列化开销（<0.1%）均可忽略。

**结论**：直接使用 Python dataclass，不引入 `array('i')` 等紧凑结构。序列化仅在缓存读写时发生（项目已有 JSON 缓存机制），不在每步处理中反复序列化/反序列化。

---

## 六、调研教训

1. **VAD 模型是 FunASR 长音频处理的关键**：首次测试未加载 fsmn-vad，导致 25 分钟音频作为单个 batch 处理，显存占满、运行 35 分钟无结果。加上 VAD 后 22.5 秒完成。项目代码中已正确配置 VAD，测试脚本不能省略。

2. **官方性能数据需看测试条件**：Fun-ASR-Nano 的 340x realtime 基于 vLLM，非 vLLM 模式下 RTF=3.24（反而比实时慢）。不能直接引用官方数据做选型判断，必须在目标硬件上实测。

3. **新模型的 checkpoint 可能不完整**：Fun-ASR-Nano 的 HuggingFace checkpoint 缺少 86 个 CTC 张量，导致时间戳功能被禁用。模型的"文本识别可用"不等于"所有功能可用"，需逐项验证。

4. **写测试脚本必须参照官方文档**：Fun-ASR-Nano 需要 `trust_remote_code=True`、`hub="hf"`、`vad_kwargs={"max_single_segment_time": 30000}`、`cache={}`、`language="中文"` 等参数，缺少任何一个都可能导致异常或性能问题。

5. **云端 ASR 的时间戳缺失是架构性限制**：豆包音频理解 API 识别质量好，但不返回时间戳。这不是参数配置问题，而是 API 能力边界。对于需要字级定位的场景，本地 ASR 不可替代。

---

*文档结束*
