# Debug 模块设计

> 版本：v1.0（待确认后定案）
> 日期：2026-09-01
> 状态：设计中，待用户确认 6 点决策后推进实现
> 所属模块：debugger/（新建独立包）
> 关联 issue：#11 [F-001]
> 前置文档：02_架构设计.md、08_文字对比定位核心算法.md、09_解题过程ASR_OCR融合提取设计.md、10_知识点深度整理设计.md

---

## 一、设计目标

### 1.1 解决的问题

当前 debug 产物散落各处：
- 知识点截图、解题过程截图已写入 `debug/{视频名}/06_截图/`（issue #12、#13 实现）
- ASR 原始逐字稿、纠错后全文、文字段、定位记录**只存 PipelineContext 内存，未持久化**
- 没有统一接口，pipeline 直接调 `os.path.join` + `screenshot_capture` 写文件

需建立独立 `debugger/` 包，作为所有 debug 输出的统一入口，release 时整体删除不影响核心代码。

### 1.2 需求（issue #11）

1. 新建独立 debug 模块 `debugger/`，release 时可整体删除
2. debug 输出到 `debug/{视频名}/` 目录（与 output 平级）
3. 现阶段始终输出 debug（不设运行时开关，但 config 提供总开关用于 release）
4. **9 类 debug 产物**（详见 §三；第 9 类为运行日志归档，由用户后续追加）
5. 所有 debug 输出通过统一接口调用
6. 核心流水线只调 `debugger.save_xxx()`，不直接写文件
7. release 退出：删 `debugger/` 包 + config 关闭 debug 即可

### 1.3 与现有模块的关系

| 现有模块 | 关系 |
|---|---|
| `core/screenshot_capture.py` | 已实现 6/8 类截图，目录名迁移到 `06_知识点截图/`、`08_解题过程截图/`；7 类新增 |
| `core/content_extractor.py` | `_locate_segment` 注入 debugger，每次定位调 `save_locate_record` |
| `core/pipeline.py` | 接收 `Optional[Debugger]`，各 `_stage_xxx` 内调 `debugger.save_xxx` |
| `utils/file_utils.py` | 底层 `save_json`/`save_text` 由 debugger 内部复用，业务模块不再直接调 |
| `utils/logger.py` | 重构：setup_logger 不再每次创建独立 file handler；console 独立，file 走 root logger 全局共享。`set_log_file(path)` 全局切换路径；`debugger.attach_log_handler()` 把日志归档到 `debug/{视频名}/09_运行日志/pipeline.log` |

---

## 二、架构与依赖注入

### 2.1 包结构

```
videocontents/
├── core/           # 核心业务，不依赖 debugger
├── utils/          # 通用工具
├── adapters/       # 适配层
├── debugger/       # 新建：debug 输出统一入口
│   ├── __init__.py    # 导出 DebugSink
│   ├── sink.py        # DebugSink 类（实现所有 save_xxx 方法）
│   └── formatter.py   # 格式化辅助（字级时间戳转人读文本等）
└── config.py        # 加 DebugConfig
```

### 2.2 依赖注入方式（用户决策：依赖注入）

`core/pipeline.py` **不直接 import debugger 包**，避免 release 删除后 import 崩溃：

```python
# core/pipeline.py
from typing import Optional, Any

class Pipeline:
    def __init__(self, config: Config, mock_llm: bool = False, debugger: Optional[Any] = None):
        self.debugger = debugger  # duck typing，无 Protocol 约束
        # ... 其他模块初始化
    
    def _stage_xxx(self, context):
        # ... 业务逻辑
        if self.debugger:
            self.debugger.save_xxx(...)
```

调用方（main.py 或入口脚本）根据 config 决定是否创建 Debugger：

```python
# main.py（入口）
from config import Config
from core.pipeline import Pipeline

config = Config.load("config.yaml")
debugger = None
if config.debug.enabled:
    from debugger import DebugSink  # 仅在需要时 import
    debugger = DebugSink(config.paths.debug_dir)

pipeline = Pipeline(config, debugger=debugger)
```

**release 退出**：config.yaml 改 `debug.enabled: false`，可同时删除 `debugger/` 包，pipeline 不受影响（无 import 依赖）。

### 2.3 与适配层模式的关系

AGENTS.md "适配层模式"约束 ASR/OCR/LLM 隔离第三方库。debugger 不是第三方库适配，是自有产物输出，无需 Protocol 倒置。依赖注入 + duck typing 足够简洁。

---

## 三、9 类 debug 产物

### 3.1 目录结构

`debug/{视频名}/` 下 9 个子目录，按产物类型编号平铺：

```
debug/{视频名}/
├── 01_ASR原始逐字稿/
│   ├── asr_raw.json           # 机读：{text, char_timestamps}
│   └── asr_readable.txt       # 人读：[mm:ss.cc] 字 逐行
├── 02_纠错后全文/
│   ├── corrected.json         # 机读：{text, raw_align}
│   └── corrected.txt          # 人读：纯文本
├── 03_知识点文字段/
│   ├── 知识点01.txt            # 每段一个文件
│   ├── 知识点02.txt
│   └── ...
├── 04_题目文字段/
│   ├── 题目01.txt
│   ├── 题目02.txt
│   └── ...
├── 05_定位记录/
│   └── locate_log.jsonl       # jsonl：每行一条定位记录，便于追加
├── 06_知识点截图/
│   ├── 知识点01_t=05m23s.jpg
│   └── ...
├── 07_题目原题截图/
│   ├── 题目01_t=05m23s.jpg
│   └── ...
├── 08_解题过程截图/
│   ├── 题目01_步骤01_t=05m23s.jpg
│   └── ...
└── 09_运行日志/
    └── pipeline.log           # 全局运行日志（多模块共享一个 file handler）
```

### 3.2 产物格式说明

| 类别 | 文件 | 格式 | 内容 |
|---|---|---|---|
| 1 | 01/asr_raw.json | json | `RawTranscript.to_dict()` 全字段 |
| 1 | 01/asr_readable.txt | txt | 每字一行 `[mm:ss.cc] 字`，标点/无语音字标 `[--:--.--] (None)` |
| 2 | 02/corrected.json | json | `{text, raw_align: [int\|null]}` |
| 2 | 02/corrected.txt | txt | 纠错后纯文本（无时间戳） |
| 3 | 03/知识点NN.txt | txt | `# 知识点NN: {name}\n时间: [ss.cc - ss.cc]\n\n{segment_text}` |
| 4 | 04/题目NN.txt | txt | `# 题目NN\n时间: [ss.cc - ss.cc]\n\n{segment_text}` |
| 5 | 05/locate_log.jsonl | jsonl | 每行 `{"segment":"...","strategy":"exact\|medium\|low\|keyword","confidence":0.95,"start_time":12.3,"end_time":15.6,"start_idx":123,"end_idx":156,"keyword":"...""}` |
| 6 | 06/知识点截图 | jpg | 文件名同 issue #12，目录从 `06_截图/知识点/` 迁移 |
| 7 | 07/题目原题截图 | jpg | 文件名 `题目NN_t=NNmNNs.jpg`，**不做颜色过滤**（见 §六） |
| 8 | 08/解题过程截图 | jpg | 文件名同 issue #13，目录从 `06_截图/解题过程/` 迁移 |
| 9 | 09/pipeline.log | log | 全局运行日志（多模块共享），由 `debugger.attach_log_handler()` 切换 root logger 的 file handler 归档 |

### 3.3 格式决策依据（参考大多数软件方案）

- **1/2 类双格式**：json 给程序读（含完整字段），txt 给人读（便于人工查阅）。参考 logging 体系常见 dual output 模式。
- **3/4 类每段一文件**：每段独立 txt 便于人工对照某知识点/题目的原文片段，避免单文件过长难定位。
- **5 类 jsonl**：定位是高频小记录，jsonl 便于流式追加、流式读取（无需加载全文）。参考 log4j/logback 的 JSON logging layout。
- **9 类单文件 pipeline.log**：所有模块共享一个 file handler，便于按视频归档查阅。切换前日志写 `logs/{timestamp}.log`（release fallback），切换后写 `debug/{视频名}/09_运行日志/pipeline.log`。

---

## 四、DebugSink 接口

### 4.1 类定义

```python
# debugger/sink.py
class DebugSink:
    def __init__(self, debug_root: str, video_name: str = ""):
        self.root = os.path.join(debug_root, video_name)  # debug/{视频名}/
    
    def set_video_name(self, video_name: str) -> None:
        """pipeline 在 _stage_probe 后调用，确定视频名"""
        self.root = os.path.join(os.path.dirname(self.root), video_name)
    
    # === 8 类产物 save 方法 ===
    def save_asr_raw(self, transcript: RawTranscript) -> None:
        """1. ASR 原始逐字稿"""
    
    def save_corrected_text(self, aligned: AlignedTranscript) -> None:
        """2. 纠错后全文"""
    
    def save_knowledge_segments(self, kps: List[KnowledgePoint]) -> None:
        """3. 知识点文字段（每段一文件）"""
    
    def save_problem_segments(self, problems: List[Problem]) -> None:
        """4. 题目文字段（每段一文件，用 asr_question_text）"""
    
    def save_locate_record(self, segment_text: str, strategy: str, confidence: float,
                            start_time: float, end_time: float,
                            start_idx: int, end_idx: int,
                            keyword: str = "") -> None:
        """5. 定位记录（追加到 jsonl）"""
    
    def save_screenshot(self, category: str, index: int, src_path: str,
                         timestamp: float, step: int = 0) -> str:
        """6/7/8 类截图：复制 src 到 debug 目录
        category: 'knowledge' / 'question' / 'solution'
        返回 debug 中的目标路径"""

    def attach_log_handler(self) -> bool:
        """9. 运行日志归档：把全局 file handler 切换到 debug/{视频名}/09_运行日志/pipeline.log
        必须在 set_video_name() 之后调用；调用后所有 setup_logger 创建的 logger 自动跟随"""

    # === 内部辅助 ===
    def _save_text(self, rel_path: str, content: str) -> None: ...
    def _save_json(self, rel_path: str, data: Any) -> None: ...
    def _append_jsonl(self, rel_path: str, record: dict) -> None: ...
```

### 4.2 调用点映射

| pipeline 阶段 | debugger 调用 |
|---|---|
| `_stage_probe` | `debugger.set_video_name(video_name)` + `debugger.attach_log_handler()`（第 9 类：运行日志归档） |
| `_stage_asr` | `debugger.save_asr_raw(context.asr_results)` |
| `_stage_correct_and_extract` | `debugger.save_corrected_text(aligned)` + `save_knowledge_segments(kps)` + `save_problem_segments(problems)`；`content_extractor._locate_segment` 内部调 `save_locate_record` |
| `_stage_capture_screenshots` | screenshot_capture 直接写到 `debug/{视频名}/06/07/08/`（debugger 不参与截图写入，仅约束目录） |
| `_stage_summarize_knowledge` | （可选）`debugger.save_knowledge_segments` 覆盖一次（含 content/supplement） |
| `_stage_summarize_solution` | （可选）`debugger.save_problem_segments` 覆盖一次（含 solution_steps） |

### 4.3 content_extractor 注入 debugger

`_locate_segment` 内每次定位（无论成功失败）都调 `debugger.save_locate_record`：

```python
# core/content_extractor.py
class ContentExtractor:
    def __init__(self, llm: LLMGenerator, debugger: Optional[Any] = None):
        self.llm = llm
        self.debugger = debugger
    
    def _locate_segment(self, segment_text, aligned, search_start=0):
        # ... 现有定位逻辑
        if self.debugger:
            self.debugger.save_locate_record(
                segment_text=segment_text, strategy=strategy,
                confidence=confidence, start_time=start_time, end_time=end_time,
                start_idx=start_idx, end_idx=end_idx, keyword=keyword_used,
            )
        return start_time, end_time, start_idx, end_idx
```

---

## 五、pipeline 集成改动

### 5.1 改动点清单

| 文件 | 改动 |
|---|---|
| `core/pipeline.py` | `__init__` 加 `debugger` 参数；各 `_stage_xxx` 调 `debugger.save_xxx`；`_stage_capture_screenshots` 路径改用 debug 子目录 |
| `core/content_extractor.py` | `__init__` 加 `debugger` 参数；`_locate_segment` 内调 `save_locate_record` |
| `core/screenshot_capture.py` | 不改实现，只改调用方传入的 `output_dir`（pipeline 控制） |
| `core/output_assembler.py` | 不改，`result.screenshot_paths` 现在指向 debug/07/，仍能复制到 output |
| `config.py` | 加 `DebugConfig`（含 `enabled: bool = True`），`PathsConfig.debug_dir` 已存在 |
| `config.example.yaml` | 加 `debug.enabled: true` |
| `debugger/__init__.py` | 新建，导出 DebugSink |
| `debugger/sink.py` | 新建，实现 DebugSink 类 |
| `debugger/formatter.py` | 新建，格式化辅助（asr_readable 等） |

### 5.2 截图目录迁移

| 现状 | 新设计 |
|---|---|
| `debug/{视频名}/06_截图/知识点/知识点01_t=NNmNNs.jpg` | `debug/{视频名}/06_知识点截图/知识点01_t=NNmNNs.jpg` |
| `debug/{视频名}/06_截图/解题过程/题目01_步骤01_t=NNmNNs.jpg` | `debug/{视频名}/08_解题过程截图/题目01_步骤01_t=NNmNNs.jpg` |
| `output/{视频名}/截图/题目01.jpg`（颜色过滤后） | `debug/{视频名}/07_题目原题截图/题目01_t=NNmNNs.jpg`（不颜色过滤）；output_assembler 复制到 `output/{视频名}/截图/题目01.jpg` |

### 5.3 pipeline 调用顺序

```python
# core/pipeline.py
def _stage_capture_screenshots(self, context: PipelineContext):
    video_name = os.path.splitext(os.path.basename(context.video_path))[0]
    
    # 题目原题截图：写到 debug/07_题目原题截图/，不颜色过滤
    q_dir = os.path.join(self.config.paths.debug_dir, video_name, "07_题目原题截图")
    context.screenshot_paths = self.screenshot_capture.capture_screenshots(
        context.video_path, context.problems, q_dir, enable_color_filter=False
    )
    
    # 知识点截图：debug/06_知识点截图/
    if context.knowledge_points:
        k_dir = os.path.join(self.config.paths.debug_dir, video_name, "06_知识点截图")
        context.knowledge_screenshot_paths = self.screenshot_capture.capture_knowledge_screenshots(
            context.video_path, context.knowledge_points, k_dir
        )
    
    # 解题过程截图：debug/08_解题过程截图/
    if context.problems and any(p.solution_steps for p in context.problems):
        s_dir = os.path.join(self.config.paths.debug_dir, video_name, "08_解题过程截图")
        context.solution_screenshot_paths = self.screenshot_capture.capture_solution_screenshots(
            context.video_path, context.problems, s_dir
        )
```

---

## 六、颜色过滤处理（冲突 2）

### 6.1 用户决策

> 冲突 2：颜色过滤已经不存在了

### 6.2 现状与处理

- `core/screenshot_capture.py` 的 `capture_screenshots` 默认 `enable_color_filter=True`，调用 `remove_color_keep_black`
- `config.example.yaml` 默认 `enable_color_filter: false`（P2 可选，不启用）
- pipeline 调用 `capture_screenshots` 时未传该参数，用默认 True → **bug**

### 6.3 本设计处理

- 题目原题截图统一写到 `debug/{视频名}/07_题目原题截图/`，**不做颜色过滤**（`enable_color_filter=False`）
- `output_assembler` 从 `debug/07/` 复制到 `output/{视频名}/截图/` 给学生看
- 颜色过滤逻辑（`remove_color_keep_black`）保留为 P2 升级空间，但默认关闭，不在本设计中调用
- 不删除 `enable_color_filter` 参数（保留 P2 升级空间）

---

## 七、config 扩展

### 7.1 新增 DebugConfig

```python
# config.py
@dataclass
class DebugConfig:
    enabled: bool = True       # release 时改 false，可同时删 debugger/ 包
    max_size_gb: float = 50   # debug 目录大小上限（防止累积）
    save_intermediate: bool = True  # 是否保存中间产物（False 时只保存最终产物）
```

`Config` 主类加 `debug: DebugConfig = field(default_factory=DebugConfig)`。

### 7.2 config.example.yaml

```yaml
debug:
  enabled: true
  max_size_gb: 50
  save_intermediate: true
```

---

## 八、已定案决策（用户已确认）

| 项 | 决策 | 来源 |
|---|---|---|
| 1. 目录编号 | 01-08 平铺，6/7/8 是三类截图 | 冲突 1 |
| 2. 颜色过滤 | 不做（debug 中题目原题截图不颜色过滤） | 冲突 2 |
| 3. 依赖注入方式 | `Pipeline.__init__` 接收 `Optional[Any]`，duck typing，无 Protocol | 冲突 3 |
| 4. debugger 覆盖范围 | 所有需要 debug 的模块（content_extractor / pipeline / screenshot_capture）都注入同一 debugger | 冲突 4 |
| 5. 1/2 类格式 | json + txt 双格式 | 暂未明确点 |
| 6. 3/4 类格式 | 每段一个 txt 文件 | 暂未明确点 |
| 7. 5 类格式 | jsonl（流式追加） | 暂未明确点 |
| 8. release 退出 | config.debug.enabled=false + 删 debugger/ 包 | 冲突 3 |

---

## 九、待实现任务清单

1. `debugger/__init__.py` + `sink.py` + `formatter.py`
2. `config.py` 加 `DebugConfig`；`config.example.yaml` 加 debug 段
3. `core/pipeline.py` 加 `debugger` 参数 + 各 stage 调用
4. `core/content_extractor.py` 加 `debugger` 参数 + `_locate_segment` 调用
5. `core/pipeline.py` `_stage_capture_screenshots` 截图目录迁移到 06/07/08
6. 测试：`tests/test_debugger.py` 覆盖 8 类产物 + 依赖注入 + release 退出场景
7. 提交 + 关闭 issue #11
