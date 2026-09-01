# Unit Test 体系设计

> 版本：v1.0（待确认后定案）
> 日期：2026-09-01
> 状态：设计中，待用户确认 6 点决策后推进实现
> 所属模块：tests/（重新启用）+ .github/workflows/
> 关联 issue：#23 [F-004]
> 前置文档：11_debug模块设计.md（debugger 已有 scripts/test_debugger.py 验收脚本）

---

## 一、设计目标

### 1.1 解决的问题

现有代码无 unit test，重构时只能靠 scripts/ 下临时验证脚本手动跑：
- 无法持续防回归（改 pipeline 调度、改 content_extractor._locate_segment 等没有自动化验证）
- 没有测试覆盖报告，不知道哪些路径没测过
- 没有 CI，PR 合并前不跑测试

### 1.2 需求（issue #23）

1. 建立 unit_test 体系，覆盖核心模块业务逻辑
2. 与 `scripts/` 临时验证脚本区分（scripts/ 是一次性验证，tests/ 是持续回归）
3. 接入 CI（GitHub Actions）自动跑测试

### 1.3 现有可复用资源

| 资源 | 位置 | 用途 |
|---|---|---|
| MockLLMAdapter | [adapters/llm/mock.py](file:///c:/work/doubaotask/projects/videocontents/adapters/llm/mock.py) | 返回固定 LLMResponse，unit test 用作 LLM mock |
| MockASRAdapter | [adapters/asr/mock.py](file:///c:/work/doubaotask/projects/videocontents/adapters/asr/mock.py) | 返回预定义教学文本 + 字级时间戳 |
| MockOCRAdapter | [adapters/ocr/mock.py](file:///c:/work/doubaotask/projects/videocontents/adapters/ocr/mock.py) | 按帧名返回模拟 OCRResult（课件/题目/公式） |
| scripts/test_debugger.py | issue #11 验收 | 8 用例，可改造为 pytest |
| scripts/test_enrich_knowledge.py | issue #9 验收 | 5 用例，可改造为 pytest |

---

## 二、6 点关键决策（待用户确认）

| # | 决策点 | 我的选择 | 备选 | 理由 |
|---|---|---|---|---|
| 1 | **测试框架** | pytest | unittest / nose2 | 主流选择，fixture 系统强大，断言简洁，生态最丰富 |
| 2 | **测试目录** | 重新启用 `tests/` | 新建 `test/` | tests/ 是 Python 社区标准命名；现有 tests/ 下有素材（test.mp4 等）可一并管理 |
| 3 | **.gitignore 调整** | `tests/*` 改为 `tests/*` + `!tests/*.py` + `!tests/conftest.py` | 全部入库 / 素材移到 tests/fixtures/ | .py 入库，素材仍 ignore（避免大文件入库） |
| 4 | **覆盖范围分层** | L1（必测）+ L2（应测）跳过 L3 | 全覆盖 / 只测 L1 | L1=utils 纯函数+debugger+adapters/mock；L2=core 业务模块（content_extractor 等）；L3=pipeline 集成（依赖太多，留 P2） |
| 5 | **mock 策略** | 复用现有 adapters/ 下 mock.py + 新建 conftest.py 提供 fixture | 全部新建 mock | 现有 mock 已实现且验证过，避免重复劳动；conftest.py 统一 fixture |
| 6 | **CI 集成** | `.github/workflows/test.yml`，push/PR 时跑 pytest + coverage 报告 | 不接 CI / 只跑 pytest 不测 coverage | push 触发，PR 必须 pass 才能 merge；coverage 防止测试缩水 |

---

## 三、测试目录结构

### 3.1 启用 tests/ 作 unit test 目录

```
tests/
├── conftest.py              # 全局 fixture（mock_config、mock_aligned、mock_kps 等）
├── test_utils/              # L1: utils/ 纯函数测试
│   ├── test_timestamp.py    # format_timestamp 各格式
│   ├── test_file_utils.py   # save_json/save_text/ensure_dir
│   ├── test_models.py       # dataclass 序列化/to_dict
│   └── test_token_counter.py
├── test_debugger/           # L1: debugger/ 测试（从 scripts/test_debugger.py 改造）
│   ├── test_sink.py         # 8 类产物 save 方法
│   ├── test_formatter.py    # 格式化辅助
│   └── test_logger.py        # set_log_file 切换
├── test_adapters/           # L1: adapters/ mock 测试
│   ├── test_llm_mock.py     # MockLLMAdapter 返回结构
│   ├── test_asr_mock.py
│   └── test_ocr_mock.py
├── test_core/               # L2: core/ 业务模块测试
│   ├── test_content_extractor.py  # _locate_segment 4 策略 + _parse_response
│   ├── test_knowledge_extractor.py # enrich_knowledge（从 scripts/test_enrich_knowledge.py 改造）
│   ├── test_problem_extractor.py  # enrich_solution
│   ├── test_frame_dedup.py        # dHash 去重逻辑
│   └── test_frame_extractor.py
└── fixtures/                 # 测试素材（.mp4 .wav .png 等，可选 P2）
```

### 3.2 .gitignore 调整

```diff
 # 测试文件
 test_*.mp4
-tests/*
+tests/*
+!tests/*.py
+!tests/conftest.py
+!tests/*/
+!tests/**/*.py
```

含义：
- `tests/*`：默认 ignore tests/ 下所有文件
- `!tests/*.py`：保留根目录 .py
- `!tests/conftest.py`：保留 conftest.py
- `!tests/*/` + `!tests/**/*.py`：保留子目录及其下 .py
- 素材文件（.mp4 .wav .png .txt）仍 ignore

---

## 四、conftest.py 设计

### 4.1 全局 fixture

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from config import Config
from utils.models import (
    RawTranscript, CharTime, AlignedTranscript,
    KnowledgePoint, Problem, SolutionStep,
)
from adapters.llm.mock import MockLLMAdapter
from core.llm.protocol import LLMGenerator


@pytest.fixture
def mock_config():
    """默认 Config 实例（不读 config.yaml）"""
    return Config()


@pytest.fixture
def tmp_debug_dir():
    """临时 debug 目录，测试结束自动清理"""
    tmpdir = tempfile.mkdtemp(prefix="test_debug_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def raw_transcript():
    """ASR 原始逐字稿"""
    text = "今天讲二次函数。"
    char_ts = [CharTime(start_ms=i * 1000, end_ms=i * 1000 + 1000) for i in range(len(text))]
    return RawTranscript(text=text, char_timestamps=char_ts)


@pytest.fixture
def aligned_transcript(raw_transcript):
    """纠错后对齐文本"""
    return AlignedTranscript(
        text=raw_transcript.text,
        raw_align=list(range(len(raw_transcript.text))),
        raw=raw_transcript,
    )


@pytest.fixture
def mock_llm_generator():
    """MockLLMAdapter 包装的 LLMGenerator"""
    adapter = MockLLMAdapter()
    return LLMGenerator(adapter)


@pytest.fixture
def sample_knowledge_points():
    """2 个知识点"""
    return [
        KnowledgePoint(index=1, name="二次函数定义", start_time=0.0, end_time=5.0,
                       confidence=0.95, content="函数 $f(x)=ax^2+bx+c$", supplement="顶点公式"),
        KnowledgePoint(index=2, name="图像性质", start_time=5.0, end_time=12.0,
                       confidence=0.92, content="开口由 $a$ 决定", supplement=""),
    ]


@pytest.fixture
def sample_problems():
    """1 道题含 2 个解题步骤"""
    return [
        Problem(index=1, start_time=15.0, end_time=30.0,
                question_text="解方程 x^2-5x+6=0",
                asr_question_text="解方程 x的平方减5x加6等于0",
                solution_steps=[
                    SolutionStep(step_number=1, content="因式分解", start_time=18.0, end_time=22.0),
                    SolutionStep(step_number=2, content="令每个因式为0", start_time=22.0, end_time=28.0),
                ],
                confidence=0.93),
    ]
```

### 4.2 fixture 设计原则

- **业务模块 fixture 不依赖外部资源**（无文件 IO、无网络）
- **mock 适配器优先**：LLM/ASR/OCR 都用 mock.py
- **临时目录 fixture 自动清理**：tmp_debug_dir 等 yield + shutil.rmtree

---

## 五、测试改造计划

### 5.1 从 scripts/ 改造为 pytest

| scripts/ 原文件 | 改造为 | 改动 |
|---|---|---|
| [scripts/test_debugger.py](file:///c:/work/doubaotask/projects/videocontents/scripts/test_debugger.py) | tests/test_debugger/test_sink.py + test_formatter.py + test_logger.py | 拆分；test_xxx 函数加 pytest 标记；fixture 替换 setup |
| [scripts/test_enrich_knowledge.py](file:///c:/work/doubaotask/projects/videocontents/scripts/test_enrich_knowledge.py) | tests/test_core/test_knowledge_extractor.py | 同上 |

### 5.2 scripts/ 原文件保留

- AGENTS.md 约定"临时验证脚本放 scripts/"——这两个文件是 issue 验收脚本，性质上属临时验证
- 改造为 pytest 后，原 scripts/ 文件**保留**作为 issue 验收证据（git log 可追溯）
- 或：删除 scripts/ 原文件，由 tests/ 替代（避免重复）

**推荐**：删除 scripts/ 原文件，由 tests/ 替代，避免代码重复维护。

---

## 六、CI 集成

### 6.1 GitHub Actions 配置

`.github/workflows/test.yml`：

```yaml
name: unit-test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov
          # 安装项目依赖（按 requirements.txt）
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml --cov-report=term
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-report
          path: coverage.xml
```

### 6.2 CI 注意事项

- **不依赖外部服务**：LLM/ASR/OCR 全用 mock，CI 环境无 API key、无 GPU
- **不读 config.yaml**：用 Config() 默认实例
- **不写 logs/debug**：临时目录 fixture 自动清理

---

## 七、待实现任务清单

1. `.gitignore` 调整 tests/* + 否定模式
2. `tests/conftest.py` 全局 fixture
3. `tests/test_utils/` L1 测试（timestamp、file_utils、models 等）
4. `tests/test_debugger/` L1 测试（从 scripts/test_debugger.py 改造）
5. `tests/test_adapters/` L1 测试（mock 适配器）
6. `tests/test_core/` L2 测试（content_extractor、knowledge_extractor、problem_extractor 等）
7. `.github/workflows/test.yml` CI 配置
8. `requirements.txt` 加 pytest + pytest-cov
9. 删除 scripts/test_debugger.py 和 scripts/test_enrich_knowledge.py（被 tests/ 替代）
10. 提交 + 关闭 issue #23

---

## 八、已定案决策（用户确认后定案）

（待用户确认 §二 的 6 点决策后填入）
