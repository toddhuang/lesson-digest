# 贡献指南

感谢你对 lesson-digest 的关注！本文档说明如何参与项目贡献。

## 开发环境

### 依赖安装

```bash
git clone https://github.com/toddhuang/lesson-digest.git
cd lesson-digest
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
```

### 配置

复制模板配置文件并填入你的 API Key：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入豆包/DeepSeek API Key
```

> **安全须知**：`config.yaml` 已在 `.gitignore` 中，切勿提交。提交前执行 `git diff --cached` 确认无明文密钥。

### GPU 环境（可选）

如需本地运行 ASR（FunASR）和 OCR（PaddleOCR），需准备 CUDA 12.x + cuDNN 9 的 GPU 环境。详见 `Document/环境文档/开发环境验证文档.md`。

## 项目结构

```
lesson-digest/
├── core/              # 核心业务模块（pipeline 12 阶段）
├── adapters/          # 适配层（ASR/OCR/LLM 第三方库隔离）
├── debugger/          # 调试模块（9 类 debug 产物）
├── utils/             # 工具函数
├── tests/             # 单元测试（pytest）
├── scripts/           # 一次性验证脚本
├── config.example.yaml  # 配置模板
├── AGENTS.md          # 项目开发约定（必读）
└── Document/          # 设计文档、接口文档、调研报告
```

## 开发约定

项目遵循 [AGENTS.md](AGENTS.md) 中的约定，核心要点：

### 设计原则
- **SOLID 原则**：单一职责、开闭、里氏替换、接口隔离、依赖倒置
- **一个文件一个 class**：interface 和具体实现分离到不同文件
- **适配层模式**：ASR/OCR/LLM 均通过适配层隔离第三方库

### 编码风格
- **禁止硬编码**：model 名称、endpoint、超时时间、阈值等从 `config.yaml` 读取
- **禁止裸 `try except Exception`**：按异常类型分别处理（参数错误不重试、限流指数退避、超时改流式等）
- **临时验证脚本放 `scripts/`**：不要放 `tests/`
- 脚本内用 `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))` 计算项目根

### 安全规范
- 绝对禁止将 API Key、密钥、token 提交到 git
- `config.example.yaml` 中 api_key 必须为空字符串
- 发现密钥已提交，立即通知维护者轮换密钥

## 提交规范

### Commit Message
```
<type>(<scope>): <subject>

<body>
```
- type：`feat | fix | refactor | docs | test | chore`
- scope：模块名（如 `pipeline`, `ocr`, `llm`, `debugger`）
- 示例：`fix(pipeline): capture_screenshots 判空 + 补 tenacity 依赖`

### 提交流程

1. Fork 仓库并创建分支：`git checkout -b feat/your-feature`
2. 编写代码，确保遵循开发约定
3. 运行测试：`pytest tests/ -q`
4. 提交 PR，填写 PR 模板中的检查清单

### Issue 规范

- **body 只写问题描述**：调研目标、背景、待回答的问题
- **结论用 comment 提交**：不要写在 body 中
- **关闭 Issue 时添加 closing comment**：说明完成原因、核心结论、产出物链接

## 测试

```bash
# 运行全部单元测试
pytest tests/ -q

# 运行指定模块测试
pytest tests/test_core/ -q

# 查看覆盖率
pytest tests/ --cov=. --cov-report=term-missing
```

## 文档

- 设计文档存放于 `Document/开发文档/`
- 环境文档存放于 `Document/环境文档/`
- 调研报告存放于 `Document/调研报告/`
- 文档中禁止出现本地绝对路径，使用 `<项目根目录>` 或 `~/` 占位
- 架构图统一使用 Mermaid，禁止存为图片文件
