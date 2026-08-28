# R-007 印刷体/手写体分开识别 OCR 方案调研报告

> **GitHub Issue**: [#17](https://github.com/toddhuang/videocontents/issues/17)
> **调研日期**: 2026-08-27
> **调研状态**: 文档调研完成，待实际验证
> **约束**: 仅考虑本地部署方案（Windows + RTX 3080 10GB），不考虑云端 OCR API

---

## 调研目标

验证是否有**本地部署**的 OCR 方案支持印刷体和手写体分开识别（输出文字类型标注），同时满足教学视频场景的数学公式识别需求。

---

## 总结论

**目前没有任何现成的本地 OCR 引擎原生支持印刷体/手写体分类标注。** 所有本地方案（PaddleOCR、Surya、EasyOCR、TrOCR 等）都只能识别文字内容，不输出"这段是印刷体/那段是手写体"的类型标签。

要实现分类，有两条可行路径：

1. **PP-OCRv5 文字检测 + 自研二分类器**：用 PP-OCRv5 检测和识别所有文字（对中文手写检测精度 0.803），再对每个检测到的文字区域训练一个轻量 CNN 分类器判断印刷/手写。这是业界已验证的两步法架构。
2. **本地 VLM（Qwen2.5-VL 3B）**：用提示词让视觉大模型一次性完成文字识别 + 印刷/手写分类 + 公式识别。3B 模型可在 10GB 显存运行，但单帧速度较慢（约 10-35 秒）。

公式识别方面，PaddleOCR-VL 原生输出 LaTeX，是目前本地公式识别的最佳选择。

**建议下一步**：用 `tests/` 目录的教学截图实测 PP-OCRv5 的检测/识别效果和 Qwen2.5-VL 3B 的分类能力，再决定最终架构。

---

## 本地方案详细分析

### 方案 1：PP-OCRv5（文字检测 + 识别）+ 二分类器

**官方文档**:
- PP-OCRv5 算法介绍: http://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html
- OCR 产线使用教程: https://www.paddleocr.ai/v3.0.1/version3.x/pipeline_usage/OCR.html
- GitHub: https://github.com/PaddlePaddle/PaddleOCR
- 许可证: Apache 2.0

**PP-OCRv5 能力**（官方数据）:

| 指标 | 手写中文 | 印刷中文 | 手写英文 | 印刷英文 |
|---|---|---|---|---|
| 检测精度 (server_det) | 0.803 | 0.945 | 0.841 | 0.917 |
| 识别精度 (server_rec) | 0.5807 | 0.9013 | 0.5806 | 0.8679 |

- 单模型同时支持印刷体和手写体的检测与识别（不需要切换模型）
- 支持简体中文、繁体中文、英文、日文、拼音
- GPU 推理速度：T4 上约 0.46 秒/帧，A100 上约 0.25 秒/帧
- 峰值显存：约 4GB（server 模型）
- Windows 兼容，需 PaddlePaddle 3.x（从当前 2.8.1 升级）

**关键限制**：输出中 `text_type` 字段当前固定为 `"general"`，**不标注印刷/手写类型**。PP-OCRv5 能检测和识别手写文字，但不告诉你哪段是手写的。

**二分类器方案**：

两步法架构（先检测所有文字区域，再逐区域分类）在业界已有实践（如处方手写/印刷分离项目）。分类器可选：

| 分类器方案 | 说明 | 优劣 |
|---|---|---|
| 轻量 CNN（MobileNet/ResNet） | 对 PP-OCRv5 裁剪出的文字行图片做二分类 | 速度快（<10ms/区域），需训练数据 |
| 图像特征启发式 | 笔画宽度变化、颜色分布、边缘锐度等 | 无需训练，但黑板场景鲁棒性未验证 |
| 本地小 VLM 做分类 | 用 Qwen2.5-VL 3B 对每个文字区域截图分类 | 无需训练，但速度慢 |

训练数据来源：RVL-CDIP（文档印刷/手写分类数据集）、合成数据（多种字体渲染印刷中文 + CASIA-HWDB 手写中文）、项目测试图片手动标注。

**优势**:
- PP-OCRv5 中文检测识别能力成熟，速度快
- 分类器轻量，不影响整体处理速度
- 模块化设计，分类器可独立迭代

**劣势**:
- 需要升级到 PaddlePaddle 3.x
- 手写中文识别精度 0.58 偏低（潦草字可能出错）
- 二分类器需要训练和验证
- 公式识别需额外组件（PaddleOCR-VL 或 pix2tex）

---

### 方案 2：PaddleOCR-VL（多模态文档解析模型）

**官方文档**: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html
**论文**: https://arxiv.org/pdf/2510.14528v2
**许可证**: Apache 2.0

**核心能力**:

| 能力 | 支持情况 |
|---|---|
| 印刷/手写分类 | ❌ 不支持。版面标签为 text/table/formula/image/chart 等，无 handwriting 标签 |
| 手写文字识别 | ✅ 可识别手写内容，但不标注类型 |
| 数学公式 | ✅ 原生输出 LaTeX（inline/display） |
| 中文支持 | ✅ 109 种语言 |
| 模型大小 | 0.9B 参数（PaddleOCR-VL-0.9B） |
| Windows | ✅ 支持 x64 CPU / NVIDIA GPU |
| 推理速度 | 未在 3080 上实测，VLM 推理比传统 OCR 慢 |

**重要澄清**：

多篇 CSDN 博客声称 PaddleOCR-VL 输出 `text_printed` / `text_handwriting` 标签。经查阅官方文档输出格式定义，`parsing_res_list` 中 `block_label` 取值为 `doc_title`、`text`、`paragraph_title`、`table`、`formula`、`image`、`chart`、`seal` 等版面元素类型，**不存在印刷/手写分类标签**。CSDN 内容来自第三方 WebUI 封装或为不实信息。

**优势**:
- 公式 LaTeX 输出是原生能力，本地免费
- 版面分析能力强（阅读顺序、表格、图表）
- 0.9B 模型相对轻量

**劣势**:
- 不满足印刷/手写分类核心需求
- 需 PaddlePaddle 3.x 大版本升级
- VLM 推理速度比 PP-OCRv5 慢
- 黑板场景未验证

---

### 方案 3：Qwen2.5-VL（本地视觉大模型）

**官方仓库**: https://github.com/QwenLM/Qwen2.5-VL
**许可证**: Apache 2.0
**OCRBench**: 7B 版本 864 分（超过 GPT-4o-mini 的 785）

**核心能力**:

| 能力 | 支持情况 |
|---|---|
| 印刷/手写分类 | ✅ 通过提示词实现，可要求逐区域标注 |
| 手写文字识别 | ✅ 官方称支持手写文档解析 |
| 数学公式 | ✅ 可输出 LaTeX |
| 中文支持 | ✅ 原生中文 |
| 文字坐标 | ⚠️ 支持目标 grounding（输出 bounding box），但精度未验证 |
| 模型大小 | 3B（约 6GB 显存）/ 7B（约 10GB+ 显存） |

**显存与速度**（社区基准）:
- Qwen2.5-VL-3B-Instruct：峰值 6.0-6.5GB 显存，RTX 3080 10GB 可运行
- Qwen2.5-VL-7B：约 35 秒/页（文档解析），3B 更快但精度降低
- 支持 Flash Attention 2 加速

**优势**:
- 一个模型同时解决识别 + 分类 + 公式，架构最简
- 无需训练分类器
- 对黑板/彩色粉笔等非文档场景的鲁棒性可能优于传统 OCR（VLM 理解语义）
- Apache 2.0，可商用

**劣势**:
- 单帧速度慢（3B 估计 10-15 秒，7B 约 35 秒），数百帧处理耗时较长
- 文字识别精度可能不如专用 OCR（尤其长文本逐字准确率）
- 存在幻觉风险
- 3B 模型的分类准确率未验证

---

### 方案 4：TrOCR（微软手写识别模型）

**官方仓库**: https://github.com/microsoft/unilm/tree/master/trocr
**许可证**: MIT

- 基于 Transformer 的手写文字识别模型，IAM 数据集 CER 2.89%
- **仅支持英文手写**，不支持中文
- 只做识别（不做检测），需要外接文字检测模块
- 不支持印刷/手写分类

**结论**: 不支持中文，排除。

---

### 方案 5：Surya OCR

**GitHub**: https://github.com/datalab-to/surya
**Stars**: 23k+
**许可证**: 代码 GPL，权重 AI Pubs Open Rail-M（商业使用受限）

官方 README 明确说明："It is for printed text, not handwriting (though it may work on some handwriting)"。

**结论**: 明确不支持手写体，且 GPL 许可证对商业项目不友好，排除。

---

### 方案 6：EasyOCR

**GitHub**: https://github.com/JaidedAI/EasyOCR
**许可证**: Apache 2.0

- 支持 80+ 语言含中文，但手写体识别能力弱
- 不支持印刷/手写分类
- 社区维护，近年更新缓慢

**结论**: 不满足分类需求，手写能力不如 PP-OCRv5。

---

### 方案 7：专用公式识别（pix2tex）

**GitHub**: https://github.com/lukas-blecher/LaTeX-OCR
**许可证**: MIT

- 公式图片 → LaTeX，基于 ViT
- 仅识别公式，不识别通用文字
- 可作为公式识别的补充组件，与 PP-OCRv5 配合使用

**结论**: 可作为方案 1 的公式识别补充，不独立解决核心问题。

---

### 方案 8：Florence-2（微软视觉基础模型）

**官方仓库**: https://huggingface.co/microsoft/Florence-2-base-ft
**模型大小**: 0.2B（极小）
**许可证**: MIT

- 支持 OCR、OCR_WITH_REGION（文字+坐标）、phrase grounding（文本描述定位区域）、开放词汇检测
- 理论上可用 phrase grounding 检测"handwritten text"区域
- **中文 OCR 能力未验证**，模型主要面向英文场景

**结论**: 模型极小速度快，但中文能力存疑，可作为备选实验对象。

---

## 方案对比总表

| 维度 | PP-OCRv5 + 分类器 | PaddleOCR-VL | Qwen2.5-VL 3B | TrOCR | Surya | pix2tex |
|---|---|---|---|---|---|---|
| 印刷/手写分类 | 需自研分类器 | ❌ | ✅ 提示词 | ❌ | ❌ | ❌ |
| 中文印刷识别 | ✅ 0.90 | ✅ | ✅ | ❌ 仅英文 | ⚠️ | ❌ |
| 中文手写识别 | ⚠️ 0.58 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 数学公式 LaTeX | 需额外组件 | ✅ 原生 | ✅ | ❌ | ❌ | ✅ |
| 文字坐标 | ✅ | ✅ | ⚠️ grounding | ❌ | ✅ | ❌ |
| 单帧速度 | ~0.5s（GPU） | 未实测 | ~10-35s | - | - | - |
| 显存需求 | ~4GB | 未实测 | ~6.5GB | - | - | - |
| Windows | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 许可证 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT | GPL | MIT |
| 是否需训练 | 分类器需训练 | 否 | 否 | - | - | - |

---

## 推荐架构方向（待实测验证）

### 方向 A：PP-OCRv5 + 轻量分类器 + PaddleOCR-VL 公式

```
关键帧
  → PP-OCRv5 检测所有文字区域（坐标 + 识别文字）
  → 对每个文字区域截图做二分类（印刷/手写）
  → 公式区域送 PaddleOCR-VL 或 pix2tex 识别 LaTeX
  → 输出：文字内容 + 类型标签 + 坐标 + 公式 LaTeX
```

- 优点：速度快，PP-OCRv5 检测成熟
- 风险：手写识别精度 0.58 偏低；分类器需训练

### 方向 B：Qwen2.5-VL 3B 一体化

```
关键帧
  → Qwen2.5-VL 3B（提示词：识别所有文字，标注印刷/手写，公式输出 LaTeX，给出坐标）
  → 输出：结构化 JSON
```

- 优点：架构最简，无需训练，语义理解强
- 风险：速度慢（数百帧可能数小时），精度和幻觉需实测

### 方向 C：混合架构

```
关键帧
  → PP-OCRv5 快速检测文字区域 + 识别
  → 对置信度低的区域（疑似手写/公式）送 Qwen2.5-VL 3B 精细识别和分类
  → 公式区域送 PaddleOCR-VL 输出 LaTeX
```

- 优点：兼顾速度和精度
- 风险：架构复杂度高

---

## 待实际验证项

1. **PP-OCRv5**：在 `tests/` 教学截图上的检测/识别效果，特别是手写中文和数学公式
2. **PP-OCRv5 升级**：PaddlePaddle 3.x 在 Windows + RTX 3080 + CUDA 12.9 上的安装兼容性
3. **Qwen2.5-VL 3B**：在 RTX 3080 10GB 上的显存占用、推理速度、印刷/手写分类准确率
4. **PaddleOCR-VL**：在 3080 上的推理速度和黑板场景公式识别效果
5. **二分类器可行性**：文字区域图像特征（颜色、笔画）对印刷/手写的区分度

---

## 调研教训

1. **CSDN 博客信息不可信**：多篇 CSDN 文章声称 PaddleOCR-VL 输出 `text_printed` / `text_handwriting` 标签，查阅官方文档后确认不存在。再次印证"官方文档优先"原则。
2. **"能识别手写"≠"能区分手写"**：PP-OCRv5 和 PaddleOCR-VL 都能识别手写文字内容，但不标注文字类型。这是两个不同能力。
3. **本地 OCR 无现成分类方案**：与 Azure DI 等云服务不同，开源 OCR 引擎普遍不提供印刷/手写分类，需要自行构建分类器或使用 VLM。
4. **PP-OCRv5 手写中文识别精度 0.58**：这意味着潦草手写约 40% 字符会识别错误，下游需要 ASR 纠错和 LLM 辅助修正。
