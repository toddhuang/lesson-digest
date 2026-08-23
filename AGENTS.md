# AGENTS.md — 项目开发约定

> 本文件记录项目级别的开发偏好和约定，所有 AI agent 会话必须遵守。

## 项目概述

教学视频内容提取与总结工具。双机分离架构：Windows 3080（主程序 + FunASR + PaddleOCR）+ Ubuntu 4090（vLLM 推理）。

## 画图约定

**所有架构图、流程图、关系图统一使用 Mermaid。**

- 直接在 Markdown 中嵌入 ` ```mermaid ` 代码块
- GitHub 原生渲染，无需生成图片文件
- 禁止使用 matplotlib / Python 手动画架构图
- 禁止将架构图存为 PNG/JPG 图片文件嵌入文档
- 模块注重：功能、输入、输出，不过度描述细节

## 文档约定

- 文档存放于 `Document/` 目录
- 环境文档存放于 `Document/环境文档/`
- 开发文档存放于 `Document/开发文档/`
- 文档中禁止出现本地绝对路径（如 `C:\work\...`、`/home/用户名/...`），使用 `<项目根目录>`、`~/` 或 `/home/你的用户名/` 等占位符
- 文档中禁止出现真实局域网 IP，使用 `192.168.x.x` 模板

## 技术栈

- Python 3.11
- FunASR（语音识别）
- PaddleOCR（文字识别）
- vLLM（大模型推理，Qwen3.6-27B AWQ）
- ffmpeg（音视频处理）
