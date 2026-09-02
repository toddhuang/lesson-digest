## 变更描述
简要描述本次 PR 做了什么改动。

## 变更类型
- [ ] 新功能 (feature)
- [ ] Bug 修复 (fix)
- [ ] 重构 (refactor)
- [ ] 文档 (docs)
- [ ] 测试 (test)
- [ ] 配置/工具 (chore)

## 涉及模块
`[probe | asr | ocr | correct_and_extract | summarize_solution | summarize_knowledge | merge_text | capture_screenshots | generate_mindmap | assemble_output | debugger | config | 其他]`

## 检查清单
- [ ] 代码遵循 SOLID 原则，一个文件一个 class
- [ ] 无硬编码（model 名称、endpoint、阈值等从 config 读取）
- [ ] 无裸 `try except Exception`（按异常类型分别处理）
- [ ] 无 API Key 等敏感信息
- [ ] `git diff --cached` 检查无明文密钥
- [ ] 新增/修改功能有对应测试
- [ ] `pytest tests/ -q` 全部通过

## 测试结果
```
粘贴 pytest 输出
```
