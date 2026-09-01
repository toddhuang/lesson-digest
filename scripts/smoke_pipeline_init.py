"""Pipeline 初始化冒烟测试：验证 config 和 adapter 工厂能跑通，不执行实际阶段。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils.gpu_env import setup_gpu_path, preload_torch
setup_gpu_path()
preload_torch()

from config import load_config
from core.pipeline import Pipeline, STAGES

print("=== 加载 config ===")
config = load_config("config.yaml")
print(f"  asr.adapter_type: {config.asr.adapter_type}")
print(f"  ocr.text_adapter_type: {config.ocr.text_adapter_type}")
print(f"  ocr.formula_adapter_type: {config.ocr.formula_adapter_type}")
print(f"  tasks: {list(config.tasks.keys())}")
print(f"  models: {list(config.llm.models.keys())}")
print(f"  providers: {list(config.llm.providers.keys())}")

print("\n=== 创建 Pipeline（真实模式，无 mock）===")
pipeline = Pipeline(config)
print(f"  STAGES: {STAGES}")
print(f"  audio_extractor: {type(pipeline.audio_extractor).__name__}")
print(f"  frame_extractor: {type(pipeline.frame_extractor).__name__}")
print(f"  asr_recognizer: {type(pipeline.asr_recognizer).__name__}")
print(f"  ocr_recognizer: {type(pipeline.ocr_recognizer).__name__}")
print(f"  content_extractor: {type(pipeline.content_extractor).__name__}")
print(f"  problem_extractor: {type(pipeline.problem_extractor).__name__}")
print(f"  knowledge_extractor: {type(pipeline.knowledge_extractor).__name__}")
print(f"  mindmap_generator: {type(pipeline.mindmap_generator).__name__}")
print(f"  screenshot_capture: {type(pipeline.screenshot_capture).__name__}")
print(f"  output_assembler: {type(pipeline.output_assembler).__name__}")
print(f"  llm_client: {type(pipeline.llm_client).__name__} (mock={pipeline.llm_client.mock})")

print("\n=== Pipeline 初始化成功，可以跑 run() ===")
