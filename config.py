"""
M1 配置管理模块
加载/校验 YAML 配置，集中管理所有参数。
对应文档：03_接口设计/M1_配置管理模块接口.md

重构记录（M11-M17）：
- 删除 .env / python-dotenv 双配置系统，统一 config.yaml
- 新增模型注册表（models）、服务商配置（providers）、任务映射（tasks）
- API Key 直接写在 config.yaml 中（config.yaml 已加入 .gitignore）
- 删除 LLM 层缓存配置（缓存由 pipeline 层管理）
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import yaml

from utils.exceptions import ConfigError


# === 枚举 ===

class ModelCapability(Enum):
    """模型能力标识"""
    TEXT = "text"
    REASONING = "reasoning"
    VISION = "vision"


# === 配置数据类 ===

@dataclass
class VideoConfig:
    frame_interval: int = 30
    frame_format: str = "jpg"
    frame_quality: int = 90


@dataclass
class ModelConfig:
    """单个模型配置（模型注册表条目）"""
    name: str = ""
    provider: str = ""
    capabilities: List[str] = field(default_factory=list)
    context_length: int = 131072
    max_output: int = 8192


@dataclass
class ProviderConfig:
    """LLM 服务商配置"""
    base_url: str = ""
    api_key: str = ""
    litellm_prefix: str = "openai"


@dataclass
class TaskConfig:
    """任务-模型映射配置"""
    model: str = ""
    temperature: float = 0.1


@dataclass
class LLMConfig:
    """LLM 全局配置"""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    max_retries: int = 3
    timeout: int = 120


@dataclass
class ASRConfig:
    adapter_type: str = "funasr"
    sample_rate: int = 16000
    channels: int = 1
    model_name: str = "paraformer-zh"


@dataclass
class OCRConfig:
    """OCR 全局配置（两引擎并行，R-008 定案）"""
    text_adapter_type: str = "paddleocr"
    formula_adapter_type: str = "formula_net"
    enable_color_filter: bool = False  # P2 可选，MVP 默认关闭
    black_threshold: int = 120


@dataclass
class FrameDedupConfig:
    """M3 关键帧去重配置（R-009 定案：dHash 阈值 0.02）"""
    algorithm: str = "dhash"
    threshold: float = 0.02
    interval_sec: float = 1.0


@dataclass
class TextOCRConfig:
    """文字识别引擎配置（PP-OCRv6）"""
    adapter_type: str = "paddleocr"
    text_detection_model_name: str = "PP-OCRv6_small_det"
    text_recognition_model_name: str = "PP-OCRv6_small_rec"


@dataclass
class FormulaOCRConfig:
    """公式识别引擎配置（PP-FormulaNet）"""
    adapter_type: str = "formula_net"
    formula_model_name: str = "PP-FormulaNet_plus-M"


@dataclass
class OutputConfig:
    transcript_filename: str = "01_逐字稿.md"
    knowledge_filename: str = "02_知识点清单.md"
    mindmap_filename: str = "03_思维导图.opml"
    problems_dirname: str = "习题"
    screenshots_dirname: str = "截图"
    timestamp_format: str = "mm:ss"


@dataclass
class PathsConfig:
    cache_dir: str = "./cache"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    log_dir: str = "./logs"


@dataclass
class CacheConfig:
    cache_ocr_frames: bool = True
    max_cache_size_gb: int = 400


@dataclass
class Config:
    """全局配置根对象"""
    video: VideoConfig = field(default_factory=VideoConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    asr: ASRConfig = field(default_factory=ASRConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    frame_dedup: FrameDedupConfig = field(default_factory=FrameDedupConfig)
    text_ocr: TextOCRConfig = field(default_factory=TextOCRConfig)
    formula_ocr: FormulaOCRConfig = field(default_factory=FormulaOCRConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


# === 配置管理器 ===

class ConfigManager:
    """配置管理器，负责加载、校验、访问配置"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config: Config = Config()

    def load(self) -> Config:
        """加载配置文件

        Returns:
            Config 对象

        Raises:
            ConfigError: 配置文件不存在或格式错误
        """
        if not os.path.exists(self.config_path):
            raise ConfigError(
                f"配置文件不存在: {self.config_path}，"
                f"请复制 config.example.yaml 为 config.yaml 并填写 API Key"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"配置文件 YAML 解析失败: {e}")

        self._apply_dict(data)
        self._validate()
        return self.config

    def _apply_dict(self, data: dict) -> None:
        """将字典数据应用到配置对象"""
        if "video" in data:
            for k, v in data["video"].items():
                if hasattr(self.config.video, k):
                    setattr(self.config.video, k, v)

        if "llm" in data:
            self._apply_llm_config(data["llm"])

        if "tasks" in data and isinstance(data["tasks"], dict):
            for task_name, task_data in data["tasks"].items():
                if isinstance(task_data, dict):
                    self.config.tasks[task_name] = TaskConfig(
                        model=task_data.get("model", ""),
                        temperature=task_data.get("temperature", 0.1),
                    )

        if "asr" in data:
            for k, v in data["asr"].items():
                if hasattr(self.config.asr, k):
                    setattr(self.config.asr, k, v)

        if "ocr" in data:
            for k, v in data["ocr"].items():
                if hasattr(self.config.ocr, k):
                    setattr(self.config.ocr, k, v)

        if "frame_dedup" in data:
            for k, v in data["frame_dedup"].items():
                if hasattr(self.config.frame_dedup, k):
                    setattr(self.config.frame_dedup, k, v)

        if "text_ocr" in data:
            for k, v in data["text_ocr"].items():
                if hasattr(self.config.text_ocr, k):
                    setattr(self.config.text_ocr, k, v)

        if "formula_ocr" in data:
            for k, v in data["formula_ocr"].items():
                if hasattr(self.config.formula_ocr, k):
                    setattr(self.config.formula_ocr, k, v)

        if "output" in data:
            for k, v in data["output"].items():
                if hasattr(self.config.output, k):
                    setattr(self.config.output, k, v)

        if "paths" in data:
            for k, v in data["paths"].items():
                if hasattr(self.config.paths, k):
                    setattr(self.config.paths, k, v)

        if "cache" in data:
            for k, v in data["cache"].items():
                if hasattr(self.config.cache, k):
                    setattr(self.config.cache, k, v)

    def _apply_llm_config(self, llm_data: dict) -> None:
        """解析 LLM 配置段"""
        # max_retries / timeout
        if "max_retries" in llm_data:
            self.config.llm.max_retries = llm_data["max_retries"]
        if "timeout" in llm_data:
            self.config.llm.timeout = llm_data["timeout"]

        # models（模型注册表，列表格式）
        if "models" in llm_data and isinstance(llm_data["models"], list):
            for model_data in llm_data["models"]:
                if not isinstance(model_data, dict):
                    continue
                model_config = ModelConfig(
                    name=model_data.get("name", ""),
                    provider=model_data.get("provider", ""),
                    capabilities=model_data.get("capabilities", []),
                    context_length=model_data.get("context_length", 131072),
                    max_output=model_data.get("max_output", 8192),
                )
                if model_config.name:
                    self.config.llm.models[model_config.name] = model_config

        # providers（服务商配置）
        if "providers" in llm_data and isinstance(llm_data["providers"], dict):
            for provider_name, provider_data in llm_data["providers"].items():
                if not isinstance(provider_data, dict):
                    continue
                self.config.llm.providers[provider_name] = ProviderConfig(
                    base_url=provider_data.get("base_url", ""),
                    api_key=provider_data.get("api_key", ""),
                    litellm_prefix=provider_data.get("litellm_prefix", "openai"),
                )

    def _validate(self) -> None:
        """校验配置参数合法性和引用完整性"""
        if self.config.video.frame_interval <= 0:
            raise ConfigError("video.frame_interval 必须大于 0")
        if self.config.llm.max_retries < 0:
            raise ConfigError("llm.max_retries 不能小于 0")
        if self.config.cache.max_cache_size_gb <= 0:
            raise ConfigError("cache.max_cache_size_gb 必须大于 0")

        # 校验模型注册表中每个模型引用的服务商是否存在
        for model_name, model_config in self.config.llm.models.items():
            if not model_config.provider:
                raise ConfigError(f"模型 {model_name} 未指定 provider")
            if model_config.provider not in self.config.llm.providers:
                raise ConfigError(
                    f"模型 {model_name} 引用了未配置的服务商: {model_config.provider}，"
                    f"已配置: {list(self.config.llm.providers.keys())}"
                )
            if model_config.context_length <= 0:
                raise ConfigError(f"模型 {model_name} 的 context_length 必须大于 0")
            if model_config.max_output <= 0:
                raise ConfigError(f"模型 {model_name} 的 max_output 必须大于 0")

        # 校验每个任务引用的模型是否存在
        for task_name, task_config in self.config.tasks.items():
            if not task_config.model:
                raise ConfigError(f"任务 {task_name} 未指定 model")
            if task_config.model not in self.config.llm.models:
                raise ConfigError(
                    f"任务 {task_name} 引用了未注册的模型: {task_config.model}，"
                    f"已注册: {list(self.config.llm.models.keys())}"
                )
            if not 0.0 <= task_config.temperature <= 2.0:
                raise ConfigError(
                    f"任务 {task_name} 的 temperature 必须在 0.0-2.0 之间，当前: {task_config.temperature}"
                )

        # 校验服务商 api_key 非空（仅校验被模型引用的服务商）
        referenced_providers = {m.provider for m in self.config.llm.models.values()}
        for provider_name in referenced_providers:
            provider_config = self.config.llm.providers[provider_name]
            if not provider_config.api_key:
                raise ConfigError(
                    f"服务商 {provider_name} 的 api_key 未配置，"
                    f"请在 config.yaml 中填写"
                )
            if not provider_config.base_url:
                raise ConfigError(f"服务商 {provider_name} 的 base_url 未配置")

    def get(self) -> Config:
        """获取当前配置"""
        return self.config


def load_config(config_path: Optional[str] = None) -> Config:
    """便捷函数：加载配置

    Args:
        config_path: 配置文件路径，默认 config.yaml

    Returns:
        Config 对象
    """
    manager = ConfigManager(config_path)
    return manager.load()
