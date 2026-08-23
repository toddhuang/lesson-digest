"""
M1 配置管理模块
加载/校验 YAML 配置，集中管理所有参数。
对应文档：03_接口设计/M1_配置管理模块接口.md
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

from utils.exceptions import ConfigError


# === 配置数据类 ===

@dataclass
class VideoConfig:
    frame_interval: int = 30
    frame_format: str = "jpg"
    frame_quality: int = 90


@dataclass
class LocalLLMConfig:
    base_url: str = "http://192.168.x.x:8000/v1"
    model: str = "qwen3.6-27b-awq"
    context_length: int = 8192


@dataclass
class CloudLLMConfig:
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    context_length: int = 131072
    api_key: str = ""


@dataclass
class LLMConfig:
    local: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    cloud: CloudLLMConfig = field(default_factory=CloudLLMConfig)
    max_retries: int = 5
    health_check_interval: int = 30


@dataclass
class ASRConfig:
    adapter_type: str = "funasr"
    sample_rate: int = 16000
    channels: int = 1
    model_name: str = "paraformer-zh"


@dataclass
class OCRConfig:
    adapter_type: str = "paddleocr"
    language: str = "ch"
    use_angle_cls: bool = True


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
    llm_cache: bool = True


@dataclass
class Config:
    """全局配置根对象"""
    video: VideoConfig = field(default_factory=VideoConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
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
            # 配置文件不存在，使用默认配置
            return self.config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"配置文件 YAML 解析失败: {e}")

        self._apply_dict(data)
        self._validate()
        self._load_env_vars()
        return self.config

    def _apply_dict(self, data: dict) -> None:
        """将字典数据应用到配置对象"""
        if "video" in data:
            for k, v in data["video"].items():
                if hasattr(self.config.video, k):
                    setattr(self.config.video, k, v)

        if "llm" in data:
            llm_data = data["llm"]
            if "local" in llm_data:
                for k, v in llm_data["local"].items():
                    if hasattr(self.config.llm.local, k):
                        setattr(self.config.llm.local, k, v)
            if "cloud" in llm_data:
                for k, v in llm_data["cloud"].items():
                    if hasattr(self.config.llm.cloud, k):
                        setattr(self.config.llm.cloud, k, v)
            for k in ("max_retries", "health_check_interval"):
                if k in llm_data:
                    setattr(self.config.llm, k, llm_data[k])

        if "asr" in data:
            for k, v in data["asr"].items():
                if hasattr(self.config.asr, k):
                    setattr(self.config.asr, k, v)

        if "ocr" in data:
            for k, v in data["ocr"].items():
                if hasattr(self.config.ocr, k):
                    setattr(self.config.ocr, k, v)

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

    def _validate(self) -> None:
        """校验配置参数合法性"""
        if self.config.video.frame_interval <= 0:
            raise ConfigError("video.frame_interval 必须大于 0")
        if self.config.llm.max_retries < 0:
            raise ConfigError("llm.max_retries 不能小于 0")
        if self.config.cache.max_cache_size_gb <= 0:
            raise ConfigError("cache.max_cache_size_gb 必须大于 0")

    def _load_env_vars(self) -> None:
        """从环境变量加载敏感信息（API Key）"""
        env_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if env_api_key and not self.config.llm.cloud.api_key:
            self.config.llm.cloud.api_key = env_api_key

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
