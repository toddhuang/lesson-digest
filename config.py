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
class LLMProviderConfig:
    """单个 LLM 服务商配置"""
    base_url: str = ""
    model: str = ""
    context_length: int = 131072
    api_key: str = ""


@dataclass
class LLMConfig:
    default_provider: str = "deepseek"
    providers: dict = field(default_factory=dict)  # key: provider名, value: LLMProviderConfig
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

        加载顺序（后者覆盖前者）：
        1. config.yaml 默认配置
        2. .env 文件中的环境变量
        3. 系统环境变量

        Returns:
            Config 对象

        Raises:
            ConfigError: 配置文件不存在或格式错误
        """
        # 1. 加载 .env 文件（在加载 config.yaml 之前，确保环境变量可用）
        self._load_dotenv()

        if not os.path.exists(self.config_path):
            # 配置文件不存在，使用默认配置
            self._validate()
            self._load_env_vars()
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

    def _load_dotenv(self) -> None:
        """加载项目根目录下的 .env 文件

        简单的 .env 解析器，不依赖 python-dotenv 库。
        支持 KEY=VALUE 格式，忽略空行和 # 开头的注释行。
        """
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(dotenv_path):
            return

        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 忽略空行和注释
                    if not line or line.startswith("#"):
                        continue
                    # 解析 KEY=VALUE
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        # 去除引号
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        # 只设置未存在的环境变量（不覆盖已有的系统环境变量）
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            # .env 文件读取失败不影响主流程
            pass

    def _apply_dict(self, data: dict) -> None:
        """将字典数据应用到配置对象"""
        if "video" in data:
            for k, v in data["video"].items():
                if hasattr(self.config.video, k):
                    setattr(self.config.video, k, v)

        if "llm" in data:
            llm_data = data["llm"]
            # default_provider
            if "default_provider" in llm_data:
                self.config.llm.default_provider = llm_data["default_provider"]
            # providers
            if "providers" in llm_data and isinstance(llm_data["providers"], dict):
                for provider_name, provider_data in llm_data["providers"].items():
                    if isinstance(provider_data, dict):
                        provider_config = LLMProviderConfig()
                        for k, v in provider_data.items():
                            if hasattr(provider_config, k):
                                setattr(provider_config, k, v)
                        self.config.llm.providers[provider_name] = provider_config
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
        """从环境变量加载敏感信息（API Key、模型名、服务地址等）

        环境变量优先级高于 config.yaml，确保敏感信息不写入配置文件。
        支持的服务商：deepseek、volcengine（可扩展）
        """
        # 服务商环境变量映射表
        # 格式：{ 服务商名: { 配置字段: 环境变量名 } }
        provider_env_map = {
            "deepseek": {
                "api_key": "DEEPSEEK_API_KEY",
                "model": "DEEPSEEK_MODEL",
                "base_url": "DEEPSEEK_BASE_URL",
            },
            "volcengine": {
                "api_key": "VOLCENGINE_API_KEY",
                "model": "VOLCENGINE_MODEL",
                "base_url": "VOLCENGINE_BASE_URL",
            },
        }

        for provider_name, env_map in provider_env_map.items():
            # 确保服务商配置存在
            if provider_name not in self.config.llm.providers:
                self.config.llm.providers[provider_name] = LLMProviderConfig()
            provider_config = self.config.llm.providers[provider_name]

            for field_name, env_var_name in env_map.items():
                env_value = os.environ.get(env_var_name, "")
                if env_value:
                    setattr(provider_config, field_name, env_value)

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
