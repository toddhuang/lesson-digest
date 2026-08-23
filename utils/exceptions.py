"""
自定义异常类
所有异常继承自 VideoContentError 基类，按模块细分。
"""


class VideoContentError(Exception):
    """项目基类异常"""
    pass


# === 配置相关 ===
class ConfigError(VideoContentError):
    """配置加载或校验失败"""
    pass


# === ffmpeg 相关 ===
class FFmpegError(VideoContentError):
    """ffmpeg 执行失败"""
    def __init__(self, message: str, stderr: str = "", returncode: int = -1):
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(message)


class InvalidVideoError(VideoContentError):
    """视频文件无效（无音轨/无视频流/损坏）"""
    pass


class VideoNotFoundError(VideoContentError):
    """视频文件不存在"""
    pass


class TimestampOutOfRangeError(VideoContentError):
    """时间戳超出视频时长范围"""
    pass


# === ASR 相关 ===
class ASRError(VideoContentError):
    """语音识别失败"""
    pass


# === OCR 相关 ===
class OCRError(VideoContentError):
    """文字识别失败"""
    pass


# === LLM 相关 ===
class LLMError(VideoContentError):
    """LLM 调用失败（含重试耗尽）"""
    pass


class LLMTimeoutError(LLMError):
    """LLM 调用超时"""
    pass


class LLMRateLimitError(LLMError):
    """LLM 速率限制（429）"""
    def __init__(self, message: str, retry_after: float = 0):
        self.retry_after = retry_after
        super().__init__(message)


class LLMServerError(LLMError):
    """LLM 服务端错误（5xx）"""
    pass


class LLMClientError(LLMError):
    """LLM 客户端错误（4xx，除429）"""
    pass


class LLMConnectionError(LLMError):
    """LLM 连接失败（健康检查未通过/重连失败）"""
    pass


class LLMContextOverflowError(LLMError):
    """输入 token 数超过模型上下文长度"""
    pass


class LLMResponseParseError(LLMError):
    """LLM 返回结果解析失败"""
    pass


class LLMContentFilterError(LLMError):
    """LLM 内容过滤"""
    pass


class InvalidBackendError(LLMError):
    """无效的后端选择"""
    pass


# === 输出相关 ===
class OutputError(VideoContentError):
    """输出组装失败"""
    pass


class FileWriteError(OutputError):
    """文件写入失败"""
    pass


class DirectoryCreateError(OutputError):
    """目录创建失败"""
    pass


# === 流水线相关 ===
class PipelineError(VideoContentError):
    """流水线执行失败"""
    def __init__(self, message: str, stage: str = "", original_error: Exception = None):
        self.stage = stage
        self.original_error = original_error
        super().__init__(message)


class EmptyResultError(VideoContentError):
    """模块返回空结果"""
    pass


# === OPML 相关 ===
class OPMLValidationError(VideoContentError):
    """OPML 格式校验失败"""
    pass


# === 适配器相关 ===
class AdapterError(VideoContentError):
    """适配器错误基类"""
    pass
