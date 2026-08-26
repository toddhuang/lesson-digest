"""
LLM 客户端
管理模型注册表和适配器实例，按任务名创建 LLMSession。

职责：
- 持有 LLM 配置（模型注册表、服务商配置）和任务配置
- 按任务名查找模型和 temperature，创建 LLMSession
- 懒加载适配器实例（按模型名缓存）
- 健康检查

M11-M17 重构：
- 删除多服务商 backend 别名（"cloud"/"deepseek"/"volcengine"）
- 删除 LLM 响应缓存（由 pipeline 层管理断点续传）
- 删除 chat/chat_stream 旧接口
- 模型和服务商通过配置注册，任务通过配置映射
"""

from typing import Dict, Optional

from config import LLMConfig, TaskConfig
from utils.logger import setup_logger
from utils.exceptions import ConfigError, LLMError
from adapters.llm.base import LLMAdapter
from adapters.llm.factory import create_llm_adapter
from core.llm.llm_session import LLMSession

logger = setup_logger("LLM_Client")


class LLMClient:
    """LLM 客户端

    管理模型注册表，按任务名创建会话。
    适配器实例按模型名缓存，同一模型复用一个适配器。
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        tasks: Dict[str, TaskConfig],
        mock: bool = False,
    ):
        """
        Args:
            llm_config: LLM 配置（模型注册表、服务商、重试策略）
            tasks: 任务-模型映射配置
            mock: 是否使用 Mock 适配器（用于测试）
        """
        self.llm_config = llm_config
        self.tasks = tasks
        self.mock = mock
        self._adapters: Dict[str, LLMAdapter] = {}

    def get_session(self, task_name: str) -> LLMSession:
        """根据任务名获取 LLM 会话

        Args:
            task_name: 任务名（对应 config.yaml 中 tasks 节的键）

        Returns:
            绑定了模型和 temperature 的 LLMSession

        Raises:
            ConfigError: 任务或模型未配置
        """
        task_config = self.tasks.get(task_name)
        if task_config is None:
            raise ConfigError(
                f"未配置的任务: {task_name}，"
                f"已配置: {list(self.tasks.keys())}"
            )

        model_config = self.llm_config.models.get(task_config.model)
        if model_config is None:
            raise ConfigError(
                f"任务 {task_name} 引用了未注册的模型: {task_config.model}，"
                f"已注册: {list(self.llm_config.models.keys())}"
            )

        adapter = self._get_adapter(model_config.name)
        session = LLMSession(
            adapter=adapter,
            temperature=task_config.temperature,
            model_name=model_config.name,
        )
        logger.info(
            f"[LLM] 创建会话: task={task_name}, model={model_config.name}, "
            f"temperature={task_config.temperature}"
        )
        return session

    def _get_adapter(self, model_name: str) -> LLMAdapter:
        """获取或创建指定模型的适配器（懒加载+缓存）"""
        if model_name in self._adapters:
            return self._adapters[model_name]

        model_config = self.llm_config.models[model_name]
        provider_config = self.llm_config.providers.get(model_config.provider)
        if provider_config is None:
            raise ConfigError(
                f"模型 {model_name} 引用了未配置的服务商: {model_config.provider}"
            )

        adapter = create_llm_adapter(
            model_config=model_config,
            provider_config=provider_config,
            max_retries=self.llm_config.max_retries,
            timeout=self.llm_config.timeout,
            mock=self.mock,
        )
        self._adapters[model_name] = adapter
        logger.info(f"[LLM] 创建适配器: model={model_name}, provider={model_config.provider}")
        return adapter

    def health_check(self, task_name: str) -> bool:
        """健康检查：发送一个极短请求验证连通性

        Args:
            task_name: 任务名（使用该任务配置的模型）

        Returns:
            True 表示健康
        """
        try:
            session = self.get_session(task_name)
            session.generate(prompt="你是一个健康检查助手。", payload="请回复：ok")
            return True
        except LLMError as e:
            logger.warning(
                f"[LLM] 健康检查失败 (task={task_name}): {type(e).__name__}: {e}"
            )
            return False
