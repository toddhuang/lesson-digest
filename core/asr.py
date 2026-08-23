"""
M4 语音识别模块
语音识别业务逻辑（批量处理、缓存、时间戳管理），通过适配层调用具体ASR引擎。
对应文档：03_接口设计/M4_语音识别模块接口.md
"""

import os
import json
from typing import List

from utils.models import Sentence
from utils.file_utils import ensure_dir, save_json, load_json
from utils.logger import setup_logger
from adapters.asr_adapter import create_asr_adapter, ASRAdapter

logger = setup_logger("M4_asr")


class ASRRecognizer:
    """语音识别器"""

    def __init__(self, adapter_type: str = "mock", config: dict = None, cache_dir: str = "./cache/asr"):
        self.adapter_type = adapter_type
        self.config = config or {}
        self.cache_dir = cache_dir
        self._adapter: ASRAdapter = None

    def _get_adapter(self) -> ASRAdapter:
        if self._adapter is None:
            self._adapter = create_asr_adapter(self.adapter_type, self.config)
        return self._adapter

    def recognize(self, audio_path: str, use_cache: bool = True) -> List[Sentence]:
        """语音识别

        Args:
            audio_path: 音频文件路径
            use_cache: 是否使用缓存

        Returns:
            Sentence 列表
        """
        logger.info(f"[M4] 语音识别: {audio_path}")

        # 缓存检查
        cache_key = os.path.splitext(os.path.basename(audio_path))[0]
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if use_cache and os.path.exists(cache_path):
            logger.info(f"[M4] 命中缓存: {cache_path}")
            data = load_json(cache_path)
            return [Sentence(**s) for s in data]

        # 调用适配层
        adapter = self._get_adapter()
        sentences = adapter.transcribe(audio_path)

        # 写入缓存
        if use_cache:
            ensure_dir(self.cache_dir)
            save_json([s.__dict__ for s in sentences], cache_path)

        logger.info(f"[M4] 语音识别完成: {len(sentences)}句")
        return sentences
