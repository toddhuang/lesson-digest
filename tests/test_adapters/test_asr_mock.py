"""adapters/asr/mock.py 测试：MockASRAdapter 返回 RawTranscript"""

import pytest

from utils.models import RawTranscript, CharTime


class TestMockASRAdapter:
    def test_transcribe_returns_raw_transcript(self):
        from adapters.asr.mock import MockASRAdapter
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        assert isinstance(result, RawTranscript)

    def test_text_is_nonempty_educational_content(self):
        from adapters.asr.mock import MockASRAdapter
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        assert len(result.text) > 100, "模拟文本应有足够教学长度"
        # 应包含数学相关关键词
        assert "方程" in result.text or "公式" in result.text

    def test_char_timestamps_length_matches_text(self):
        from adapters.asr.mock import MockASRAdapter
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        assert len(result.char_timestamps) == len(result.text), \
            "char_timestamps 长度应与 text 等长"

    def test_punctuation_positions_are_none(self):
        from adapters.asr.mock import MockASRAdapter, _PUNCTUATION
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        # 遍历每个标点位置，验证对应 char_timestamps 为 None
        none_count = 0
        for i, ch in enumerate(result.text):
            if ch in _PUNCTUATION:
                assert result.char_timestamps[i] is None, \
                    f"标点 '{ch}' 位置 {i} 应为 None"
                none_count += 1
        assert none_count > 0, "应有标点字符"

    def test_non_punctuation_has_valid_char_time(self):
        from adapters.asr.mock import MockASRAdapter, _PUNCTUATION
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        for i, ch in enumerate(result.text):
            if ch not in _PUNCTUATION:
                ct = result.char_timestamps[i]
                assert ct is not None, f"非标点 '{ch}' 位置 {i} 应有 CharTime"
                assert isinstance(ct, CharTime)
                assert ct.end_ms > ct.start_ms, "end_ms 应 > start_ms"

    def test_timestamps_monotonically_increasing(self):
        from adapters.asr.mock import MockASRAdapter, _PUNCTUATION
        adapter = MockASRAdapter()
        result = adapter.transcribe("/fake/audio.wav")
        prev_end = 0
        for i, ch in enumerate(result.text):
            ct = result.char_timestamps[i]
            if ct is None:
                continue
            assert ct.start_ms >= prev_end, \
                f"字 {i} start_ms({ct.start_ms}) 应 >= 前一字 end_ms({prev_end})"
            prev_end = ct.end_ms

    def test_load_unload_lifecycle(self):
        from adapters.asr.mock import MockASRAdapter
        adapter = MockASRAdapter()
        assert adapter._loaded is False
        adapter.load_model({})
        assert adapter._loaded is True
        adapter.unload_model()
        assert adapter._loaded is False

    def test_transcribe_ignores_audio_path(self):
        """transcribe 不实际读音频文件，任何路径都返回相同 mock 数据"""
        from adapters.asr.mock import MockASRAdapter
        adapter = MockASRAdapter()
        r1 = adapter.transcribe("/path1.wav")
        r2 = adapter.transcribe("/path2.wav")
        assert r1.text == r2.text
