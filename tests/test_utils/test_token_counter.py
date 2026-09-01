"""utils/token_counter.py 测试：count_tokens 估算规则"""

from utils.token_counter import count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_pure_chinese(self):
        # 4 个中文字 → 4 * 1.5 = 6
        assert count_tokens("二次函数") == 6

    def test_pure_english(self):
        # 5 个英文字 → 5 * 0.4 = 2 (int 取整)
        assert count_tokens("hello") == 2

    def test_mixed(self):
        # "学a习b" → 2 中 + 2 英 → 2*1.5 + 2*0.4 = 3.8 → int 3
        assert count_tokens("学a习b") == 3

    def test_numbers(self):
        # 数字按 other_count 处理
        assert count_tokens("12345") == int(5 * 0.4)  # 2

    def test_none_input_returns_zero(self):
        # count_tokens(None) 走 `if not text` 分支，返回 0
        assert count_tokens(None) == 0
