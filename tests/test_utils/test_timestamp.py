"""utils/timestamp.py 测试：format_timestamp + parse_timestamp 边界覆盖"""

import pytest

from utils.timestamp import format_timestamp, parse_timestamp


class TestFormatTimestamp:
    def test_mmss_basic(self):
        assert format_timestamp(65, "mm:ss") == "01:05"

    def test_mmss_zero(self):
        assert format_timestamp(0, "mm:ss") == "00:00"

    def test_negative_clamps_to_zero(self):
        assert format_timestamp(-10, "mm:ss") == "00:00"

    def test_mmss_centiseconds(self):
        assert format_timestamp(65.25, "mm:ss.cc") == "01:05.25"

    def test_centiseconds_round_up_to_99(self):
        # 0.999 秒 → 百分秒 100 → clamp 99
        assert format_timestamp(0.999, "mm:ss.cc") == "00:00.99"

    def test_hhmmss_force_hours(self):
        assert format_timestamp(3661, "hh:mm:ss") == "01:01:01"

    def test_hhmmss_centiseconds(self):
        assert format_timestamp(3661.5, "hh:mm:ss.cc") == "01:01:01.50"

    def test_auto_switch_to_hours_when_over_hour(self):
        # 即使 fmt=mm:ss，超过 1 小时也自动用 hh:mm:ss
        assert format_timestamp(3700, "mm:ss").startswith("01:")


class TestParseTimestamp:
    def test_mmss(self):
        assert parse_timestamp("01:05") == 65.0

    def test_hhmmss(self):
        assert parse_timestamp("01:01:01") == 3661.0

    def test_seconds_only(self):
        assert parse_timestamp("30") == 30.0

    def test_with_whitespace(self):
        assert parse_timestamp("  01:05  ") == 65.0

    def test_roundtrip(self):
        for sec in [0, 30, 65, 3661]:
            assert parse_timestamp(format_timestamp(sec)) == sec


class TestRoundtrip:
    @pytest.mark.parametrize("sec,fmt", [
        (0, "mm:ss"),
        (65, "mm:ss"),
        (65.25, "mm:ss.cc"),
        (3661, "hh:mm:ss"),
        (3661.5, "hh:mm:ss.cc"),
    ])
    def test_format_then_parse(self, sec, fmt):
        formatted = format_timestamp(sec, fmt)
        # 解析回秒数（不含百分秒精度损失）
        parsed = parse_timestamp(formatted)
        assert abs(parsed - int(sec)) < 1
