"""adapters/ocr/mock.py 测试：MockOCRAdapter 按帧名返回不同 OCRResult"""

import pytest

from utils.models import OCRResult


class TestMockOCRAdapter:
    @pytest.fixture
    def adapter(self):
        from adapters.ocr.mock import MockOCRAdapter
        return MockOCRAdapter()

    def test_recognize_returns_list_of_ocr_result(self, adapter):
        results = adapter.recognize("/fake/frame_000001.jpg")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, OCRResult)

    def test_frame_000001_returns_definition(self, adapter):
        results = adapter.recognize("/fake/frame_000001.jpg")
        texts = " ".join(r.text for r in results)
        assert "一元二次方程" in texts
        assert "定义" in texts

    def test_frame_000002_returns_example_problem(self, adapter):
        results = adapter.recognize("/fake/frame_000002.jpg")
        texts = " ".join(r.text for r in results)
        assert "例题" in texts
        assert "x²-5x+6=0" in texts

    def test_frame_000003_returns_formula_derivation(self, adapter):
        results = adapter.recognize("/fake/frame_000003.jpg")
        texts = " ".join(r.text for r in results)
        assert "求根公式" in texts

    def test_frame_000004_returns_discriminant(self, adapter):
        results = adapter.recognize("/fake/frame_000004.jpg")
        texts = " ".join(r.text for r in results)
        assert "判别式" in texts or "Δ" in texts

    def test_unknown_frame_returns_default(self, adapter):
        results = adapter.recognize("/fake/random_frame.jpg")
        assert len(results) == 1
        assert results[0].text == "教学课件"

    def test_bounding_box_normalized_0_to_1(self, adapter):
        results = adapter.recognize("/fake/frame_000001.jpg")
        for r in results:
            assert len(r.bounding_box) == 4
            for v in r.bounding_box:
                assert 0.0 <= v <= 1.0, f"bounding_box 值应在 0-1 范围内，实际 {v}"

    def test_confidence_in_valid_range(self, adapter):
        for frame_name in ("frame_000001", "frame_000002", "frame_000003", "frame_000004", "unknown"):
            results = adapter.recognize(f"/fake/{frame_name}.jpg")
            for r in results:
                assert 0.0 <= r.confidence <= 1.0, f"置信度应 0-1，实际 {r.confidence}"

    def test_load_unload_lifecycle(self, adapter):
        assert adapter._loaded is False
        adapter.load_model({})
        assert adapter._loaded is True
        adapter.unload_model()
        assert adapter._loaded is False

    def test_recognize_ignores_real_file_existence(self, adapter):
        """recognize 不实际读图片文件，任何路径都返回 mock 数据"""
        r1 = adapter.recognize("/nonexistent/frame_000001.jpg")
        r2 = adapter.recognize("/different/path/frame_000001.jpg")
        assert [r.text for r in r1] == [r.text for r in r2]
