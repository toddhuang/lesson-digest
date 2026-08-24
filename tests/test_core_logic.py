"""
核心模块单元测试
覆盖纯逻辑函数，不依赖外部服务（LLM/ASR/OCR）
运行方式: python -m pytest tests/test_core_logic.py -v
"""

import os
import sys
import tempfile
import unittest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import Sentence, Problem
from core.text_merger import TextMerger
from utils.asr_corrector import ASRCorrector
from core.problem_extractor import ProblemExtractor
from core.mindmap_generator import MindmapGenerator


class TestTextMerger(unittest.TestCase):
    """M6 文本整理模块测试"""

    def setUp(self):
        self.merger = TextMerger()

    def test_merge_basic(self):
        """基本合并：按时间戳排序，带时间戳前缀"""
        sentences = [
            Sentence(start_time=35.0, end_time=40.0, text="第二句", confidence=0.9),
            Sentence(start_time=0.0, end_time=5.0, text="第一句", confidence=0.95),
        ]
        result = self.merger.merge(sentences)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("[00:00]"))
        self.assertIn("第一句", lines[0])
        self.assertTrue(lines[1].startswith("[00:35]"))
        self.assertIn("第二句", lines[1])

    def test_merge_empty(self):
        """空输入返回空字符串"""
        result = self.merger.merge([])
        self.assertEqual(result, "")

    def test_merge_timestamp_format_hhmmss(self):
        """超过1小时的时间戳格式"""
        sentences = [
            Sentence(start_time=3661.0, end_time=3665.0, text="一小时后的句子", confidence=0.9),
        ]
        result = self.merger.merge(sentences)
        # format_timestamp 默认 mm:ss，但超过1小时会自动用 hh:mm:ss
        self.assertIn("一小时后的句子", result)


class TestASRCorrector(unittest.TestCase):
    """ASR 纠错器测试（纯逻辑部分，不调用 LLM）"""

    def setUp(self):
        # 用 None 作为 llm_client，只测试纯逻辑方法
        self.corrector = ASRCorrector(llm_client=None)

    def test_align_lines_exact_match(self):
        """行数完全匹配：一一对应，时间戳保留"""
        original = [
            Sentence(start_time=0.0, end_time=5.0, text="地物里面你要研究公鸡", confidence=0.9),
            Sentence(start_time=5.0, end_time=10.0, text="和蛤蟆", confidence=0.85),
        ]
        corrected_lines = ["生物里面你要研究公鸡", "和蛤蟆"]
        result = self.corrector._align_lines(original, corrected_lines)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "生物里面你要研究公鸡")
        self.assertEqual(result[0].start_time, 0.0)
        self.assertEqual(result[1].text, "和蛤蟆")
        self.assertEqual(result[1].start_time, 5.0)

    def test_align_lines_empty_lines_filtered(self):
        """空行被过滤"""
        original = [
            Sentence(start_time=0.0, end_time=5.0, text="第一句", confidence=0.9),
        ]
        corrected_lines = ["第一句", "", "   "]
        result = self.corrector._align_lines(original, corrected_lines)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "第一句")

    def test_text_similarity_identical(self):
        """完全相同的文本相似度为1.0"""
        sim = self.corrector._text_similarity("你好世界", "你好世界")
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_text_similarity_no_overlap(self):
        """完全不重叠的文本相似度为0"""
        sim = self.corrector._text_similarity("abc", "xyz")
        self.assertAlmostEqual(sim, 0.0, places=2)

    def test_text_similarity_partial(self):
        """部分重叠的文本相似度在0-1之间"""
        sim = self.corrector._text_similarity("你好世界", "你好中国")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_correct_empty_input(self):
        """空输入直接返回空列表（不调用LLM）"""
        result = self.corrector.correct([])
        self.assertEqual(result, [])


class TestProblemExtractor(unittest.TestCase):
    """M8 题目提取器测试（纯逻辑部分）"""

    def setUp(self):
        # 用 None 作为 llm_client，只测试纯逻辑方法
        self.extractor = ProblemExtractor(llm_client=None)

    def test_is_duplicate_same_time_similar_text(self):
        """时间接近、文本相似 → 重复"""
        p1 = Problem(start_time=100.0, question_text="已知函数f(x)=x^2+1，求f(2)的值")
        p2 = Problem(start_time=110.0, question_text="已知函数f(x)=x^2+1求f(2)")
        self.assertTrue(self.extractor._is_duplicate(p1, p2))

    def test_is_duplicate_far_time(self):
        """时间相差超过120秒 → 不重复（即使文本相似）"""
        p1 = Problem(start_time=100.0, question_text="已知函数f(x)=x^2+1，求f(2)的值")
        p2 = Problem(start_time=300.0, question_text="已知函数f(x)=x^2+1，求f(2)的值")
        self.assertFalse(self.extractor._is_duplicate(p1, p2))

    def test_is_duplicate_different_text(self):
        """时间接近但文本完全不同 → 不重复"""
        p1 = Problem(start_time=100.0, question_text="求一元二次方程x^2-5x+6=0的根")
        p2 = Problem(start_time=110.0, question_text="证明三角形内角和等于180度")
        self.assertFalse(self.extractor._is_duplicate(p1, p2))

    def test_is_duplicate_empty_text(self):
        """空文本 → 不重复"""
        p1 = Problem(start_time=100.0, question_text="")
        p2 = Problem(start_time=110.0, question_text="某道题目")
        self.assertFalse(self.extractor._is_duplicate(p1, p2))


class TestMindmapGenerator(unittest.TestCase):
    """M10 思维导图生成器测试（纯逻辑部分）"""

    def setUp(self):
        self.generator = MindmapGenerator(llm_client=None)

    def test_validate_opml_valid(self):
        """合法 OPML 通过校验"""
        valid_opml = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head><title>测试</title></head>
  <body>
    <outline text="根节点">
      <outline text="子节点"/>
    </outline>
  </body>
</opml>"""
        # 不抛异常即为通过
        self.generator._validate_opml(valid_opml)

    def test_validate_opml_invalid_xml(self):
        """非法 XML 抛 OPMLValidationError"""
        from utils.exceptions import OPMLValidationError
        with self.assertRaises(OPMLValidationError):
            self.generator._validate_opml("不是合法的XML")

    def test_validate_opml_wrong_root(self):
        """根节点不是 opml 抛异常"""
        from utils.exceptions import OPMLValidationError
        wrong_opml = '<root version="2.0"><head/><body/></root>'
        with self.assertRaises(OPMLValidationError):
            self.generator._validate_opml(wrong_opml)

    def test_validate_opml_missing_body(self):
        """缺少 body 节点抛异常"""
        from utils.exceptions import OPMLValidationError
        missing_body = '<opml version="2.0"><head><title>t</title></head></opml>'
        with self.assertRaises(OPMLValidationError):
            self.generator._validate_opml(missing_body)

    def test_clean_opml_removes_codeblock(self):
        """清理 Markdown 代码块标记"""
        content = "```xml\n<opml>test</opml>\n```"
        result = self.generator._clean_opml(content)
        self.assertFalse(result.startswith("```"))
        self.assertIn("<opml>", result)


class TestImagePreprocess(unittest.TestCase):
    """颜色过滤预处理测试（用合成图片）"""

    def test_remove_color_keep_black(self):
        """黑色文字保留，彩色文字被去除"""
        import cv2
        import numpy as np
        from utils.image_preprocess import remove_color_keep_black

        # 创建合成图片：白色背景，黑色文字区域，彩色文字区域
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # 黑色矩形（模拟黑色题目文字）
        img[20:40, 20:80] = [0, 0, 0]
        # 红色矩形（模拟老师彩色手写）
        img[60:80, 20:80] = [0, 0, 255]  # BGR格式，红色

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            input_path = f.name
        output_path = input_path.replace(".jpg", "_out.jpg")
        cv2.imwrite(input_path, img)

        try:
            result_path = remove_color_keep_black(input_path, output_path, black_threshold=120)
            result = cv2.imread(result_path)
            self.assertIsNotNone(result)

            # 黑色区域应该保留（接近黑色）
            black_region = result[30, 50]
            self.assertLess(np.mean(black_region), 50)

            # 红色区域应该被去除（变成白色背景）
            red_region = result[70, 50]
            self.assertGreater(np.mean(red_region), 200)
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
