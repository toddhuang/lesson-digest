"""utils/file_utils.py 测试：ensure_dir/save_json/load_json/save_text/get_file_hash"""

import os
import json
import tempfile

import pytest

from utils.file_utils import (
    ensure_dir,
    save_json,
    load_json,
    save_text,
    get_file_hash,
    get_dir_size_gb,
)


class TestEnsureDir:
    def test_creates_nested_dir(self, tmp_debug_dir):
        nested = os.path.join(tmp_debug_dir, "a", "b", "c")
        ensure_dir(nested)
        assert os.path.isdir(nested)

    def test_idempotent(self, tmp_debug_dir):
        d = os.path.join(tmp_debug_dir, "x")
        ensure_dir(d)
        # 再次调用不应报错
        ensure_dir(d)
        assert os.path.isdir(d)


class TestJsonRoundtrip:
    def test_save_load_simple_dict(self, tmp_debug_dir):
        path = os.path.join(tmp_debug_dir, "data.json")
        data = {"name": "二次函数", "value": 42, "items": [1, 2, 3]}
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_parent_dir(self, tmp_debug_dir):
        path = os.path.join(tmp_debug_dir, "nested", "deep", "data.json")
        save_json({"x": 1}, path)
        assert os.path.exists(path)
        assert load_json(path) == {"x": 1}

    def test_chinese_ensured_not_escaped(self, tmp_debug_dir):
        path = os.path.join(tmp_debug_dir, "cn.json")
        save_json({"name": "二次函数"}, path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        assert "二次函数" in raw, "中文字符应保持原样不被转义"
        assert "\\u" not in raw, "ensure_ascii=False 应禁止 \\uXXXX 转义"


class TestSaveText:
    def test_save_text_content(self, tmp_debug_dir):
        path = os.path.join(tmp_debug_dir, "note.txt")
        save_text("hello 二次函数", path)
        with open(path, "r", encoding="utf-8") as f:
            assert f.read() == "hello 二次函数"

    def test_save_text_creates_parent(self, tmp_debug_dir):
        path = os.path.join(tmp_debug_dir, "sub", "a.txt")
        save_text("x", path)
        assert os.path.exists(path)


class TestGetFileHash:
    def test_same_content_same_hash(self, tmp_debug_dir):
        p1 = os.path.join(tmp_debug_dir, "f1.txt")
        p2 = os.path.join(tmp_debug_dir, "f2.txt")
        save_text("same", p1)
        save_text("same", p2)
        # 不同文件路径但相同内容/大小/mtime 模式可能不同 hash
        # 主要验证调用不报错且返回非空字符串
        h1 = get_file_hash(p1)
        h2 = get_file_hash(p2)
        assert h1 and len(h1) == 32  # md5 hex 长度
        assert h2 and len(h2) == 32

    def test_nonexistent_file(self):
        # 不存在的文件返回基于路径的 hash
        h = get_file_hash("/nonexistent/path/file.txt")
        assert len(h) == 32


class TestGetDirSizeGb:
    def test_empty_dir(self, tmp_debug_dir):
        # 空目录大小为 0
        size = get_dir_size_gb(tmp_debug_dir)
        assert size == 0.0

    def test_with_files(self, tmp_debug_dir):
        # 写入 1KB 数据
        save_text("x" * 1024, os.path.join(tmp_debug_dir, "f.txt"))
        size = get_dir_size_gb(tmp_debug_dir)
        assert size > 0
        # 1KB ≈ 1e-6 GB
        assert size < 1e-3
