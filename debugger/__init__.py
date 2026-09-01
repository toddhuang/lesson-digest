"""
debugger 包：debug 产物统一入口。

设计文档：Document/开发文档/11_debug模块设计.md
release 退出：删除本包 + config.debug.enabled=false，pipeline 不受影响（无 import 依赖）。
"""

from debugger.sink import DebugSink
from debugger.formatter import DebugFormatter

__all__ = ["DebugSink", "DebugFormatter"]
