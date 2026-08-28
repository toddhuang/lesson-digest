"""
Windows GPU DLL 路径设置
解决 PyTorch 与 PaddlePaddle 共存时的 CUDA/cuDNN DLL 冲突。

背景：
- PaddlePaddle 3.2.2 依赖 nvidia pip 包的 cudnn 9.5（.venv/Lib/site-packages/nvidia/）
- PyTorch 2.13.0+cu126 自带 cudnn 9.9 DLL（torch/lib/），与 PaddlePaddle 的同名冲突
- 统一方案：PyTorch 自带的 cudnn DLL 已改名 .bak，两框架都用 nvidia 包的 cudnn 9.5

使用：必须在 import torch / paddle 之前调用 setup_gpu_path()。
"""

import os
import sys


def setup_gpu_path() -> None:
    """Windows 平台：将 CUDA/cuDNN DLL 目录加入 PATH。

    PATH 顺序（优先级从高到低）：
    1. .venv/Lib/site-packages/nvidia/*/bin — nvidia pip 包的 CUDA/cuDNN 库
    2. .venv/Lib/site-packages/torch/lib — PyTorch 自带的其他 CUDA 库

    仅在 Windows 平台生效，Linux/macOS 不处理。
    """
    if sys.platform != "win32":
        return

    prepend_dirs = []
    site_packages = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "Lib", "site-packages"
    )

    # 1. nvidia pip 包的所有 bin 目录（cudnn、cuda_runtime、cublas 等）
    nvidia_dir = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_dir):
        for pkg in sorted(os.listdir(nvidia_dir)):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir):
                prepend_dirs.append(bin_dir)

    # 2. torch/lib（PyTorch 自带 CUDA 库，cudnn DLL 已改名 .bak 避免冲突）
    torch_lib = os.path.join(site_packages, "torch", "lib")
    if os.path.isdir(torch_lib):
        prepend_dirs.append(torch_lib)

    if prepend_dirs:
        os.environ["PATH"] = os.pathsep.join(prepend_dirs) + os.pathsep + os.environ.get("PATH", "")


def preload_torch() -> None:
    """预加载 PyTorch 的 CUDA DLL（必须在 import paddle 之前调用）。

    原因：Windows DLL 搜索顺序中应用目录/System32 优先于 add_dll_directory 目录，
    若 paddle 独自加载 cudnn，其依赖可能命中系统中的旧版 CUDA DLL 导致
    ERROR_PROC_NOT_FOUND (WinError 127)。先 import torch 可将正确版本的
    CUDA 12.6 DLL 加载进进程，paddle 后续加载时按模块名复用，避免命中旧版本。

    FunASR 本身依赖 torch，提前加载无额外开销。
    """
    if sys.platform != "win32":
        return
    import torch  # noqa: F401
