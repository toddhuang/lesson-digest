"""
cudnn DLL 统一脚本
解决 PyTorch 与 PaddlePaddle 同名 cudnn DLL 版本冲突（WinError 127）。

背景：
- torch 自带 cudnn 9.9（torch/lib/），paddle 的 nvidia-cudnn-cu12 依赖是 9.5（nvidia/cudnn/bin/）
- paddle 编译时使用 cudnn 9.9，两份 DLL 同名不同构建，同进程加载会符号错配崩溃
- 统一方案：torch/lib 的 cudnn 改名 .bak（loader 不再加载），
  用 9.9 版本覆盖 nvidia/cudnn/bin（torch 和 paddle 都从这里加载同一份文件）

使用：安装完 requirements.txt 后运行一次：
    python scripts/setup_cudnn.py

注意：pip 重装 torch 或 nvidia-cudnn-cu12 会恢复原状，需重新运行本脚本。
"""

import os
import shutil
import sys

SITE_PACKAGES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv", "Lib", "site-packages"
)
TORCH_LIB = os.path.join(SITE_PACKAGES, "torch", "lib")
NVIDIA_BIN = os.path.join(SITE_PACKAGES, "nvidia", "cudnn", "bin")


def main():
    if sys.platform != "win32":
        print("非 Windows 平台无需此脚本")
        return

    if not os.path.isdir(TORCH_LIB):
        print(f"torch/lib 不存在: {TORCH_LIB}")
        sys.exit(1)
    if not os.path.isdir(NVIDIA_BIN):
        print(f"nvidia/cudnn/bin 不存在: {NVIDIA_BIN}")
        sys.exit(1)

    # 1. torch/lib 的 cudnn DLL 改名 .bak
    for name in os.listdir(TORCH_LIB):
        if name.startswith("cudnn") and name.endswith(".dll"):
            src = os.path.join(TORCH_LIB, name)
            dst = src + ".bak"
            shutil.move(src, dst)
            print(f"renamed: torch/lib/{name} -> .bak")

    # 2. 用 torch 自带的 9.9（.bak 文件）覆盖 nvidia/cudnn/bin 的 9.5
    for name in os.listdir(TORCH_LIB):
        if name.startswith("cudnn") and name.endswith(".dll.bak"):
            src = os.path.join(TORCH_LIB, name)
            dst = os.path.join(NVIDIA_BIN, name.replace(".dll.bak", ".dll"))
            if os.path.exists(dst):
                bak = dst + ".v95.bak"
                if not os.path.exists(bak):
                    shutil.move(dst, bak)  # 保留 9.5 备份
            shutil.copy2(src, dst)
            print(f"copied 9.9: nvidia/cudnn/bin/{os.path.basename(dst)}")

    print("\ncudnn DLL 统一完成（版本 9.9，torch 与 paddle 共用 nvidia/cudnn/bin）")


if __name__ == "__main__":
    main()
