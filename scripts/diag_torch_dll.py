"""诊断 torch shm.dll 加载失败：尝试加载并列出缺失的依赖 DLL。"""
import os
import sys
import ctypes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
torch_lib = os.path.join(ROOT, ".venv", "Lib", "site-packages", "torch", "lib")
shm_path = os.path.join(torch_lib, "shm.dll")

print(f"shm.dll exists: {os.path.exists(shm_path)}")
print(f"Python: {sys.version}")
print(f"ARCH: {sys.maxsize > 2**32 and 'x64' or 'x86'}")

# 把 torch/lib 加入 DLL 搜索路径（模拟 torch.__init__ 的行为）
os.add_dll_directory(torch_lib)

try:
    dll = ctypes.WinDLL(shm_path)
    print(f"OK: shm.dll loaded, handle={dll._handle}")
except OSError as e:
    print(f"FAIL: {e}")
    # 列出 torch/lib 下所有 DLL，逐个试探哪个缺失
    dlls = sorted(f for f in os.listdir(torch_lib) if f.endswith(".dll"))
    print(f"\nTrying {len(dlls)} DLLs in torch/lib to find the missing one:")
    for name in dlls:
        path = os.path.join(torch_lib, name)
        try:
            ctypes.WinDLL(path)
            print(f"  OK   {name}")
        except OSError as e2:
            print(f"  FAIL {name}: {e2}")
