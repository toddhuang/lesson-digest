"""解析 PE 文件导入表，列出 shm.dll 依赖的所有 DLL 及其是否找到。"""
import os
import struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
torch_lib = os.path.join(ROOT, ".venv", "Lib", "site-packages", "torch", "lib")


def parse_imports(path):
    """返回该 DLL 的导入表（依赖的 DLL 名列表）"""
    with open(path, "rb") as f:
        data = f.read()

    # DOS header
    if data[:2] != b"MZ":
        return []
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        return []

    # COFF header
    machine = struct.unpack_from("<H", data, e_lfanew + 4)[0]
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_hdr_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    opt_hdr_offset = e_lfanew + 24

    # Optional header magic
    magic = struct.unpack_from("<H", data, opt_hdr_offset)[0]
    is_pe32_plus = magic == 0x20B

    # Data directories
    if is_pe32_plus:
        # PE32+: opt hdr starts at opt_hdr_offset, data dirs after 112 bytes
        num_rva_and_sizes = struct.unpack_from("<I", data, opt_hdr_offset + 108)[0]
        data_dir_offset = opt_hdr_offset + 112
    else:
        num_rva_and_sizes = struct.unpack_from("<I", data, opt_hdr_offset + 92)[0]
        data_dir_offset = opt_hdr_offset + 96

    # Import directory = directory index 1
    import_rva = struct.unpack_from("<I", data, data_dir_offset + 1 * 8)[0]
    import_size = struct.unpack_from("<I", data, data_dir_offset + 1 * 8 + 4)[0]
    if import_rva == 0:
        return []

    # Section headers
    sections = []
    sec_offset = opt_hdr_offset + opt_hdr_size
    for i in range(num_sections):
        sec = data[sec_offset + i * 40 : sec_offset + (i + 1) * 40]
        sections.append({
            "vaddr": struct.unpack_from("<I", sec, 12)[0],
            "vsize": struct.unpack_from("<I", sec, 8)[0],
            "raw_offset": struct.unpack_from("<I", sec, 20)[0],
            "raw_size": struct.unpack_from("<I", sec, 16)[0],
        })

    def rva_to_offset(rva):
        for s in sections:
            if s["vaddr"] <= rva < s["vaddr"] + s["vsize"]:
                return s["raw_offset"] + (rva - s["vaddr"])
        return None

    # Parse import descriptor entries (20 bytes each)
    dlls = []
    offset = rva_to_offset(import_rva)
    if offset is None:
        return []
    while True:
        entry = data[offset : offset + 20]
        if len(entry) < 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        if name_rva == 0:
            break
        name_offset = rva_to_offset(name_rva)
        if name_offset is None:
            offset += 20
            continue
        # Read null-terminated string
        end = data.find(b"\x00", name_offset)
        name = data[name_offset:end].decode("ascii", errors="replace")
        dlls.append(name)
        offset += 20
    return dlls


# Windows DLL search dirs
import ctypes
sys_dirs = os.environ.get("PATH", "").split(os.pathsep)
sys32 = r"C:\Windows\System32"
sys_dirs = [sys32] + sys_dirs

# Also search torch/lib
search_dirs = [torch_lib] + sys_dirs

print("=== shm.dll imports ===")
shm_path = os.path.join(torch_lib, "shm.dll")
imports = parse_imports(shm_path)
for dll in imports:
    found = os.path.exists(os.path.join(torch_lib, dll))
    if not found:
        for d in sys_dirs:
            if os.path.exists(os.path.join(d, dll)):
                found = True
                break
    status = "FOUND" if found else "MISSING"
    print(f"  {status:7} {dll}")

print("\n=== torch_python.dll imports ===")
tp_path = os.path.join(torch_lib, "torch_python.dll")
imports = parse_imports(tp_path)
for dll in imports:
    found = os.path.exists(os.path.join(torch_lib, dll))
    if not found:
        for d in sys_dirs:
            if os.path.exists(os.path.join(d, dll)):
                found = True
                break
    status = "FOUND" if found else "MISSING"
    print(f"  {status:7} {dll}")
