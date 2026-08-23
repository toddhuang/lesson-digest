# 4090 工作站 — Ubuntu 24.04 + vLLM 部署文档

> 适用硬件：NVIDIA RTX 4090（24GB 显存）
> CPU：Intel i5-13600KF 14核20线程
> 内存：64GB DDR5-5600
> 磁盘：SOLIDIGM 2TB NVMe + Lexar ARES 4TB NVMe + Samsung 970 EVO Plus 1TB NVMe
> 网络：Intel I226-V 2.5Gbps 有线 + Wi-Fi 6E AX211
> 操作系统：Ubuntu 24.04 LTS（Server 或 Desktop 均可）
> 用途：局域网内提供 LLM 推理 API 服务（OpenAI 兼容格式）
> 文档版本：v1.8

---

## 一、软件清单总览

| 序号 | 软件                     | 版本                                           | 用途                                              |
| ---- | ------------------------ | ---------------------------------------------- | ------------------------------------------------- |
| 1    | Ubuntu                   | 24.04 LTS                                      | 操作系统                                          |
| 2    | NVIDIA Driver            | >= 595（以 `ubuntu-drivers devices` 推荐为准） | 显卡驱动（24.04 源内版本，支持 4090 + CUDA 12.x） |
| 3    | CUDA Toolkit             | 12.4                                           | GPU 计算环境（24.04 官方原生支持）                |
| 4    | Python                   | 3.12（系统自带）                               | vLLM 运行环境                                     |
| 5    | vLLM                     | 最新稳定版（>=0.6）                            | LLM 推理引擎，提供 OpenAI 兼容 API                |
| 6    | modelscope               | 最新版                                         | 国内模型下载（加速 Qwen 模型拉取）                |
| 7    | tclf90/Qwen3.6-27B-AWQ  | —                                              | 本地大语言模型（社区 AWQ 4bit，约 15GB）          |

---

## 二、系统准备

### 2.1 Ubuntu 安装建议

- 下载地址：https://releases.ubuntu.com/24.04/
- 推荐下载 **Ubuntu Server 24.04 LTS**（无图形界面，资源占用最小）
- 安装时勾选：
  - `Install OpenSSH server`（开启远程 SSH）
  - 不勾选第三方专有驱动（后续手动装 NVIDIA 驱动）

### 2.2 首次登录与系统更新

```bash
# SSH 远程登录（在 Windows 机的 PowerShell 中执行）
ssh 用户名@工作站IP

# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git wget curl vim htop net-tools
```

### 2.3 确认工作站 IP 地址

```bash
ip addr show | grep "inet "
```

记下局域网 IP（如 `192.168.1.100`），后续 Windows 端调用 API 需要用到。

---

## 三、NVIDIA 驱动安装

### 3.1 查看推荐驱动版本

```bash
ubuntu-drivers devices
```

输出中会显示 `recommended` 的驱动版本。24.04 源内 4090 通常推荐 `nvidia-driver-595-open` 或更高版本（如 610），以实际输出为准。

### 3.2 安装驱动

```bash
sudo apt install -y nvidia-driver-595-open
```

> 将 `595-open` 替换为 `ubuntu-drivers devices` 输出中 `recommended` 的版本。4090（Ada 架构）的开源内核模块已稳定，vLLM 推理无问题。驱动包默认带 DKMS，内核更新后会自动重编译。

### 3.3 重启

```bash
sudo reboot
```

### 3.4 验证驱动

重启后重新 SSH 登录，执行：

```bash
nvidia-smi
```

应输出显卡信息，显示：

- `NVIDIA GeForce RTX 4090`
- `CUDA Version: 12.x`（驱动支持的最高 CUDA 版本）
- 显存 `24564 MiB`

---

## 四、CUDA Toolkit 12.4 安装

> vLLM 通过 PyTorch 自带 CUDA runtime 运行，**系统 CUDA Toolkit 为可选安装**，主要用于 `nvcc` 编译和验证。如不需要编译自定义 CUDA 扩展，可跳过本节直接到第五章。

### 4.1 下载安装包

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
```

### 4.2 运行安装程序

```bash
sudo sh cuda_12.4.1_550.54.15_linux.run
```

安装过程中：

1. 输入 `accept` 接受许可协议
2. 在组件选择界面，**取消勾选 `Driver`**（已单独安装过驱动），只保留 `CUDA Toolkit 12.4`
3. 选择 `Install` 回车

### 4.3 配置环境变量

编辑 `~/.bashrc`：

```bash
vim ~/.bashrc
```

在文件末尾添加：

```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

保存退出后生效：

```bash
source ~/.bashrc
```

### 4.4 验证 CUDA

```bash
nvcc --version
```

应输出：`release 12.4`

---

## 五、Python 虚拟环境

Ubuntu 24.04 自带 Python 3.12，直接使用即可。vLLM >= 0.6 已完整支持 Python 3.12。

### 5.1 安装 venv 模块

```bash
sudo apt install -y python3-venv python3-pip
```

> **24.04 注意**：Ubuntu 24.04 启用了 PEP 668（externally-managed-environment），禁止直接 `pip install` 到系统 Python。本文档使用虚拟环境（venv），不受此限制。切勿使用 `--break-system-packages` 强制安装到系统环境。

### 5.2 创建虚拟环境

```bash
# 创建工作目录
mkdir -p ~/vllm-workspace
cd ~/vllm-workspace

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

> 激活成功后命令行前缀出现 `(venv)`。

---

## 六、vLLM 安装

### 6.1 安装 vLLM

```bash
pip install vllm
```

> vLLM 会自动安装匹配的 PyTorch（CUDA 12.x 版），无需单独装 PyTorch。vLLM >= 0.6 完整支持 Python 3.12 + Ubuntu 24.04。
> 安装包较大（约 2-3GB），耗时较长，请耐心等待。

### 6.2 安装 modelscope（国内模型加速下载）

```bash
pip install modelscope
```

### 6.3 验证 vLLM

```bash
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

应输出版本号（如 `0.18.x` 或更高）。

---

## 七、模型下载

### 7.1 模型选择

4090 24GB 显存，推荐使用 **Qwen3.6-27B-AWQ**（社区 AWQ 4bit 量化）：

| 模型 | 架构 | 量化 | 显存占用 | 推理速度 | 推荐度 |
|---|---|---|---|---|---|
| **tclf90/Qwen3.6-27B-AWQ** | Dense | AWQ 4bit | ~15GB | 快 | 推荐（中文最佳+稳定） |
| Qwen3.6-35B-A3B MoE | MoE | AWQ 4bit | ~17-20GB | 极快 | 备选（质量更高，MoE 较新） |
| Gemma 4 31B Dense | Dense | AWQ 4bit | ~16-18GB | 中等 | 不推荐（中文能力弱于 Qwen） |

> Qwen3.6 采用 Gated Delta-Network 架构，在编码/推理/Agent 能力上全面超越 Qwen3.5，中文原生优化，是中文教学视频总结场景的最佳选择。官方仅发布 FP8 量化版（~27GB 显存，4090 跑不了），因此选用社区 AWQ 4bit 量化版 `tclf90/Qwen3.6-27B-AWQ`（5.2 万次下载，质量可靠）。27B Dense 模型在 vLLM 上支持最成熟，踩坑概率最低。

### 7.2 使用 modelscope CLI 下载（推荐）

直接用命令行下载，无需 Python 脚本：

```bash
# 确保已激活 venv 并安装了 modelscope
source ~/vllm-workspace/venv/bin/activate
pip install modelscope

# 下载模型到指定目录（--local_dir 直接下载到该目录，无缓存嵌套）
modelscope download --model tclf90/Qwen3.6-27B-AWQ \
  --local_dir ~/vllm-workspace/models/Qwen3.6-27B-AWQ
```

> 模型约 15-20GB，下载时间取决于网络速度。modelscope 国内下载较快。
>
> 如果下载报错，先确认模型名正确（`tclf90/Qwen3.6-27B-AWQ`），网络正常后重试。

### 7.3 确认模型路径

下载完成后，模型位于：

```
~/vllm-workspace/models/Qwen3.6-27B-AWQ/
```

验证目录内有 `config.json` 和 `.safetensors` 文件：

```bash
ls ~/vllm-workspace/models/Qwen3.6-27B-AWQ/
```

> 如果使用默认缓存下载（未加 `--local_dir`），模型路径会是缓存嵌套结构：
> `~/.cache/modelscope/models/tclf90--Qwen3.6-27B-AWQ/snapshots/master/`
> 用 `find ~ -name "config.json" -path "*Qwen3.6*"` 确认实际路径。

---

## 八、启动 vLLM API 服务

### 8.1 启动命令

```bash
cd ~/vllm-workspace
source venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model /home/你的用户名/vllm-workspace/models/Qwen3.6-27B-AWQ \
  --served-model-name Qwen3.6-27B-AWQ \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.97 \
  --enforce-eager
```

> **参数说明（4090 24GB + Qwen3.6-27B AWQ 实测验证可用）：**
> - `--max-model-len 8192`：上下文 8K（分段总结每段最多 3000 token，够用；16K 会 OOM）
> - `--gpu-memory-utilization 0.97`：显存利用率 97%（模型权重占 19.92GB，仅剩约 0.9GB 给 KV cache，极限配置）
> - `--enforce-eager`：禁用 CUDA graph，省 1-2GB 显存（推理速度略降，但我们对速度不敏感）
> - `--served-model-name Qwen3.6-27B-AWQ`：指定 API 调用时的模型短名称（否则要用完整路径）
> - `--dtype bfloat16`：bf16 精度（4090 支持，比 fp16 更稳定）
>
> **Qwen3.6 注意**：DeltaNet 架构使用 FP8 KV cache 会导致静默输出损坏，必须保持 `--kv-cache-dtype auto`（默认），**切勿手动设置 fp8**。
>
> **显存警告**：0.97 利用率是极限配置，推理时余量很小，可能遇到偶发 OOM。如不稳定，建议换 Qwen3.6-14B-AWQ（约 10GB，余量充足）。

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--model` | 模型的绝对路径（替换为你的实际路径） |
| `--quantization awq` | 指定 AWQ 量化格式 |
| `--host 0.0.0.0` | 监听所有网卡，允许局域网访问 |
| `--port 8000` | API 服务端口 |
| `--dtype bfloat16` | 推理精度（Qwen3.6 推荐 bf16，Ada 架构原生支持） |
| `--served-model-name Qwen3.6-27B-AWQ` | API 调用时的模型短名称（否则要用完整路径） |
| `--max-model-len 8192` | 最大上下文长度 8K（分段总结够用，16K 会 OOM） |
| `--gpu-memory-utilization 0.97` | GPU 显存使用率上限 97%（模型权重 19.92GB，极限配置） |
| `--enforce-eager` | 禁用 CUDA graph 编译，省 1-2GB 显存 |

### 8.2 启动成功标志

等待约 1-2 分钟模型加载完成，看到以下输出表示成功：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

## 九、防火墙配置

### 9.1 开放 8000 端口

```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

### 9.2 确认防火墙状态

```bash
sudo ufw status
```

应显示 `8000/tcp ALLOW Anywhere`。

---

## 十、验证 API 服务

### 10.1 本地验证（在 Ubuntu 工作站上）

新开一个 SSH 终端，执行：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-AWQ",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
    "max_tokens": 100
  }'
```

应返回 JSON 格式的模型回复。

### 10.2 远程验证（在 Windows 机的 PowerShell 中）

```powershell
curl http://工作站IP:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"Qwen3.6-27B-AWQ\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}], \"max_tokens\": 50}'
```

> 将 `工作站IP` 替换为实际 IP，如 `192.168.1.100`。
>
> 如果连接失败，检查：①Ubuntu 防火墙是否开放 8000 端口 ②vLLM 服务是否在运行 ③两台机器是否能互相 ping 通。

### 10.3 查看模型列表

```bash
curl http://localhost:8000/v1/models
```

应返回已加载的模型信息。

---

## 十一、配置开机自启（可选但推荐）

使用 systemd 管理 vLLM 服务，实现开机自动启动。

### 11.1 创建服务文件

```bash
sudo vim /etc/systemd/system/vllm.service
```

写入以下内容（**替换 `你的用户名` 和模型路径**）：

```ini
[Unit]
Description=vLLM OpenAI API Server
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/home/你的用户名/vllm-workspace
Environment="PATH=/home/你的用户名/vllm-workspace/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/你的用户名/vllm-workspace/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/你的用户名/vllm-workspace/models/Qwen3.6-27B-AWQ \
  --served-model-name Qwen3.6-27B-AWQ \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.97 \
  --enforce-eager
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

> **注意**：`PATH` 必须同时包含 venv 的 bin 和系统路径（`/usr/bin` 等）。Triton 需要调用 gcc 编译 CUDA kernel，如果 PATH 里只有 venv 路径，systemd 服务会报 `Failed to find C compiler` 错误。手动终端运行不会有这个问题，因为交互式 shell 的 PATH 是完整的。

### 11.2 启用并启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

### 11.3 查看服务状态

```bash
sudo systemctl status vllm
```

### 11.4 查看实时日志

```bash
sudo journalctl -u vllm -f
```

---

## 十二、Windows 端调用方式

vLLM 提供 OpenAI 兼容 API，Windows 主程序使用 `openai` SDK 调用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://工作站IP:8000/v1",
    api_key="not-needed"  # vLLM 本地服务不需要真实 key
)

response = client.chat.completions.create(
    model="Qwen3.6-27B-AWQ",
    messages=[
        {"role": "system", "content": "你是一个专业的教学内容总结助手。"},
        {"role": "user", "content": "请总结以下内容的精华..."}
    ],
    temperature=0.3,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

> 与 DeepSeek 云端 API 的调用方式完全一致，只需切换 `base_url` 和 `model` 参数。

---

## 十三、常见问题

### Q1: vLLM 启动报错 `CUDA out of memory`
- 降低 `--gpu-memory-utilization` 到 0.80
- 或减小 `--max-model-len` 到 16384
- 27B 模型通常不会 OOM，如出现请检查是否有其他进程占用显存

### Q2: Windows 端连接超时

- 检查 Ubuntu 防火墙：`sudo ufw status`
- 检查 vLLM 是否监听 `0.0.0.0`（不是 `127.0.0.1`）
- 两台机器互相 ping 测试连通性

### Q3: 模型下载慢或找不到 AWQ 版本
- 使用 modelscope 国内源（本文档已采用）
- 如 ModelScope 上无官方 AWQ 版本，搜索 `Qwen3.6-27B AWQ` 选择社区量化版本
- 或手动从 https://modelscope.cn/models 搜索后下载上传

### Q4: vLLM 版本与模型不兼容
- 升级 vLLM 到最新版：`pip install --upgrade vllm`
- Qwen3.6 系列需要 **vLLM >= 0.18.0**，且 transformers >= 5.3.0（模型类型识别）
- Python 3.12 需要 vLLM >= 0.6.0

### Q5: Qwen3.6 输出乱码或质量异常
- 确认未设置 `--kv-cache-dtype fp8`（DeltaNet 架构用 FP8 KV 会静默损坏）
- 确认 `--dtype bfloat16`（不要用 float16）
- 确认 vLLM >= 0.18.0

---

## 十四、硬件资源评估（基于实际配置）

| 资源 | 配置                  | 预估占用                             | 剩余       | 评估                       |
| ---- | --------------------- | ------------------------------------ | ---------- | -------------------------- |
| CPU  | i5-13600KF 14核20线程 | vLLM 数据预处理+网络 IO 占 2-4 核    | 10+ 核空闲 | 过剩                       |
| 内存 | 64GB DDR5-5600        | vLLM 进程 ~2-4GB + 系统 ~2GB         | ~58GB      | 过剩                       |
| 显存 | RTX 4090 24GB | 模型权重 ~20GB + KV cache ~2GB @ 0.92 利用率 | ~2GB | 紧张但可行，需 16K 上下文 + --enforce-eager |
| 磁盘 | 2TB+4TB+1TB 全 NVMe | 模型 ~15GB + 环境 ~5GB | 极充足 | 模型建议放 4TB Lexar 上 |
| 网络 | I226-V 2.5Gbps 有线   | 与 Windows 机通信，传输请求/响应     | —          | 充足，约 250MB/s           |

> Qwen3.6-27B AWQ 实际权重约 20GB（含未量化的视觉编码器），4090 24GB 显存较紧张。需配合 16K 上下文 + `--enforce-eager` + 0.92 显存利用率才能跑起来。Server 版无图形界面，比 Desktop 版省约 200MB 显存。

---

## 十五、Ubuntu 平台注意事项

| 问题                       | 影响                                                                                     | 应对                                                                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Secure Boot**            | NVIDIA 驱动未签名则内核加载失败，`nvidia-smi` 报错                                       | 安装驱动前进入 BIOS 关闭 Secure Boot；或安装后手动签名内核模块（较麻烦，建议直接关）                                                         |
| **内核更新后驱动失效**     | `apt upgrade` 升级内核后，第三方驱动可能无法加载                                         | 24.04 的 NVIDIA 驱动默认带 DKMS，会自动为新内核编译模块；如仍失效，执行 `sudo dpkg-reconfigure nvidia-driver-595-open`（替换为实际安装版本） |
| **PEP 668 禁止系统 pip**   | 24.04 默认禁止 `pip install` 到系统 Python，报错 `error: externally-managed-environment` | 始终使用虚拟环境（venv），本文档已遵循此规范；切勿使用 `--break-system-packages`                                                             |
| **休眠/挂起后 GPU 异常**   | 系统挂起恢复后 vLLM 可能 CUDA 报错                                                       | Server 版默认不挂起；如用 Desktop 版，在「电源设置」中关闭自动挂起                                                                           |
| **SSH 断开导致 vLLM 停止** | 前台运行 vLLM 时 SSH 断开，进程被杀                                                      | 使用 systemd 服务（第十一章）或 `nohup` / `tmux` / `screen` 运行                                                                             |
| **时间同步**               | 系统时间不准可能导致 SSL 证书验证失败（模型下载时）                                      | Ubuntu 默认启用 systemd-timesyncd，一般无需手动配置；如时间异常执行 `sudo timedatectl set-ntp true`                                          |
| **Netplan 网络配置**       | 24.04 使用 Netplan 管理网络，静态 IP 配置方式与 ifupdown 不同                            | 编辑 `/etc/netplan/` 下的 yaml 文件，执行 `sudo netplan apply` 生效；默认 DHCP 无需配置                                                      |

---

## 十六、依赖说明

vLLM 的依赖树相对简单，主要冲突点：

| 依赖                      | 说明                                                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **PyTorch**               | vLLM 安装时自动拉取匹配的 CUDA 12.x 版 PyTorch（当前为 cu121/cu124），无需手动安装；与系统 CUDA Toolkit 版本独立，互不影响 |
| **Python 3.12** | Qwen3.6 需要 vLLM >= 0.18.0 完整支持 3.12；如遇个别依赖不兼容，可降级到 Python 3.11（`sudo apt install python3.11 python3.11-venv`） |
| **numpy**                 | vLLM 要求 numpy<2.0，pip 自动解析；如手动装了 numpy 2.x 会报错，执行 `pip install "numpy<2.0"` 回退                        |
| **modelscope**            | 仅用于模型下载，与 vLLM 无依赖冲突                                                                                         |
| **与 Windows 端完全隔离** | 两台机器各自独立的 Python 环境，vLLM 端不装 FunASR/PaddleOCR，不存在跨框架冲突                                             |

---

_文档结束_
