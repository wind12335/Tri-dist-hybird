# 环境信息抓取方法与已抓取数据（M2 平台/软件表数据来源）

## 一、用什么抓的

自写的一次性只读脚本（不跑任何 benchmark、不改代码/git）：

| 平台 | 脚本 | 说明 |
|---|---|---|
| A100（NVIDIA） | `benchmark/collect_env_snapshot.py` | 抓 nvidia-smi(-q/-topo)、nvcc、lscpu、meminfo、os-release、uname、ibstat、pip 关键包、git 状态，并用 python 直接 import torch/triton 读版本 |
| K100_AI（DCU/DTK） | `benchmark/collect_env_snapshot_dcu.py` | 同结构 DCU 版：按序探测 hy-smi / rocm-smi / dcmi / xpu-smi、rocminfo、lspci；抓 DTK/ROCM/HIP/DUSHMEM/RCCL 环境变量、ldconfig 中的 librccl/libdushmem、/opt/dtk 版本文件、torch(triton+hip)、pip 包 |

两者都输出：`env_snapshot*.json`（结构化汇总）+ 若干原始 txt（每个命令的完整 stdout/stderr）。

## 二、A100 已抓取的有效数据（batch r2）

- 采集时间：2026-09-01 15:21:25（本地），4×A100 空闲状态下
- 位置：`benchmark/e2e_controlled_reruns/20260901_4xa100_m3_r2/env_snapshot/`
- `env_snapshot.json` SHA-256：`49a2caea82ff13f531dd126e8f3caef8aa3ecbccbc2a8dcf73c5224cc64315e5`
- ⚠️ 同批 `..._m3_r1/env_snapshot` 的 `nvidia-smi -q` 输出被截断，**已废弃，不要引用**

抓到的关键值（写进论文 §5.1.1 的就是这些）：

| 项 | 值 | 来源文件 |
|---|---|---|
| GPU | 4 × NVIDIA A100-SXM4-80GB，每卡 79.25 GiB 可见显存 | `nvidia_smi.txt` / `nvidia_smi_short.txt` |
| GPU 互连 | 任意 GPU 对 NV12（NVLink）；跨 NUMA SYS/NODE | `nvidia_smi_topo.txt` |
| CPU | 双路 AMD EPYC 7742（2 Socket，各 64 核） | `lscpu.txt` |
| 内存 | MemTotal ≈ 1 TiB | `meminfo.txt` |
| OS | Ubuntu 24.04.2 LTS，Linux 5.15.0-94-generic | `os_release.txt` / `uname.txt` |
| Driver | 580.65.06（驱动报告 CUDA 13.0） | `nvidia_smi.txt` |
| nvcc | 12.9 | `nvcc.txt` |
| Python | 3.12.9（/root/miniconda3） | json: python_stack |
| PyTorch | 2.7.0+cu126（CUDA runtime 12.6，cuDNN 9.5.1） | json: python_stack |
| NCCL | 2.26.2 | json: python_stack |
| Triton | 3.4.0 | json: python_stack |
| NVSHMEM | 3.3.9（pip nvidia-nvshmem-cu12） | `pip_selected.txt` |
| git | commit b3ba2b753c316c2dd834f95756184f1a61d3fdf8，`b3ba2b7-dirty` | json: git |

IB：`ibstat.txt` 为空（本机无 IB，NVLink 机内互连）。

## 三、在 K100_AI 上复刻（你要做的）

在那台机器的 DCU python 环境里、跑任何实验之前执行一次：

```bash
cd <你的 Triton-distributed 检出路径>
python3 python/triton_dist/benchmark/collect_env_snapshot_dcu.py \
    --output_dir python/triton_dist/benchmark/e2e_controlled_reruns/<批次名>/env_snapshot_k100 \
    --batch_id <批次名>
```

脚本只读不写仓库；找不到的工具（比如该 DTK 版本没有 hy-smi）会在 json 里记
`"xxx not on PATH"`，不会中断。抓完把整个 `env_snapshot_k100/` 目录拷回来即可，
论文里 K100_AI 的平台行就从这份 json 填，和 A100 同一证据等级。

注意两点：
1. RCCL 版本优先看 `ldconfig` 抓到的 `librccl` 行 / pip `rccl` 包；DUSHMEM 看 `libdushmem` 行和环境变量。
2. 这份快照只绑定"采集之后在同机同环境跑的新实验"；它不能倒填历史 K100_AI 图的环境信息（与 A100 侧同一条纪律）。

## 四、K100_AI 实测快照（已完成，2026-09-01）

- 批次：`20260901_4xk100_paperdraft`，采集于 2026-09-01 20:21（+08:00）
- 位置：`benchmark/env_snapshots/20260901_4xk100_paperdraft_r1/env_snapshot_dcu.json`
- SHA-256：`d64e82bb6adb1cd2c73d4246ea585e4b5df144fbe4e0a8f93f2d9a05ec211468`
- 抓取脚本：用户在海光侧实跑后改进的版本（DTK_ROOT 解析、hy-smi /opt/hyhal/bin 路径、
  /opt/dtk/.info 版本读取、RCCL/DUSHMEM soname 版本解析、DUSHMEM 插件清单、platform_label 段），
  已采纳为正式版 `benchmark/collect_env_snapshot_dcu.py`；实跑副本存于同目录 `..._asrun.py`。

实测关键值（已写入论文 §5.1.1）：

| 项 | 实测值 |
|---|---|
| GPU | K100_AI（gfx936），4 卡，每卡 64 GiB |
| DTK | 26.04（/opt/dtk-26.04，rocm_version 26.04） |
| DUSHMEM | 3.2.5（libdushmem_host.so.3.2.5，含 MPI/PMI/UID bootstrap 与 IBGDA/IBRC transport 插件） |
| RCCL | 2.22.3（/opt/dtk/lib/librccl.so，不在 ldconfig 缓存中） |
| Python / PyTorch | 3.11.9 / 2.7.1（+das.opt1.dtk2604.torch271 DTK 构建，HIP 6.3.26093） |
| Triton | 3.1.0（DTK 构建） |
| OS | CentOS 7（内核 3.10.0-957），容器环境 |

⚠️ 与此前凭记忆记录的差异（已按实测修正论文）：Python 3.10 → **3.11.9**；Triton 3.5.1 → **3.1.0**。
platform_label 注明：两台 DTK 环境软件栈相同，GPU 行按论文平台口径填写。
