# AG/RS 优化与实验记录（持续更新）

> 状态：实验仍在进行，本文件是实时台账，不是最终论文结论。  
> 平台：4 × NVIDIA A800-SXM4-80GB。  
> 更新日期：2026-09-17。  
> 计时口径：除另有说明外，均为四个 rank 中最大延迟（rank-max），使用 synchronized eager；speedup = Torch rank-max latency / selected rank-max latency。

## 1. 已落盘的完整实验结果

下列每项都保存了一个 rank-0 汇总 JSON 和四个逐-rank JSON。JSON 中包含完整参数、正确性、每个 rank 的延迟以及 rank-max speedup。

| 模块 | 配置 | Torch rank-max (ms) | 优化路径 rank-max (ms) | Speedup | 正确性 | 结果目录 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Attention prefill | 新 AG + 旧 RS；记录了 tile=512 参数，但模块实际导入 rank-ready AG，tile 参数未生效 | 5.333557 | 5.142443 | **1.037164×** | 4 ranks 通过 | `attn_ag_new_rs_old/` |
| MLP | 新 AG + 旧 RS；记录了 tile=512 参数，但模块实际导入 rank-ready AG，tile 参数未生效 | 14.432768 | 15.895648 | **0.907970×** | 4 ranks 通过 | `mlp_ag_new_rs_old/` |
| MLP | 新 AG + 旧 RS；记录了 tile=1024 参数，但模块实际导入 rank-ready AG，tile 参数未生效 | 14.545472 | 14.991339 | **0.970258×** | 4 ranks 通过 | `mlp_ag_new_rs_old_tile1024/` |
| MLP | 新 AG + 旧 RS；记录了 tile=2048 参数，但模块实际导入 rank-ready AG，tile 参数未生效 | 14.587605 | 16.736576 | **0.871600×** | 4 ranks 通过 | `mlp_ag_new_rs_old_tile2048/` |
| MLP | 旧 AG + 新 RS；frontier=2，bands=2 | 14.532779 | 16.724533 | **0.868950×** | 4 ranks 通过 | `mlp_ag_old_rs_new_f2_b2/` |
| MLP | 新 AG + 新 RS；AG tile 参数未生效，RS frontier=2，bands=2 | 14.380608 | 14.842304 | **0.968893×** | 4 ranks 通过 | `mlp_ag_rs_new_eager/` |
| Attention prefill | rank-ready AG context 输出复用 + 旧 RS，5 iterations | 5.338330 | 5.152902 | **1.035985×** | 4 ranks 通过 | `attn_context_output_reuse/` |
| MLP | rank-ready AG context 输出复用 + 旧 RS，首次复测 | 14.556102 | 16.637248 | **0.874910×** | 4 ranks 通过 | `mlp_context_output_reuse_ag_only/` repeat 0 |
| MLP | rank-ready AG context 输出复用 + 旧 RS，5 warmups / 10 iterations | 14.007411 | 14.966122 | **0.935941×** | 4 ranks 通过 | `mlp_context_output_reuse_ag_only/` repeat 1 |

当前模块级最优结果：

- Attention：**1.037164×**（新 AG + 旧 RS）。
- MLP：**0.970258×**（修改前的 rank-ready 新 AG + 旧 RS 运行；当时记录的 tile=1024 参数实际未进入内核），尚未超过 1.0×。

### 1.1 模块 AG 调用路径纠正

`TP_Attn` 与 `TP_MLP` 当前实际导入：

```text
kernels/nvidia/new_allgather_gemm.py
```

而 AG 算子基准导入：

```text
kernels/nvidia/new_allgather_gemm_hurastic.py
```

模块脚本虽然把 `ag_tile_rows_per_chunk` 等参数写进 JSON，但
`TP_Attn._init_new_ag_ctx()` / `TP_MLP._init_new_ag_ctx()` 会根据实际 context
构造函数签名过滤参数；`new_allgather_gemm.py` 不接受这些 tile-ready 参数，
因此此前模块结果属于 rank-ready 新 AG，而不是 heuristic/tile-ready AG。
这些 speedup 数字本身仍然有效，但配置标签必须按实际调用路径解释，不能用于证明
tile-ready AG 的模块级收益。

## 2. 算子级证据

### 2.1 Attention 对应 AG 形状

算子入口：

```text
benchmark/bench_new_allgather_gemm_8rank.py
```

形状与数据类型：

```text
M=8192, N=10240, K=8192, dtype=bfloat16, world_size=4
```

默认启发式配置下，`M_per_rank=2048` 小于默认 `min_m_per_rank_for_tile_ready=4096`，因此 tile-ready 路径未启用。终端输出记录的各 rank speedup 为：

```text
rank 0: 1.13×
rank 1: 1.17×
rank 2: 1.11×
rank 3: 1.13×
```

按最保守 rank-max 口径约为 **1.11×**。该次算子脚本没有自动写 JSON，因此这里只作为手工登记的算子证据；后续正式复测应把完整 stdout 同步保存为日志。

### 2.2 Llama3-70B MLP 对应 RS 真实形状

此前得到约 `1.24×` 的 RS 算子配置为：

```text
M=8192, N=29568, K_global=8192, K_local=2048, world_size=4
```

Llama3-70B 四卡 MLP 的后置 GEMM–RS 实际形状为：

```text
M=8192, N=8192, K_global=28672, K_local=7168, world_size=4
```

2026-09-17 使用相同窗口参数对真实形状进行复测：

```text
chunk_rows=512
active_chunk_window=4
stage_slots=4
steady_sms=8
tail_sms=20
comm_lanes=2
n_bands=2
frontier_chunks=2
local_seed_direct=True
```

四个 rank 的结果为：

| Rank | Torch total (ms) | V2 total (ms) | Speedup | Torch GEMM (ms) | Torch RS (ms) | V2 GEMM (ms) | V2 RS (ms) | V2 internal overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.30 | 5.34 | **0.81×** | 3.63 | 0.89 | 5.13 | 1.89 | 23.91% |
| 1 | 4.32 | 4.83 | **0.89×** | 3.81 | 0.88 | 4.48 | 1.93 | 24.59% |
| 2 | 4.32 | 4.69 | **0.92×** | 3.68 | 0.87 | 4.48 | 1.87 | 26.04% |
| 3 | 4.32 | 5.30 | **0.81×** | 3.74 | 0.87 | 4.40 | 1.93 | 16.19% |

按 rank-max 延迟计算，保守 speedup 约为：

```text
max(torch_total) / max(v2_total) = 4.32 / 5.34 ≈ 0.81×
```

因此，`1.24×` 不是可直接迁移到 Llama3-70B MLP 的固定收益。真实 MLP
形状下，Torch RS 只有约 `0.87–0.89 ms`，可隐藏的通信窗口很小；当前 V2
的 GEMM 和 RS 分阶段时间都高于 Torch，内部重叠仅为 `16%–26%`，不足以抵消
frontier/window 调度与通信开销。MLP 模块级 `0.868950×` 与这一真实形状的
算子级结果一致。

## 3. 已发生但不能计作性能结果的实验

这些实验在结果写入前停滞或正确性失败，因此没有完整 JSON。它们在这里永久登记，不能被误认为“没测过”，也不能用于论文性能数字。

### 3.1 AG tile-ready 附加时间戳采集停滞

配置要点：

```text
tile_rows_per_chunk=512
min_m_per_rank_for_tile_ready=1024
```

八次 correctness 均通过，但随后停在算子脚本的 `_record_ag_timing_evidence()`。这是附加时间戳证据采集停滞，不是 AG 内核 correctness 失败。进程已人工终止；不计 speedup。

### 3.2 Attention 新 RS 正确性失败

真实 Attention 形状：

```text
M=8192, N=8192, K=8192
chunk_rows=512
active_chunk_window=4
stage_slots=4
steady_sms=8
tail_sms=20
comm_lanes=2
n_bands=1
frontier_chunks=1
```

错误明显超过 BF16 容差，例如：

```text
rank 3: max_abs_diff=13.6875, mismatch_fraction≈7.21%
rank 2: max_abs_diff=13.6015625, mismatch_fraction≈9.10%
```

因此该路径不能计 speedup。相同真实 Attention 形状的 RS 算子级脚本也未完成，已人工终止。

### 3.3 MLP 新 RS 的 CUDA Graph 捕获停滞

新 AG + 新 RS 配置启用 `--cuda_graph --graph_warmup 3` 后停在 Graph 捕获阶段，已人工终止。由于未进入结果写入阶段，没有 JSON；不计 speedup。当前不能依赖 CUDA Graph 为该 RS 路径提供有效模块级结果。

### 3.4 首次 context 复测缺少 NVSHMEM 动态库路径

第一次启动 context 复测时，非交互环境未加载 `libnvshmem_host.so.3`，在
NVSHMEM 初始化前失败，没有进入模型或内核。此后实验统一使用：

```bash
source ./scripts/setenv.sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

该次属于环境启动失败，不计 speedup。

## 4. 正确性容差

按当前实验约定：

```text
Attention BF16: atol=rtol=0.125
MLP BF16:       atol=rtol=0.25
```

已落盘的九组模块实验均在各自容差下通过四个 rank 的正确性检查。

## 5. 代码修改与恢复点

AG/RS 测试基础设施修改前的恢复清单：

```text
benchmark/backups/ag-rs-test-infra-20260917-prechange/MANIFEST.md
```

其中记录了以下文件修改前的不可变 Git blob 与 SHA256：

```text
models/dense.py
test/nvidia/test_tp_e2e_innov_real.py
test/nvidia/test_tp_mlp_innov.py
test/nvidia/test_tp_attn_innov.py
```

新增的公共测试辅助文件：

```text
test/nvidia/tp_ag_rs_innov_common.py
```

AG/RS context 输出复用修改前的内核快照：

```text
benchmark/backups/ag-rs-context-output-reuse-20260917-prechange/MANIFEST.md
```

该快照包含模块实际调用的 `new_allgather_gemm.py`、两个 AG 实验变体以及
`new_3rd_v5_frontier_windowed_panel_rsgemm.py` 的不可变 Git blob 和 SHA256。

当前测试脚本会把成功完成的实验自动保存为：

```text
<result-dir>/<test>_repeat_<id>.json
<result-dir>/<test>_repeat_<id>_rank_0.json
<result-dir>/<test>_repeat_<id>_rank_1.json
<result-dir>/<test>_repeat_<id>_rank_2.json
<result-dir>/<test>_repeat_<id>_rank_3.json
```

## 6. AR 参考结果

当前可信的 80 层 Llama3-70B Transformer-body AR-v23 结果为：

```text
Torch rank-max: 1709.58 ms
AR-v23 rank-max: 1757.87 ms
speedup: 0.97253×
```

结果目录：

```text
benchmark/ar_panel_frontier_runs/e2e_llama3_70b_ar_v23_p1_sync_20260917/
```

因此目前不能声称 AR 已获得端到端正加速。

## 7. 当前结论与论文使用边界

- 当前唯一模块级正 speedup 是 Attention 的 **1.037164×**；context 输出复用后的首次复测为 **1.035985×**，基本持平。
- MLP 当前最好是 **0.970258×**，仍需优化。
- Llama3-70B MLP 真实 RS 形状的算子级保守结果约为 **0.81×**；此前 **1.24×** 来自不同的 `N/K` 形状，不能直接外推。
- 上述结果尚未完成足够的重复性验证，不能直接替换论文中的最终 E2E 数字。
- 当前尚无完整 1/8/80 层 AG/RS 结果。
- 未包含 LM-head 的运行只能称为 Transformer-body prefill，不能称为完整模型前向 E2E。
- 只有含 LM-head、正确性通过、多个重复实验稳定且 rank-max speedup 大于 1 的结果，才可报告为 model-forward E2E；它仍不等同于服务级 TTFT。

## 8. 后续记录规则

从本文件建立后，每次实验均按以下规则保存：

1. 成功完成：保留 rank-0 汇总 JSON、四个逐-rank JSON，并在本报告追加 speedup。
2. 正确性失败：记录形状、参数、误差统计和失败原因，不计 speedup。
3. 停滞或人工终止：记录停滞阶段与原因，不把不完整计时当作结果。
4. 代码改动：先创建定向备份和校验清单，再修改并复测。
5. 全部实验结束后，将本文件整理为最终版，补齐所有命令、参数、代码差异、稳定性重复、可用于论文的结果和不可用于论文的结果。
