
已按你的思路落地了可执行原型代码（intra-node 优先）：

1. 新增 [chunk_allgather.py](d:\研0学习\通信策略\创新点\Triton-distributed\python\triton_dist\kernels\nvidia\chunk_allgather.py)
- `ChunkKSchedule`：K 分片计划结构。
- `build_dynamic_k_schedule(...)`：基于代价模型动态选 prefetch chunk 大小与剩余 chunk 顺序（支持 `rank_swizzle / in_order / largest_first`）。
- `launch_chunked_allgather_intra_node(...)`：把 `local_copy_and_barrier_all + cp_engine_producer_all_gather_intra_node` 放到通信流异步执行，便于和本地计算重叠。

2. 新增 [chunk_allgather_gemm.py](d:\研0学习\通信策略\创新点\Triton-distributed\python\triton_dist\kernels\nvidia\chunk_allgather_gemm.py)
- `create_chunk_ag_gemm_context(...)`：创建 chunk 版本上下文（复用原 `create_ag_gemm_context`）。
- `chunk_ag_gemm(...)`：实现
  - 先异步发起 copy+AG；
  - 当前流先算本地 prefetch K chunk；
  - AG 完成后补远端 prefetch chunk；
  - 再按剩余 chunk 顺序累计完成全量 GEMM。
- 目前显式限制为 intra-node（多机会抛 `NotImplementedError`），符合你当前 A100-NVLink 场景。

3. 导出接口
- 更新 [__init__.py](d:\研0学习\通信策略\创新点\Triton-distributed\python\triton_dist\kernels\nvidia\__init__.py)
- 新增导出：
  - `ChunkKSchedule`
  - `build_dynamic_k_schedule`
  - `launch_chunked_allgather_intra_node`
  - `create_chunk_ag_gemm_context`
  - `chunk_ag_gemm`

4. 校验
- 对以上 3 个文件做了 `py_compile`，语法通过。

最小调用示例（后续你可接到 benchmark）：
```python
from triton_dist.kernels.nvidia import create_chunk_ag_gemm_context, chunk_ag_gemm

ctx = create_chunk_ag_gemm_context(
    max_M=M,
    N=N,
    K=K,
    dtype=A.dtype,
    rank=rank,
    num_ranks=world_size,
    num_local_ranks=local_world_size,
)
C = chunk_ag_gemm(A, B, ctx)  # A:[M/world,K], B:[K,N/world], C:[M,N/world]
```

下一步建议：
1. 我可以直接给你补一个 `bench_chunk_allgather_gemm.py`，与 `bench_allgather_gemm.py` 同风格对比 `ag_gemm` vs `chunk_ag_gemm`。  
2. 我可以再加一个 `--dump_schedule` 开关，把每次动态 K 分片结果打印/写 CSV，便于论文里展示“代价模型决策”。









下面按“函数职责 -> 公式 -> 代码路径”拆开讲。

**总体思路**
- 你这版实现的核心是：先异步启动 `copy+barrier+allgather`，同时在计算流上先算一段本地 `K` 分片；AG 完成后补齐远端行，再算剩余 `K` 分片。
- 入口函数是 `chunk_ag_gemm(...)`，在 [chunk_allgather_gemm.py:99](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather_gemm.py:99)。

---

**chunk_allgather.py 主要函数**

- `_clamp`, `_round_up`  
  作用：边界裁剪和对齐（`K` 分片按 `k_alignment` 对齐）。  
  位置：[chunk_allgather.py:35](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:35)

- `estimate_copy_and_barrier_ms(...)`  
  作用：估算 copy+barrier 时间，用于决定预取 chunk 大小。  
  公式：  
  `nbytes = M_per_rank * K * dtype_size * (num_ranks - 1)`  
  `t_copy_ms = nbytes / (BW_GBps * 1e9) * 1e3`  
  位置：[chunk_allgather.py:45](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:45)

- `estimate_gemm_ms(...)`  
  作用：标准 GEMM 时间估算函数。  
  公式：  
  `FLOPs = 2 * M * N * K`  
  `t_gemm_ms = FLOPs / (TFLOPS * 1e12) * 1e3`  
  位置：[chunk_allgather.py:52](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:52)

- `ChunkKSchedule`  
  作用：保存分片计划：
  - `chunks`: `[(k_start, k_end), ...]`
  - `prefetch_chunk_id`: 先算哪一片
  - `remaining_chunk_order`: 后续顺序（可重排）  
  位置：[chunk_allgather.py:59](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:59)

- `build_dynamic_k_schedule(...)`  
  作用：代价模型驱动的动态 K 分片。  
  关键步骤：
  1. 估计 `copy_ms`
  2. 设目标重叠时间 `target_prefetch_ms = overlap_target * copy_ms`
  3. 反推首片 K：
     `ideal_prefetch_k = target_prefetch_ms * gemm_tflops * 1e9 / (2*M_per_rank*N_per_rank)`
  4. 对齐并裁剪得到 `prefetch_k`
  5. 剩余 `K` 按最多 `max_k_chunks` 切分
  6. 通过 `reorder_policy` 生成后续顺序（`rank_swizzle/in_order/largest_first`）  
  位置：[chunk_allgather.py:78](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:78)

- `launch_chunked_allgather_intra_node(...)`  
  作用：在通信流中异步做两步：
  1. `local_copy_and_barrier_all(...)`
  2. `cp_engine_producer_all_gather_intra_node(...)`  
  并返回通信流，供计算流 `wait_stream`。  
  位置：[chunk_allgather.py:132](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather.py:132)

---

**chunk_allgather_gemm.py 主要函数**

- `ChunkAllGatherGEMMContext`  
  作用：把原 `AllGatherGEMMTensorParallelContext` 包一层，并增加 K 分片策略参数（`k_alignment/min_prefetch_k/max_k_chunks/...`）。  
  位置：[chunk_allgather_gemm.py:35](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather_gemm.py:35)

- `create_chunk_ag_gemm_context(...)`  
  作用：先调用原 `create_ag_gemm_context`，再附加 chunk 策略。  
  位置：[chunk_allgather_gemm.py:52](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather_gemm.py:52)

- `_accumulate_chunk(c, a, b, ks, ke)`  
  作用：分片累加核心：  
  `C += A[:, ks:ke] @ B[ks:ke, :]`  
  位置：[chunk_allgather_gemm.py:93](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather_gemm.py:93)

- `chunk_ag_gemm(...)`（主流程）  
  作用：实现你图里的重叠逻辑。  
  流程：
  1. 校验 shape/dtype，只支持 intra-node。  
  2. 若没给 schedule，调用 `build_dynamic_k_schedule`。  
  3. 异步启动 AG（通信流）。  
  4. 先算本地行的 prefetch chunk（计算流）。  
  5. 等 AG 完成，取 `A_full = symm_workspace`。  
  6. 补算远端行的 prefetch chunk。  
  7. 按 `remaining_chunk_order` 算剩余 chunk。  
  8. 返回 `C[M, N_per_rank]`。  
  位置：[chunk_allgather_gemm.py:99](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/chunk_allgather_gemm.py:99)

---

**代码构成与依赖关系**
- `chunk_allgather.py` 负责“调度与通信启动”。
- `chunk_allgather_gemm.py` 负责“计算编排与结果累加”。
- 两者复用原框架：
  - `local_copy_and_barrier_all` 与 `cp_engine_producer_all_gather_intra_node`（来自原 AG 通信路径）
  - `create_ag_gemm_context`（复用原上下文和对称内存管理）
- 导出在 [__init__.py:47](d:/研0学习/通信策略/创新点/Triton-distributed/python/triton_dist/kernels/nvidia/__init__.py:47)。

---

**你现在这版的边界**
- 目前是 intra-node 可执行原型，多机路径显式 `NotImplemented`。
- “重排序”目前是“chunk 计算顺序重排”，不是额外数据搬运流重排。
- 计算核目前用 `torch.matmul` 做 chunk 累加（便于先验证机制），后续可替换成 Triton chunk kernel 继续压榨性能。