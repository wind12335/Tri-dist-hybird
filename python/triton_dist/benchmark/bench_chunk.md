已完成，新增可执行脚本：

- [bench_chunk_allgather_gemm.py](d:\研0学习\通信策略\创新点\Triton-distributed\python\triton_dist\benchmark\bench_chunk_allgather_gemm.py)

实现要点：
- 核心路径使用你新模块：
  - `create_chunk_ag_gemm_context`
  - `chunk_ag_gemm`
  - `build_dynamic_k_schedule`
- 对照组支持：
  - `torch_ag_gemm`（all_gather + matmul）
  - 原版 `ag_gemm`（可用 `--compare_base/--no-compare_base` 控制）
- 指标输出：
  - `chunk_total`, `base_total`, `torch_total`, `torch_ag_only`, `torch_gemm_only`
  - `chunk_speedup_vs_torch`, `chunk_speedup_vs_base`
  - `chunk_overlap_ratio`
  - `prefetch_k`, `num_chunks`
- 支持：
  - `--N --K` 自定义单 shape
  - 或不写时遍历 `LAYER_CONFIGS`
  - `--dump_schedule` 打印动态 K 分片
  - `--dump_csv` 落盘 csv

我已做语法检查通过：`py_compile`。

可直接运行示例：
```bash
torchrun --nproc_per_node=2 python/triton_dist/benchmark/bench_chunk_allgather_gemm.py \
  --M 8192 --N 14336 --K 4096 --dtype bfloat16 \
  --iters 10 --warmup_iters 5 --compare_base --dump_schedule
```

如果你要，我下一步可以再补一个专门的 `bench_chunk_allgather_gemm2.py`，只保留你关心的“chunk vs 原版 ag_gemm”两条曲线，输出更干净用于画图。