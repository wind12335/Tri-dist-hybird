# AR-GEMM refresh cross-check — 2026-09-17

This file preserves the exploratory runs executed while refreshing
`/root/做的实验/ar-gemm.txt`. These runs are not the source of the three updated
AnchorOverlap ranges in that file: the authoritative values there were copied
from `/root/做的实验/9.16 AR最新实验.pdf`, which used 30 measured iterations.

## Environment and common parameters

- Device: 4 × NVIDIA A800-SXM4-80GB
- Data type: BF16
- `NVSHMEM_SYMMETRIC_SIZE=8589934592` (8 GiB/PE)
- 5 warmups and 10 measured iterations
- `chunk_rows=1024`, `stripe_rows=1024`
- `active_chunk_window=4`, `n_bands=1`, `frontier_chunks=0`
- `stage_slots=4`, `comm_lanes=2`, `num_comm_sms=64`
- `streaming_depth=1`, no autotune, GemmARLayer baseline disabled
- Correctness checking remained enabled.

## Results

The reported range is the minimum--maximum `speedup_vs_torch` printed across
the four ranks in one launch.

| Producer path | Shape `(M,N,K)` | Per-rank speedups | Range |
|---|---:|---|---:|
| `panel_recursive_doubling` | `(8192,29568,8192)` | 1.0291, 1.0292, 1.0286, 1.0286 | 1.0286--1.0292× |
| `panel_recursive_doubling` | `(8192,53248,16384)` | 0.9914, 0.9914, 0.9915, 0.9911 | 0.9911--0.9915× |
| `panel_recursive_doubling_cublas` | `(8192,29568,8192)` | 1.0158, 1.0163, 1.0160, 1.0157 | 1.0157--1.0163× |
| `panel_recursive_doubling_cublas` | `(8192,49152,12288)` | 1.1517, 1.1518, 1.1516, 1.1516 | 1.1516--1.1518× |
| `panel_recursive_doubling_cublas` | `(8192,53248,16384)` | 1.1822, 1.1820, 1.1819, 1.1823 | 1.1819--1.1823× |
| `panel_recursive_doubling_cublas` | `(16384,29568,8192)` | 0.9554, 0.9553, 0.9553, 0.9554 | 0.9553--0.9554× |

## Command template

```bash
source /root/Triton-distributed/scripts/setenv.sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export NVSHMEM_SYMMETRIC_SIZE=8589934592
export NVSHMEM_DISABLE_CUDA_VMM=1
export PYTHONUNBUFFERED=1

timeout --kill-after=15s 300s torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py \
  --M M --N N --K K --dtype bfloat16 \
  --chunk_rows 1024 --stripe_rows 1024 \
  --active_chunk_window 4 --n_bands 1 --frontier_chunks 0 \
  --producer_order PRODUCER_PATH \
  --stage_slots 4 --comm_lanes 2 --num_comm_sms 64 \
  --streaming_depth 1 --warmup_iters 5 --iters 10 \
  --no-autotune --no-run_baseline
```

## Interpretation boundary

The last shape did not appear in `9.16 AR最新实验.pdf`; therefore its old range
was retained in `ar-gemm.txt`. The single exploratory launch above must not be
silently substituted for the older multi-run record.
