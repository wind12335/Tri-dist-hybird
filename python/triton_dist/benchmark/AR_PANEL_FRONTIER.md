# AR whole-panel experiment — 2026-09-16

## Purpose and exact scope

Evaluate whole-panel handoff for the existing direct-peer AllReduce. Each rank
still sends its contribution to every other rank and retains the complete output.
This does not replace AR with an RS output-owner schedule or RS+AG algorithm.

The hypothesis is that fewer handoffs and a fused local reduction can outweigh
waiting longer for a complete panel. It is an implementation hypothesis, not a
verified claim of novelty over the literature.

## Modes

- `logical`, `stripe_frontier`: existing paths, preserved.
- `panel_bulk`: whole panels, fused reduction, window-batched producer submission
  before submitting that window's consumers. A control for submission timing.
- `panel_frontier`: same panel order, allocation, and fused reduction, but submit
  each panel's consumer immediately after its producer.

Both panel modes require explicit `stripe_rows == chunk_rows`. The stripe field
is retained for buffer-layout compatibility: each panel has exactly one handoff,
with no first-stripe/tail production split. `frontier_chunks` does not affect
these modes. The main benchmark normalizes it to zero.

The word frontier here denotes immediate whole-panel handoff, NOT a new
permutation of GEMM panels or RS-style cross-owner frontier scheduling.
`panel_bulk` vs `panel_frontier` isolates host submission policy, not producer
panel order. A stripe-vs-panel comparison changes several mechanisms together.

## Runtime changes

1. Generate a whole `(row chunk, column band)` panel and publish ready once.
2. Copy its contribution to the other ranks using existing ready/free checks.
3. On each receiving rank, wait for all remote arrivals, then use the existing
   `kernel_reduce_window_slot_from_scatter_with_local_v23` in one launch. It reads
   the local contribution plus remote contributions, accumulates in FP32 and
   writes output once. The local-rank scatter slice is not read.
4. Publish free after the reduction finishes. Slot tickets remain generation
   sensitive, including reuse across drained invocations.
5. Bound local producer lookahead by
   `min(stage_slots, active_chunk_window * n_bands, total_panels)`.
   Before producing beyond this bound, enqueue a wait on the earlier local
   consumer's recorded completion event. Remote overwrite protection still uses
   the separate per-destination free ticket.

Unlike the old consumer, the fused consumer does not copy local data to output
and then read/write that output once for each remote rank. It waits for all
contributors before starting, so it gives up incremental early accumulation.
Whether this tradeoff improves latency depends on the workload.

Full local GEMM partial and final output tensors remain allocated. Only symmetric
data staging is bounded by slots; this is not a bound on total operator memory.
Panel modes currently require `drain=True` / `streaming_depth=1`; overlapping
invocations reusing the full local source tensor are not supported in this mode.
Full-mesh, single-node NVLink restrictions are unchanged.

## Equal symmetric data budget

For 4 ranks, BF16, N=49152, and two column bands:

| Mode | chunk rows | handoff rows | slots | symmetric data bytes per rank |
|---|---:|---:|---:|---:|
| stripe_frontier | 1024 | 256 | 16 | 805306368 (768 MiB) |
| panel_frontier | 1024 | 1024 | 4 | 805306368 (768 MiB) |
| panel_frontier candidate | 2048 | 2048 | 2 | 805306368 (768 MiB) |

Signal storage is separate and small, and differs with slot/task counts.
The direct-peer communication volume is unchanged by increasing the handoff size.

## GPU checks and initial measurements

Actual device: 4 × NVIDIA A800-SXM4-80GB, PyTorch 2.7.0+cu126.
These are A800 exploratory measurements, not paper A100-platform measurements.

CPU: six real-host-code tests passed (coverage, launch/publication order, panel
submission, completion wait placement, and cross-round ticket bookkeeping).

GPU: both new modes passed six changing-input rounds for M=1025, N=1030,
K=1024, chunk=512, two bands, including:

- BF16 with two slots;
- FP16 with one slot, forcing repeated reuse;
- non-divisible row and column tails in both cases.

Both modes also passed three FP32-reference rounds for M=8192, N=49152, K=12288.
The reference is FP32 GEMM plus distributed all-reduce, with elementwise and
relative-L2 checks aggregated over all ranks. This is not bitwise equivalence.

Initial fixed-GEMM-config results (20 alternating pairs, per-sample maximum
latency over ranks; no autotuning or graph replay):

| Comparison | first mode median ms | second mode median ms | ratio of medians |
|---|---:|---:|---:|
| panel_bulk vs panel_frontier | 22.81925 | 22.56950 | 1.01107 |
| stripe_frontier vs panel_frontier, equal 768 MiB data staging | 24.18360 | 22.35379 | 1.08186 |
| panel_bulk vs panel_frontier, 2048-row panels / 2 slots | 25.79450 | 26.04269 | 0.99047 |

The immediate-submission difference is small and samples fluctuate; it does not
establish a robust scheduling gain. The stripe-vs-panel result supports further
investigation of the combined change, not attribution to frontier order alone.
Relative L2 in the stripe-vs-panel comparison was approximately 0.003155 (stripe) versus
0.002373 (panel), consistent with fewer low-precision output roundings.

The 2048-row test also passed three changing-input numerical rounds. Its latency
was higher than the separately measured 1024-row candidate; this does not support
blindly enlarging AR panels. The panel-size comparison was not interleaved within
one run, so it should be treated as exploratory rather than a controlled effect.

The original main benchmark was also executed with panel_frontier, 1024-row
panels / 4 slots, 5 warmups / 20 iterations, no autotuning, correctness checking
enabled, and the optional GemmARLayer baseline explicitly disabled. Rank 0:

| Path | total ms | GEMM-only ms | AR-only ms |
|---|---:|---:|---:|
| panel_frontier | 22.611987 | 13.058353 | 21.938219 |
| Torch | 17.018481 | 9.707022 | 8.964475 |

Thus this implementation still loses to Torch (0.752631× total speedup).
The first experiment does not establish competitive AllReduce performance.
Data: `../csv/perf_new_windowed_panel_gemm_allreduce_v23_panel_frontier_4_ranks.csv`.
This CSV stores rank-0 batch-average times, unlike the paired JSON's medians of
per-sample rank maxima; do not combine their statistics as a single comparison.

Raw outputs, commands, checks, configuration, allocation, and kernel hashes:

- `ar_panel_frontier_runs/smoke_bf16_4gpu_20260916_r1.json`
- `ar_panel_frontier_runs/smoke_fp16_slot1_4gpu_20260916_r1.json`
- `ar_panel_frontier_runs/panel1024_paired_4gpu_20260916_r1.json`
- `ar_panel_frontier_runs/stripe_vs_panel1024_4gpu_20260916_r1.json`
- `ar_panel_frontier_runs/panel2048_paired_4gpu_20260916_r1.json`

## Reproduce the equal-budget comparison

From the repository root, after the other GPU jobs have finished:

```bash
source scripts/setenv.sh
NVSHMEM_SYMMETRIC_SIZE=4294967296 PYTHONUNBUFFERED=1 \
timeout --kill-after=15s 240s torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  python/triton_dist/benchmark/bench_ar_stripe_frontier_compare.py \
  --modes stripe_frontier panel_frontier --first_mode_stripe_rows 256 \
  --M 8192 --N 49152 --K 12288 --dtype bfloat16 \
  --chunk_rows 1024 --stripe_rows 1024 --n_bands 2 \
  --active_chunk_window 4 --stage_slots 4 --comm_lanes 2 --num_comm_sms 24 \
  --check_rounds 3 --warmup_iters 5 --iters 20 \
  --output_json python/triton_dist/benchmark/ar_panel_frontier_runs/stripe_vs_panel1024_repeat.json
```

The comparison entry refuses to overwrite JSON. The optional first-mode stripe
size scales its slot count to preserve symmetric data bytes and validates the
actual allocation after context construction. Both contexts use identical input
and GEMM configuration, and execution order alternates between samples.

To isolate submission timing, remove `--first_mode_stripe_rows 256` and use
`--modes panel_bulk panel_frontier`, with a new output filename.

## Modified implementation files

- `kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py`
- `benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py`
- `benchmark/bench_ar_stripe_frontier_compare.py`
- `benchmark/tests/test_ar_stripe_frontier.py`

No paper source, abstract, or RS kernel was changed for this experiment.
