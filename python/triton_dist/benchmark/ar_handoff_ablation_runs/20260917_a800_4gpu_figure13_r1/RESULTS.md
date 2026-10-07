# Figure 13 AR handoff-policy ablation

## Scope

This is a controlled four-A800 comparison of two host submission policies for
the final whole-panel GEMM--AllReduce path. Both treatments use the same BF16
inputs, cuBLAS panel producer, logical panel order, recursive-doubling
AllReduce, active-window staging, allocation, slot lifecycle, and communication
parameters.

- `bulk`: produce all panels in the current active window before submitting
  their AllReduce consumers (`panel_recursive_doubling_bulk_cublas`).
- `frontier`: submit each panel's AllReduce consumer immediately after its
  producer (`panel_recursive_doubling_cublas`).

This comparison isolates immediate handoff. It is not an RS-style cross-owner
producer-order experiment.

## Protocol

- Hardware: 4 x NVIDIA A800-SXM4-80GB.
- Independent experimental unit: one four-rank `torchrun` launch.
- Repetitions: five launches per shape.
- Within each launch: both modes use the same tensors, pass distributed
  correctness checks, receive five warmup pairs, and then receive twenty paired
  measurements per mode.
- Ordering: the mode order alternates within every launch; the first mode is
  counterbalanced across launches.
- Response: every timing sample is the maximum latency across four ranks. Each
  launch contributes the median of twenty rank-maximum samples. The table below
  reports the median launch-level ratio and its observed min--max across five
  launches.
- Fixed parameters: `chunk_rows=stripe_rows=1024`,
  `active_chunk_window=4`, `n_bands=1`, `stage_slots=4`, `comm_lanes=2`,
  `num_comm_sms=64`, no autotuning.
- NVSHMEM symmetric heap limit: 8 GiB per PE.

## Results

| Shape `(M,N,K)` | Bulk latency (ms) | Frontier latency (ms) | Bulk / frontier | Observed launch range | Symmetric scatter per PE |
|---|---:|---:|---:|---:|---:|
| `(8192,29568,8192)` | 8.4219 | 8.5522 | 0.9726x | 0.9578--1.0049x | 0.9023 GiB |
| `(8192,49152,12288)` | 14.9292 | 14.9201 | 1.0026x | 0.9794--1.0317x | 1.5000 GiB |
| `(8192,53248,16384)` | 18.4153 | 18.4600 | 0.9979x | 0.9974--1.0114x | 1.6250 GiB |

All 15 launches completed. Both modes passed the distributed Torch-reference
check in every launch. The campaign contains 15 `summary.json` files, 60
per-rank JSON files, 15 console logs, the command for every launch, a run
ledger, a campaign manifest, `per_launch.csv`, and `aggregate.csv`.

## Interpretation boundary

The immediate-handoff treatment does not produce a stable positive end-to-end
latency effect under these fixed final-path settings. The middle and largest
shapes are effectively neutral, while the smallest shape has a median below
1.0. Therefore these data do not support adding AR to Figure 13 as another
positive frontier-speedup bar. They do support a narrower conclusion: unlike
RS cross-owner frontier ordering, AR immediate handoff alone is insufficient;
its benefit is limited by overlap/resource contention in the recursive-doubling
whole-panel path.

No plot or manuscript asset was generated or replaced from this campaign.

## Authoritative artifacts

- `aggregate.csv`: shape-level aggregation for any later approved figure.
- `per_launch.csv`: all fifteen independent launch-level medians and ratios.
- `run_ledger.csv`: completion status and return code for every launch.
- `campaign_manifest.json`: implementation hashes and environment controls.
- `<shape>/rep_<n>/summary.json`: all-rank raw paired samples and correctness.
- `<shape>/rep_<n>/rank_<r>.json`: per-rank timing records.
- `<shape>/rep_<n>/console.log`: complete child-process output.

An earlier relative-path dry run wrote duplicate preliminary JSON below the
noncanonical nested path
`python/triton_dist/python/triton_dist/benchmark/ar_handoff_ablation_runs/`.
It is excluded from all aggregation. This dated directory is the sole
authoritative campaign root.
