# Evaluation Table Schemas

| Table | Purpose | Rows | Required columns | Data source | Status |
|---|---|---|---|---|---|
| Table 5-1a | Controlled A100 setup. | Hardware, OS, driver, runtime, and source state. | GPU, topology, CPU/RAM, OS/kernel, driver, CUDA, PyTorch, Triton, NCCL, NVSHMEM, commit/patch state. | `benchmark/e2e_controlled_reruns/20260901_4xa100_m3_r2/env_snapshot/`. | Rendered in the current evaluation chapter; applies only to the controlled A100 environment. |
| Table 5-1b | Integrated prefill workload and timing protocol. | TP Attention, TP MLP, Transformer-body prefill. | Model/module, TP, batch, sequence, M/shape, dtype, warmup, iterations, LM-head boundary, synchronization, aggregation boundary. | Existing runner definitions and historical prefill panel protocol. | Rendered in the current evaluation chapter; model correctness compares FP32-softmax probabilities after the LM head, whereas timing excludes the LM head. No latency or speedup value is changed. |
| Table 5-2 | Operator latency and range summary. | Operator x shape x method. | M/N/K, TP, observed speedup range or rank-max latency, range semantics, correctness, failure class. | `/root/做的实验/*-gemm.txt`; optional E1 CSV. | Current range figures are real author-side measurements; structured rank-max logs may supplement them later. |
| Table 5-3 | Search cost and selected latency. | GEMM--RS x frozen candidate space x search method. | Candidate-plan SHA-256, coarse coverage, total launches, selected tuple, verified rank-max ms, selection relation, and observed latency ratio. | Frozen plans in `benchmark/search_compare_runs/20260828_space{64,128}/`; completed checkpoints and strategy logs in the matching `_r1/` directories. | Completed on 2026-08-28: 64/64 and 128/128 exhaustive checkpoints; valid 13- and 15-launch two-stage runs. |
| Table 5-4 | Layer and Transformer-body prefill latency. | Workload x run type x mode. | Model, TP, batch, sequence, graph/eager, timing boundary, speedup, correctness boundary, and record provenance. | Historical prefill inputs; future structured E2 CSV if independently repeated. | Figure 12 uses one retained historical value per condition. Correctness includes LM head plus FP32 softmax; timing excludes the LM head and reports no cross-launch dispersion or rank-max step-latency estimate. |
| Table 5-5 | Resource and feasibility. | Shape x active-window depth x repetition aggregate. | L, logical chunks, rows, symmetric-staging footprint GiB, rank-max latency median/mean/std, completed/failed repetitions, status, failure reason. | `benchmark/active_window_sweep_runs/20260828_4xa100_active_window_multipoint_r1/{run_ledger,active_window_sweep_aggregate}.csv`. | Completed: three shapes, $L\in\{1,2,4,8\}$, five launches per point, 60/60 completed. The footprint is the RS data buffer plus arrival/free ticket arrays, excluding allocator padding, auxiliary runtime state, and total GPU memory. |

## Aggregation Contract

- Preserve all available launch records in CSV or JSONL for newly run experiments.
- Compute a published distributed latency from a rank maximum per launch when rank-wise logs are available.
- For historical range artifacts, publish the recorded midpoint and observed min--max range with its exact meaning; do not relabel it as a confidence interval.
- Retain allocation failures as categorical observations with the recorded failure class.


## Search Table Handoff

`tables/constraint_search_cost.tex` records the completed frozen 64- and
128-space comparisons. The plan SHA-256 values are
`5bd37886711a973d14443c054f5a6318cd3eccd11ebd954826d21581598c425a` and
`ac1a43387dfaaae1fb65b2f591a037df9a7b1a80022be9b721e465e639a18475`.
The raw evidence is the immutable candidate plan, exhaustive checkpoint, and
append-only strategy console in each run directory. Final rank-max values are
independent verification observations; because exhaustive validates only its
top-three coarse candidates, the table does not call either ratio regret or a
final-measurement oracle. A non-completed future exhaustive row may report only
its own completed-candidate count and status.

## 2026-09-22 — Author-confirmed evaluation corrections

Current bilingual Evaluation Table 3 has Workload / Model / Precision and timing columns. All rows name Llama3-70B and Qwen2.5-72B; detailed shape and E2E timing boundaries remain in prose.

Table 4 has Chain / BF16 tolerance / Reuse stress and pass record columns. AG atol=rtol=0.001; RS and AR atol=rtol=0.02. Author-confirmed counts: AG 121/129 launches; RS window 60/60 passed, controlled order 11/12 completed (one numerical pass followed by timing timeout); AR window 60/60 passed, controlled order 12/12 completed. AR counts and tolerance are supplied by the author in this revision, not independently reconstructed or rerun here. RS's three-shape/four-depth/five-launch construction is not attributed to AR.

Search table values remain unchanged. Its candidates all fix active_chunk_window=4, so it does not establish that search selected L=4 over other depths.
