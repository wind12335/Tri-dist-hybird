RS Active-Window Footprint Analysis

Generated: 2026-07-01T22:15:00
Shapes: 8192x53248x16384,16384x53248x16384,32768x53248x16384
world_size/local_world_size: 4/4
dtype: bfloat16
chunk_rows: 512
target_chunks_per_rank: 2
min_chunk_rows: 512
active_chunk_window: 4
stage_slots: 4
n_bands: 2
frontier_chunks: 2

Interpretation:
- windowed_v2_*: actual bounded symmetric-memory envelope of the final frontier-windowed implementation.
- logical_no_reuse_*: same panelized design without slot reuse; isolates the active-window mechanism itself.
- legacy_old_*: actual symmetric-memory envelope of the older implementation-level baseline.

Per-shape notes:
- 8192x53248x16384: num_chunks<=active_window, so this shape does not exhibit slot reuse; it only shows absolute footprint
- 16384x53248x16384: none
- 32768x53248x16384: none
