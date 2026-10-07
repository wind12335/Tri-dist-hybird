run_id=20260812_155327
shape_tag=8192x28672x8192
shapes=8192,28672,8192
nproc_per_node=4
repeats=2
dtype=bfloat16
iters=3
warmup_iters=2
autotune=False
granularity_values=4096,1024
include_rank_ready=True
include_heuristic=False
latency_metric=rank-max over ranks; per-launch values in _rankmax_ columns
aggregation=median / min--max over successful independent launches (>=5 recommended)
first_ready_ms_proxy=host-observed proxy; first_consumer_ts_is_proxy=1; do not claim device first-consumer
expected_pattern=coarse_to_fine tradeoff; look for a lowest-latency region rather than assuming a perfect symmetric U-shape
