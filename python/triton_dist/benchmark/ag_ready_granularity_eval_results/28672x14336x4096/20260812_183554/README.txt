run_id=20260812_183554
shape_tag=28672x14336x4096
shapes=28672,14336,4096
nproc_per_node=4
repeats=2
dtype=bfloat16
iters=10
warmup_iters=5
autotune=False
granularity_values=2048,1024,512,256
include_rank_ready=True
include_heuristic=False
latency_metric=rank-max over ranks; per-launch values in _rankmax_ columns
aggregation=median / min--max over successful independent launches (>=5 recommended)
first_ready_ms_proxy=host-observed proxy; first_consumer_ts_is_proxy=1; do not claim device first-consumer
expected_pattern=coarse_to_fine tradeoff; look for a lowest-latency region rather than assuming a perfect symmetric U-shape
