# AR handoff ablation pre-change snapshot

The repository was already dirty before this experiment. These Git blob IDs and
SHA-256 values preserve the exact pre-change contents without overwriting any
earlier user work.

| File | Git blob | SHA-256 |
|---|---|---|
| `kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py` | `b5c8c174020f7203a75d0c7a538615b34af3dd25` | `6908745b5963e92a8266aa851627631e157ce0573b9b3960ab59d2f03d441ebe` |
| `benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py` | `9c4417f20ef73544942706badc31678d0678373c` | `08ad18cb48730bd7247083393dbc544fd6b8179d41deb07437cd19db1b856dd8` |
| `benchmark/tests/test_ar_stripe_frontier.py` | `72a211817ea66542bfc9c10b69fc0c3df02a79f6` | `3533d471bfab5ce6970dcb987e23d23bce570b40ad5724511887063ccde40e40` |

Restore one file with, for example:

```bash
git show b5c8c174020f7203a75d0c7a538615b34af3dd25 > /tmp/restored.py
```

Review the restored file before replacing anything in the dirty worktree.
