# AG/RS test-infrastructure pre-change snapshot

This snapshot records the exact working-tree contents immediately before the
2026-09-17 AG/RS test-infrastructure optimization.  The files are stored as
immutable Git blobs so they can be restored without touching unrelated work.

| File | Git blob | SHA256 |
| --- | --- | --- |
| `models/dense.py` | `61f5625abd0a53d7b399c3f25721a2d583069d7d` | `a2bc226b7b411f76f3024fbd613b95dbd5f6a037722b4a8f72b3682cf9fc23e2` |
| `test/nvidia/test_tp_e2e_innov_real.py` | `f6cfa5055caa360949e654736ba620e7d04566ac` | `3f8f5bbdb62df13e33cce1ff68c3dd6b370206604a76cf8523c607b23455cf9c` |
| `test/nvidia/test_tp_mlp_innov.py` | `4c0f9c0f7c920acc6737e32cceb42a3731596e28` | `7108d61f2d5bf88a1556cd18c8ed4e0c8c45b8da2581642835f5b986d4ce71c7` |
| `test/nvidia/test_tp_attn_innov.py` | `e009bba52d8d36f5a454484fc72e210797115d56` | `216fca5600e93f3686e8183c77b3e54d62ad55f359c63f8a57ad0827bd43eea1` |

To inspect a saved file without changing the working tree:

```bash
git cat-file blob <blob-id>
```

To restore one file deliberately, first inspect the target and then materialize
the corresponding blob into that exact path.  Do not restore the whole snapshot
over unrelated later work.
