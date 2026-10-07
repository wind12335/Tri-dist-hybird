# AG/RS context output-reuse pre-change snapshot

This snapshot records the exact working-tree contents immediately before the
2026-09-17 AG/RS context output/workspace reuse optimization. The files are
stored as immutable Git blobs so that each kernel can be restored without
touching unrelated work.

| File | Git blob | SHA256 |
| --- | --- | --- |
| `kernels/nvidia/new_allgather_gemm.py` | `c4b417cfd0f85ebb57df2a6b57ef7b6f85b2db08` | `ce0ce350ff4e88cba4566a5ab7eb234825e0b068391d42d7f05e653e5c2c033d` |
| `kernels/nvidia/new_allgather_gemm_hurastic.py` | `30c90c1c805d1a10e96ff542632cd51e386cff83` | `ef142f48b169cfe7445e3e51061686641e9b13cec3a814215b358bec3c85ab0d` |
| `kernels/nvidia/new_allgather_gemm_tileready.py` | `1bf1746803d377795ddb5f6b6f9f014941cdd57b` | `166a00c89c5b207191d87486460383136adaf34acb8e4d04e60bd4e4bf865567` |
| `kernels/nvidia/new_3rd_v5_frontier_windowed_panel_rsgemm.py` | `101f7973b6421dc272e153ea434ba0cbd812e38a` | `40492753f6fa07b6df3e43eed7a9ff0d61d028ef1a9ec9090d474402f9b2f567` |

The snapshot deliberately includes all pre-existing working-tree changes,
including the tile-ready all-gather launcher imports in both AG kernels.

To inspect a saved file without modifying the working tree:

```bash
git cat-file blob <blob-id>
```

Restore only an explicitly selected file after inspecting both the current
file and the saved blob. Do not restore the whole directory over later work.
