# Figure 13 R9 Replot Manifest

- Figure: AG--GEMM ready-granularity ablation (Figure 13 in English R9)
- Data type: real, previously collected aggregate measurements
- Input CSV: `benchmark/ag_ready_granularity_eval_results/ag_ready_combined.csv`
- Input SHA-256 before plotting:
  `9a62b6a8e30633427d9fc8c16cf76e7a7c8428678de0351ebd3202b28b751854`
- Plot script: `benchmark/plot_ag_ready_ablation.py`
- Script SHA-256:
  `43aa5dc85f50c470d86039ba149cdc5f2973ff73a2bb800e8c0fd99e8f069c55`
- Plotting rule: run the original script without source modification; its
  `speedup` values are read from the author-checked combined CSV.
- Requested outputs: `ag_ready_ablation.png`, `.svg`, and `.pdf` at 450 DPI
  for raster output.
- Manuscript derivative: `ag_ready_ablation_tight.pdf`, cropped only to remove
  external PDF whitespace without changing plotted marks, labels, or data.
- Manuscript copy SHA-256:
  `4b4aa7ffa22d8ca8e1e8d7ffeba730a55ec76575004658ca83af21763564aced`
- GPU work: none.
