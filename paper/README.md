# paper/ — AnchorOverlap 论文工作区

本目录集中存放 EuroSys 2027 投稿（AnchorOverlap）的全部论文材料。整理日期 2026-10-07。

## 目录说明

| 目录 | 内容 | 是否入 git |
|---|---|---|
| `submission/` | **最终投稿版**。`AnchorOverlap_EN_main.pdf`（15 页主稿：12 页正文 + 3 页文献）、`anchoroverlap-supplementary.pdf`（2 页技术附录）、`AnchorOverlap_EN_12p_Overleaf.zip`（Overleaf 源码包，pdfLaTeX）、`AnchorOverlap_ZH_审阅版.pdf`（终稿中文对照）、`EuroSys_27___Yichen.pdf`（投稿系统回执版） | ✅ 入库 |
| `reviews/` | 三份导师批注 PDF（EuroSys_26 第一轮、EUROSYS27 第二轮、AnchorOverlap_ZH 第三轮） | ❌ gitignore |
| `figures/` | 论文图资产（14 张正式图 + 绘图脚本 + 重绘版本） | ❌ gitignore |
| `working-logs/` | 原 `python/triton_dist/plan/` 全部工作日志、任务包、备份（686MB） | ❌ gitignore |
| `archive/en/` | 英文稿历史版本（r7/r8/r9/advisor2 等全部 overleaf 目录） | ❌ gitignore |
| `archive/zh/` | 中文稿历史版本（含早期方法论稿、r9 重译版、compressed 版） | ❌ gitignore |
| `archive/2026-09-rounds/` | 9.24 overleaf 整轮（批注落实、去重、缩页、双版定稿、各轮报告） | ❌ gitignore |
| `archive/scratch/` | 散落残件（引用论文 PDF、草稿 SVG、根目录 tex 编译残渣、游离章节） | ❌ gitignore |

## 快速指引

- **要投稿/复现**：用 `submission/` 里的 zip 上传 Overleaf（Main document = main.tex，Compiler = pdfLaTeX）。
- **要查导师意见落实**：看 `archive/2026-09-rounds/9.24 overleaf/` 里各 `*_report_zh.md` 与 `working-logs/progress.md`。
- **要改图**：图源在 `figures/`，重绘版本带日期子目录。

## 迁移对照（原位置 → 新位置）

| 原位置（python/triton_dist/ 下） | 新位置 |
|---|---|
| `9.24 overleaf/` | `archive/2026-09-rounds/9.24 overleaf/` |
| `overleaf_eurosys27_en_*` | `archive/en/` |
| `overleaf_eurosys27_zh_*`、`overleaf_methodology_zh*` | `archive/zh/` |
| `plan/` | `working-logs/` |
| `figures/` | `figures/` |
| 三份导师批注 PDF、`中文初稿修改及各种问题.pdf` | `reviews/` |
| `引用论文/`、`_tmp_arch_figs/`、根目录 tex/svg/log 残渣、`chapters/` | `archive/scratch/` |
| `benchmark_old/`、`layers_old/`、`models_old/`、`test_old/` | 仓库根 `archive/legacy-code/` |
| `python/`（嵌套误产物，实验 json 输出） | `archive/legacy-code/nested-python-output/` |
| 根目录 `计算通信重叠专利/`、`（慧子老师…）/` | 仓库根 `_personal/` |
