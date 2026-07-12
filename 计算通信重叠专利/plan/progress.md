# 进度记录

## 当前阶段

- S6 Revise / 说明书公式补强与图文一致性复核

## 已完成

- 已复核权利要求书2.docx与最新说明书、摘要的一致性，并修正其中残留的旧版 AR 条款表述。
- 已将权利要求1中的“关键路径优先的分岔调度方式”收敛为更通用的“关键路径优先的推进方式”，避免主权项过早绑定 AG 专属实现。
- 已将权利要求11和权利要求12中的“条带工作单元”口径改回与图4和说明书一致的“面板工作单元/归约槽位”表述。
- 已对说明书.docx中的“附图说明”和“具体实施方式”再次复核，确认图1至图5的正文指向与当前附图语义一致。
- 已按导师示例的正文嵌入公式方式，在说明书.docx中补入并行规模、局部分块、面板构造、活动窗口映射与槽位复用等关键公式表达。
- 已将步骤1至步骤4中的关键段落压缩重写，使配置参数确定、AG-GEMM细粒度就绪、RS-GEMM面板调度和AR-GEMM窗口归约之间的逻辑关系更紧。
- 已为说明书.docx保留一份修改前备份文件到 /tmp/说明书_before_formula_20260703.docx 以便必要时回退比对。
- 已将 figure1 和 figure2 改写为纯中文短句版本，减少混合符号带来的显示风险。
- 已放大 figure1 与 figure2 中的关键框体，并将 figure4 底部箭头改为直线箭头。
- 已重建覆盖当前附图全部实际字符的中文子集字体，并重新生成5张附图。
- 已引入自动换行与字号适配逻辑，收紧文字越界风险。
- 已将 figures 目录中的图1至图5全部重绘为中文标签版本。
- 已将新的中文附图资源回写到说明书附图.docx。
- 已修订说明书中的图1至图5附图说明，使其与当前附图文件命名保持一致。
- 已修订具体实施方式中的步骤1至步骤4关键段落，并补入图1至图5对应关系。
- 已清除图4和图5的旧表述，避免说明书正文与现有附图内容错位。
- 已根据导师模板补齐说明书摘要.docx。
- 已根据现有说明书中的附图说明生成图1至图5，并写入说明书附图.docx。
- 已生成 5 张附图 PNG，分别对应统一依赖总览、AG-GEMM、RS-GEMM、AR-GEMM 和配置参数确定流程。

- 已定位导师示例目录与当前专利目录。
- 已确认当前目录下的旧题目主要分布在权利要求书2和说明书中。
- 已将权利要求书2.docx与说明书.docx中的旧发明名称统一替换为“基于细粒度就绪与活动窗口”的新题目。
- 已完成 DOCX 重新打包，并验证旧题目在两个目标文件中的出现次数均为 0。

## Capability-use audit

- Required skills: using-research-writing, paper-orchestration, brainstorming-research, writing-core
- Skills actually used: using-research-writing, paper-orchestration, brainstorming-research, writing-core
- Inputs consumed: 导师专利示例目录、计算通信重叠专利目录中的 DOCX 文件
- Inputs not used and why: 第二版初稿修改.pdf，当前机器缺少直接 PDF 文本提取工具，暂未完成正文抽取
- Artifacts produced: plan/project-overview.md, plan/outline.md, plan/progress.md
- Verification run: 说明书.docx 开包校验、word/document.xml 公式对象计数、附图说明与具体实施方式目标段落抽取核对、公式落点 XML 片段核对、权利要求书2.docx 开包校验与目标条款抽取核对
- Remaining risk: 当前环境未进行 Word 或 LibreOffice 的可视化打开校验，公式与权利要求版面的最终显示仍建议在本机图形界面中顺手目检一次
