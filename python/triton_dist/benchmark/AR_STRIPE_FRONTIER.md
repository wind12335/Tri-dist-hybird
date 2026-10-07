# AR v23 条带前沿实验版（2026-09-16）

## 当前状态

已增加可选的 `stripe_frontier` 生产路径，保留默认 `logical` 路径。
**未完成 GPU 编译、GPU 数值验证、通信无死锁验证或性能验证，不能用于宣称论文收益。**
本次未修改中英文论文、RS 实现或既有实验数据。

本会话实际环境：Python 3.12 / PyTorch 2.7.0+cu126，
`torch.cuda.is_available() == False`、`torch.cuda.device_count() == 0`，
没有 `/dev/nvidia*`；`/usr/bin/nvidia-smi` 是 0 字节文件，不能作为 GPU 状态证据。
执行以下四进程命令后，各进程均在 GPU 预检查中报告可见卡数 0 并退出，退出码 1：

```bash
timeout 30s torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  benchmark/bench_ar_stripe_frontier_compare.py --iters 2 --warmup_iters 1
```

## 为什么旧 AR 的参数没有实现真正的前沿生产

旧 `_build_chunk_schedule_v23(C, F)` 返回 `[0,...,F-1] + [F,...,C-1]`，
始终是逻辑顺序；每个面板的 GEMM 完成后才逐个发布该面板的条带就绪。
函数名包含 frontier，或设置 `frontier_chunks`，均不意味着实际优先生产了前沿。

RS 参考实现将每个输出归属 rank 段内的前 F 个 chunk 集合先算完，再计算尾部，
使用独立的 frontier/tail kernel launch 建立两阶段顺序。
AR v23 则直接向所有 peer 发送条带并在各 rank 累加，并没有相同的输出归属段。

## 此次实现的确切定义

借鉴 RS 的两阶段生产思路，AR 将消费者启动所需的**首条带**作为面板内前沿。
对于每个 band 的前 `frontier_chunks` 个 chunk：

1. 计算该面板的首条带。
2. 在同一生产流上发布首条带 ready。
3. 计算该面板剩余行。
4. 发布剩余条带 ready。

首条带不再被该面板其余行的计算阻塞。实际通信/消费能否更早开始、是否改善总延迟，
仍取决于主机提交进度、通信流、设备资源竞争和 GPU 实测。
本实现不改变逻辑面板遍历顺序，不是 RS 跨目标 rank 面板重排的直接复制。
`frontier_chunks=0`、前沿范围外的面板以及仅含一个条带的面板均保留整面板生产。
每个真正拆分的面板增加一次 GEMM launch，可能增加启动开销、降低计算利用率。

消费者任务顺序、槽位映射、generation ticket、远端写入前的 free/ready 等待、
arrival 发布以及最后一次读取后的 free 发布没有改变。
不增加对称数据暂存或 ready 数组容量；原有完整本地 GEMM partial buffer 仍然存在。

**这比较的是“整面板生产并发布”与“两阶段条带前沿生产并发布”，
同时改变生产划分和发布时机，不能标作只改变生产顺序的单变量消融。**

## 文件

- `kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py`：可选生产路径；autotune key 纳入模式。
- `benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py`：新增 `--producer_order logical|stripe_frontier`，
  搜索、复验、日志和 CSV 记录模式。CSV 文件名增加模式，避免两条路径相互覆盖。
- `benchmark/bench_ar_stripe_frontier_compare.py`：单节点多卡成对比较入口。
- `benchmark/tests/test_ar_stripe_frontier.py`：CPU host 调度测试。

## 已执行的非 GPU 检查

```bash
python -m unittest discover -s benchmark/tests -p test_ar_stripe_frontier.py -v
python -m py_compile kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py \
  benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py \
  benchmark/bench_ar_stripe_frontier_compare.py benchmark/tests/test_ar_stripe_frontier.py
git diff --check -- kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py \
  benchmark/bench_new_windowed_panel_gemm_allreduce_v23.py
```

4 项 host 测试通过：范围和退化情况；多种非整除尺寸的行/条带完整覆盖；
真实 host wrapper 的 launch/ready 调用顺序；跨模式和变长轮次的票据复用。
测试通过 AST 提取源文件中的实际 host 函数，替换 GPU 调用作顺序检查，
不是 Triton kernel 执行测试，更不验证设备端内存可见性。
Python 语法检查及指定文件的 diff 空白检查通过。

## GPU 节点上的复测

在项目的 CUDA / Triton-distributed / NVSHMEM 环境中，工作目录为
`Triton-distributed/python/triton_dist`。v23 原有 full-mesh NVLink 限制仍保留，
不要为了启动实验而绕过检查，也不要将其他互连平台结果当成论文 A100 平台结果。

先测尾块、多 band、少槽位反复复用：

```bash
timeout 300s torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  benchmark/bench_ar_stripe_frontier_compare.py \
  --M 1025 --N 1030 --K 1024 --dtype bfloat16 \
  --chunk_rows 512 --stripe_rows 128 --n_bands 2 --frontier_chunks 2 \
  --active_chunk_window 2 --stage_slots 2 --comm_lanes 2 \
  --check_rounds 6 --warmup_iters 3 --iters 10 \
  --output_json benchmark/ar_stripe_frontier_runs/smoke_bf16.json
```

之后以新输出文件名分别检查 `--dtype float16`、`--stage_slots 1`、`--n_bands 1`、
`--frontier_chunks 0/1/2`，以及单条带配置 `--stripe_rows 512`。
其中 F=0 或单条带是无操作对照，应该具有相同的生产 launch 划分。
测试脚本拒绝覆盖已有 JSON，请为复测选择新文件名。

上述正确性检查通过后，再测论文规模，例如：

```bash
timeout 600s torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  benchmark/bench_ar_stripe_frontier_compare.py \
  --M 8192 --N 49152 --K 12288 --dtype bfloat16 \
  --chunk_rows 1024 --stripe_rows 256 --n_bands 2 --frontier_chunks 2 \
  --active_chunk_window 4 --stage_slots 16 --comm_lanes 2 --num_comm_sms 24 \
  --check_rounds 4 --warmup_iters 5 --iters 30 \
  --output_json benchmark/ar_stripe_frontier_runs/8192_49152_12288_r1.json
```

该脚本在同一 context 上使用相同输入和固定 GEMM 配置，关闭 autotune；
每轮完整 drain，交替执行 logical/frontier 与 frontier/logical；
记录每次采样的各 rank 最大延迟、两路径中位数和成对加速比。
数值检查每轮更换输入，参考值为 FP32 GEMM 后 all-reduce，
同时要求逐元素误差和相对 L2 误差合格，所有 rank 汇总判定。
BF16 阈值为 atol=0.05、rtol=0.03、relative-L2≤0.02；
FP16 阈值为 atol=0.008、rtol=0.004、relative-L2≤0.003。
固定配置对照不是各路径最佳调优性能；也没有覆盖 `drain=False` 的多轮 streaming。
中位数和原始样本是描述性结果，单批结果不能证明稳定加速。

若沿用原 benchmark，只需在原命令末尾添加：

```bash
--producer_order stripe_frontier
```

切回旧路径使用 `--producer_order logical`。历史命令不添加选项时仍运行旧路径。
新旧模式的 autotune cache 分离；做受控比较请优先用成对测试入口。
