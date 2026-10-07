"""CPU tests of the real v23 host code, extracted without importing CUDA modules.

These check launch/publication order and ticket bookkeeping, NOT GPU execution.
"""
import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "kernels/nvidia/new_windowed_panel_gemm_allreduce_v23.py"


def load_host_code():
    names = {
        "CompactStripeTaskMetaV23", "FrontierWindowedPanelGEMMARContextV23",
        "_chunk_row_range_v23", "_num_runtime_stripes_v23", "_build_chunk_schedule_v23",
        "_build_task_schedule_v23", "_build_panel_schedule_v23", "_panel_production_steps_v23",
        "_launch_windowed_chunk_panel_producer_v23", "_task_meta_v23",
        "_producer_window_panel_count_v23", "_run_whole_panel_pipeline_v23",
        "_uses_bulk_panel_handoff_v23",
        "_recursive_doubling_partners_v23",
    }
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                        type_ignores=[])
    ns = {"dataclasses": dataclasses, "triton": NS(cdiv=lambda a, b: (a + b - 1) // b)}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns


class StripeFrontierTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_host_code()

    def ctx(self, **overrides):
        args = dict(chunk_rows=1024, stripe_rows=256, frontier_chunks=2,
                    producer_order="stripe_frontier", n_bands=2, stage_slots=2,
                    num_chunks=3, max_stripes_per_chunk=4, round_base=0,
                    prev_round_last_ticket_per_slot=[0, 0], reduce_slots=[])
        args.update(overrides)
        return NS(**args)

    def test_frontier_tail_and_noop_ranges(self):
        fn = self.ns["_panel_production_steps_v23"]
        self.assertEqual(fn(self.ctx(), 0, 2500), [(0, 256, 0, 1), (256, 1024, 1, 4)])
        self.assertEqual(fn(self.ctx(), 2, 2500), [(2048, 2500, 0, 2)])
        self.assertEqual(fn(self.ctx(frontier_chunks=3), 2, 2500),
                         [(2048, 2304, 0, 1), (2304, 2500, 1, 2)])
        self.assertEqual(fn(self.ctx(frontier_chunks=0), 0, 2500), [(0, 1024, 0, 4)])
        self.assertEqual(fn(self.ctx(producer_order="logical"), 0, 2500), [(0, 1024, 0, 4)])
        self.assertEqual(fn(self.ctx(), 0, 111), [(0, 111, 0, 1)])

    def test_exact_row_and_stripe_coverage(self):
        fn = self.ns["_panel_production_steps_v23"]
        for M in (1, 255, 256, 257, 1023, 1024, 1025, 2500):
            for chunk in (256, 513, 1024):
                for stripe in (1, 127, 256):
                    for F in (0, 1, 2, 99):
                        for mode in ("logical", "stripe_frontier"):
                            ctx = self.ctx(chunk_rows=chunk, stripe_rows=stripe,
                                           frontier_chunks=F, producer_order=mode)
                            for c in range((M + chunk - 1) // chunk):
                                steps = fn(ctx, c, M)
                                rows = [r for lo, hi, _, _ in steps for r in range(lo, hi)]
                                stripes = [s for _, _, lo, hi in steps for s in range(lo, hi)]
                                self.assertEqual(rows, list(range(c * chunk, min(M, (c + 1) * chunk))))
                                self.assertEqual(stripes, list(range((len(rows) + stripe - 1) // stripe)))

    def test_real_wrapper_publishes_before_tail_launch(self):
        trace = []
        ns = self.ns
        ns["_launch_panel_rows_v23"] = lambda a, b, ctx, cfg, lo, hi, band: trace.append(("gemm", lo, hi))
        ns["_task_meta_v23"] = lambda ctx, c, b, s: NS(ticket=100 + s)
        ns["_stripe_ready_view_v23"] = lambda ctx, c, b, s: s
        ns["_set_signal_cuda"] = lambda stripe, ticket, stream: trace.append(("ready", stripe, ticket))
        ns["torch"] = NS(cuda=NS(current_stream=lambda: None))
        launch = ns["_launch_windowed_chunk_panel_producer_v23"]
        launch(NS(shape=(1024, 64)), None, self.ctx(), None, 0, 0)
        self.assertEqual(trace, [("gemm", 0, 256), ("ready", 0, 100), ("gemm", 256, 1024),
                                 ("ready", 1, 101), ("ready", 2, 102), ("ready", 3, 103)])
        trace.clear()
        launch(NS(shape=(1024, 64)), None, self.ctx(), None, 0, 0, signal_tasks=False)
        self.assertEqual(trace, [("gemm", 0, 256), ("gemm", 256, 1024)])
        trace.clear()
        launch(NS(shape=(1024, 64)), None, self.ctx(producer_order="logical"), None, 0, 0)
        self.assertEqual(trace[0], ("gemm", 0, 1024))
        self.assertEqual([item[0] for item in trace], ["gemm"] + ["ready"] * 4)

    def test_ticket_reuse_across_modes_and_variable_runtime_M(self):
        begin = self.ns["FrontierWindowedPanelGEMMARContextV23"].begin_round
        for slots in (1, 2, 7, 40):
            ctx = self.ctx(stage_slots=slots, prev_round_last_ticket_per_slot=[0] * slots)
            previous = [0] * slots
            seen = set()
            for M, mode in ((2500, "logical"), (1025, "stripe_frontier"), (2500, "stripe_frontier"), (111, "logical")):
                ctx.producer_order = mode
                begin(ctx, M)
                self.assertEqual(ctx.initial_free_ticket_per_slot, previous)
                for i, task in enumerate(ctx.current_round_tasks):
                    meta = ctx.current_round_meta[task]
                    self.assertEqual(meta.slot_id, i % slots)
                    self.assertEqual(meta.prev_free_ticket, previous[meta.slot_id])
                    self.assertNotIn(meta.ticket, seen)
                    seen.add(meta.ticket)
                    previous[meta.slot_id] = meta.ticket
                self.assertEqual(ctx.prev_round_last_ticket_per_slot, previous)
                panels = self.ns["_build_panel_schedule_v23"](ctx, M)
                expected = [(c, b, s) for c, b in panels
                            for s in range(self.ns["_num_runtime_stripes_v23"](ctx, c, M))]
                self.assertEqual(ctx.current_round_tasks, expected)

    def test_whole_panel_submission_and_bounded_lookahead(self):
        for mode in ("panel_bulk", "panel_frontier", "panel_recursive_doubling",
                     "panel_recursive_doubling_cublas", "panel_recursive_doubling_bulk_cublas"):
            for slots in (1, 2, 5):
                trace = []
                recorded = set()
                produced = []
                consumed = []
                panels = [(c, b) for c in range(3) for b in range(2)]
                ctx = self.ctx(producer_order=mode, stripe_rows=1024, max_stripes_per_chunk=1,
                               active_chunk_window=1, stage_slots=slots,
                               prev_round_last_ticket_per_slot=[0] * slots,
                               reduce_slots=[NS(done_event=s) for s in range(slots)])
                self.ns["FrontierWindowedPanelGEMMARContextV23"].begin_round(ctx, 2500)

                def wait(event):
                    self.assertIn(event, recorded, "wait must follow a recorded consume event")
                    trace.append(("wait", event))

                self.ns["torch"] = NS(cuda=NS(current_stream=lambda: NS(wait_event=wait)))

                def produce(c, b):
                    produced.append((c, b))
                    trace.append(("produce", c, b))

                def consume(c, b):
                    self.assertIn((c, b), produced)
                    recorded.add(ctx.current_round_meta[c, b, 0].slot_id)
                    consumed.append((c, b))
                    trace.append(("consume", c, b))

                self.ns["_run_whole_panel_pipeline_v23"](ctx, panels, produce, consume)
                self.assertEqual(produced, panels)
                self.assertEqual(consumed, panels)
                self.assertEqual(sum(t[0] == "wait" for t in trace), len(panels) - min(slots, 2))
                operations = [t[0] for t in trace if t[0] != "wait"]
                bulk = mode in ("panel_bulk", "panel_recursive_doubling_bulk_cublas")
                expected = (["produce", "consume"] * 6 if not bulk or slots == 1
                            else ["produce", "produce", "consume", "consume"] * 3)
                self.assertEqual(operations, expected)

    def test_recursive_doubling_bulk_control_classification(self):
        classify = self.ns["_uses_bulk_panel_handoff_v23"]
        self.assertTrue(classify(self.ctx(producer_order="panel_bulk")))
        self.assertTrue(classify(self.ctx(producer_order="panel_recursive_doubling_bulk_cublas")))
        self.assertFalse(classify(self.ctx(producer_order="panel_frontier")))
        self.assertFalse(classify(self.ctx(producer_order="panel_recursive_doubling_cublas")))

    def test_whole_panel_ticket_reuse_with_partial_tail(self):
        begin = self.ns["FrontierWindowedPanelGEMMARContextV23"].begin_round
        ctx = self.ctx(stripe_rows=1024, max_stripes_per_chunk=1)
        previous = [0, 0]
        for mode, M in (("panel_bulk", 2500), ("panel_frontier", 1025), ("logical", 2500)):
            ctx.producer_order = mode
            begin(ctx, M)
            for task in ctx.current_round_tasks:
                c, b, stripe = task
                self.assertEqual(stripe, 0)
                meta = ctx.current_round_meta[task]
                self.assertEqual(meta.prev_free_ticket, previous[meta.slot_id])
                previous[meta.slot_id] = meta.ticket
                self.assertEqual(self.ns["_panel_production_steps_v23"](ctx, c, M),
                                 [(c * 1024, min((c + 1) * 1024, M), 0, 1)])

    def test_recursive_doubling_contribution_coverage(self):
        partners = self.ns["_recursive_doubling_partners_v23"]
        for world in (1, 2, 4, 8, 16):
            schedules = [partners(rank, world) for rank in range(world)]
            contributions = [{rank} for rank in range(world)]
            for phase in range(world.bit_length() - 1):
                before = contributions
                contributions = []
                for rank in range(world):
                    peer = schedules[rank][phase]
                    self.assertEqual(schedules[peer][phase], rank)
                    self.assertTrue(before[rank].isdisjoint(before[peer]))
                    contributions.append(before[rank] | before[peer])
            self.assertEqual(contributions, [set(range(world)) for _ in range(world)])
        for rank, world in ((0, 0), (0, 3), (-1, 4), (4, 4)):
            with self.assertRaises(ValueError):
                partners(rank, world)


if __name__ == "__main__":
    unittest.main()
