#!/usr/bin/env python3
"""Non-GPU regression tests for resumable search and active-window sweeps."""

from __future__ import annotations

import csv
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEARCH = ROOT / "benchmark/bench_3rdv5_frontier_windowed_panel_gemmrs_search.py"
SWEEP = ROOT / "benchmark/bench_active_window_sweep_driver.py"
PLOT = ROOT / "benchmark/plot_active_window_sweep_gemmrs.py"


FAKE_TORCHRUN = """#!/usr/bin/env python3
import csv
import pathlib
import sys

argv = sys.argv[1:]
if '--output_csv' in argv:
    output = pathlib.Path(argv[argv.index('--output_csv') + 1])
    window = int(argv[argv.index('--active_chunk_window') + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            'Model', 'M', 'N', 'K', 'torch_total_ms', 'windowed_total_ms',
            'torch_rank_max_total_ms', 'windowed_rank_max_total_ms',
            'windowed_symmetric_staging_gib', 'windowed_active_chunk_window',
            'windowed_num_chunks', 'windowed_stage_slots', 'windowed_comm_lanes',
            'windowed_n_bands', 'windowed_frontier_chunks',
        ])
        writer.writeheader()
        writer.writerow({
            'Model': 'custom', 'M': '8192', 'N': '29568', 'K': '8192',
            'torch_total_ms': '20.0', 'windowed_total_ms': str(10.0 + window),
            'torch_rank_max_total_ms': '20.5', 'windowed_rank_max_total_ms': str(10.5 + window),
            'windowed_symmetric_staging_gib': str(0.05 * window),
            'windowed_active_chunk_window': str(window), 'windowed_num_chunks': '8',
            'windowed_stage_slots': '4', 'windowed_comm_lanes': '2',
            'windowed_n_bands': '2', 'windowed_frontier_chunks': '2',
        })
else:
    print('Rank 0 [fake] latency (ms): torch_total=2.0, v2_total=1.0, v2_gemm_only=0.7, v2_rs_only=0.6, v2_speedup_vs_torch=2.0, v2_internal_overlap=0.23')
"""


class SearchAndWindowPreflightTest(unittest.TestCase):
    def make_fake_torchrun(self, directory: Path) -> Path:
        fake = directory / "fake_torchrun.py"
        fake.write_text(FAKE_TORCHRUN, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        return fake

    def run_command(self, command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=str(cwd or ROOT), text=True, capture_output=True, check=True)

    def test_exhaustive_checkpoint_resumes_completed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            checkpoint = temp / "exhaustive_checkpoint.csv"
            base = [
                sys.executable,
                str(SEARCH),
                "--nproc_per_node", "4", "--M", "8192", "--N", "29568", "--K", "8192",
                "--search_chunk_rows_list", "512",
                "--search_active_chunk_window_list", "4",
                "--search_stage_slots_list", "2",
                "--search_steady_sms_list", "4,6",
                "--search_tail_sms_list", "12",
                "--search_comm_lanes_list", "1",
                "--search_n_bands_list", "1",
                "--search_frontier_chunks_list", "1",
                "--search_strategy", "exhaustive",
                "--exhaustive_checkpoint_csv", str(checkpoint),
            ]
            fake = self.make_fake_torchrun(temp)
            command = base + [
                "--torchrun_bin", str(fake), "--fast_iters", "1", "--fast_warmup_iters", "0",
                "--iters", "1", "--warmup_iters", "0", "--verify_topk", "1",
            ]
            first = self.run_command(command)
            self.assertIn("exhaustive search: 2 candidates", first.stdout)
            with checkpoint.open(newline="", encoding="utf-8") as fin:
                self.assertEqual(len(list(csv.DictReader(fin))), 2)

            second = self.run_command(command)
            self.assertIn("exhaustive checkpoint: 2/2 completed", second.stdout)
            self.assertEqual(second.stdout.count("[search][coarse] resume: retain candidate"), 2)
            self.assertNotIn("[search][coarse] candidate 1/2:", second.stdout)

    def test_window_sweep_aggregates_fake_children_and_plotter_reads_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            fake = self.make_fake_torchrun(temp)
            output_dir = temp / "sweep"
            self.run_command([
                sys.executable,
                str(SWEEP),
                "--nproc_per_node", "4",
                "--shapes", "8192x29568x8192",
                "--active_window_values", "1,8",
                "--repetitions", "2",
                "--chunk_rows", "256",
                "--torchrun_bin", str(fake),
                "--output_dir", str(output_dir),
            ])
            with (output_dir / "run_ledger.csv").open(newline="", encoding="utf-8") as fin:
                ledger = list(csv.DictReader(fin))
            self.assertEqual(len(ledger), 4)
            self.assertTrue(all(row["completion_status"] == "completed" for row in ledger))
            with (output_dir / "active_window_sweep_aggregate.csv").open(newline="", encoding="utf-8") as fin:
                aggregate = list(csv.DictReader(fin))
            self.assertEqual(len(aggregate), 2)
            self.assertEqual(aggregate[1]["active_chunk_window"], "8")
            self.assertEqual(aggregate[1]["latency_ratio_vs_baseline"], "1.000000")
            self.run_command([
                sys.executable,
                str(PLOT),
                "--input_csv", str(output_dir / "active_window_sweep_aggregate.csv"),
                "--output_dir", str(output_dir / "plots"),
            ])
            for suffix in ("pdf", "png", "svg"):
                self.assertTrue((output_dir / "plots" / f"active_window_sweep.{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
