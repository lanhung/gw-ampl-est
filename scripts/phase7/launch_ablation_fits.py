#!/usr/bin/env python3
"""Run three seeds concurrently for each of the two frozen ablation views."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

VIEWS = ("gw_only", "em_only")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-publication", required=True, type=Path)
    parser.add_argument("--correction-publication", required=True, type=Path)
    parser.add_argument("--terminal-size-decision", required=True, type=Path)
    parser.add_argument("--selected-architecture-decision", required=True, type=Path)
    parser.add_argument("--primary-rung-preparation", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--psd-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--gpu-indices", default="0,1,2")
    arguments = parser.parse_args(argv)
    gpu_indices = tuple(value.strip() for value in arguments.gpu_indices.split(","))
    if len(gpu_indices) != 3 or len(set(gpu_indices)) != 3 or not all(
        value.isdigit() for value in gpu_indices
    ):
        raise ValueError("ablation launcher requires three distinct numeric GPUs")
    script = arguments.root / "scripts/phase7/run_ablation_fit.py"
    common = [
        sys.executable,
        str(script),
        "--root",
        str(arguments.root),
        "--authorization",
        str(arguments.authorization),
        "--stage-a-publication",
        str(arguments.stage_a_publication),
        "--stage-b-publication",
        str(arguments.stage_b_publication),
        "--combined-publication",
        str(arguments.combined_publication),
        "--correction-publication",
        str(arguments.correction_publication),
        "--terminal-size-decision",
        str(arguments.terminal_size_decision),
        "--selected-architecture-decision",
        str(arguments.selected_architecture_decision),
        "--primary-rung-preparation",
        str(arguments.primary_rung_preparation),
        "--environment-lock",
        str(arguments.environment_lock),
        "--psd-root",
        str(arguments.psd_root),
        "--output-root",
        str(arguments.output_root),
        "--training-commit",
        arguments.training_commit,
        "--device",
        "cuda:0",
        "--execute",
    ]
    log_root = arguments.output_root / "logs"
    result_root = arguments.output_root / "launcher-results"
    all_results = {}
    for view in VIEWS:
        processes = []
        streams = []
        for seed, gpu_index in zip((0, 1, 2), gpu_indices):
            log_path = log_root / view / f"seed-{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("ab")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu_index
            result_path = result_root / view / f"seed-{seed}.json"
            process = subprocess.Popen(
                common
                + [
                    "--view",
                    view,
                    "--seed",
                    str(seed),
                    "--result",
                    str(result_path),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append((seed, gpu_index, process, log_path))
            streams.append(stream)
        try:
            view_results = {}
            for seed, gpu_index, process, log_path in processes:
                view_results[str(seed)] = {
                    "gpu_index": gpu_index,
                    "return_code": process.wait(),
                    "log_path": str(log_path),
                }
        finally:
            for stream in streams:
                stream.close()
        all_results[view] = view_results
        if any(value["return_code"] != 0 for value in view_results.values()):
            break
    complete = len(all_results) == 2 and all(
        value["return_code"] == 0
        for view in all_results.values()
        for value in view.values()
    )
    summary = {
        "status": "completed_six_ablation_fits" if complete else "ablation_fit_failed",
        "views": all_results,
        "maximum_concurrent_fits": 3,
        "maximum_total_fits": 6,
        "additional_architecture_tuning_performed": False,
        "calibration_or_sbc_accessed": False,
        "final_evaluation_accessed": False,
    }
    summary_path = result_root / "ablation-launcher-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
