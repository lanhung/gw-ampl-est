#!/usr/bin/env python3
"""Prepare one rung once, then launch its three frozen seeds on separate GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--psd-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--rung", required=True, type=int, choices=(16384, 32768))
    parser.add_argument("--gpu-indices", default="0,1,2")
    arguments = parser.parse_args(argv)
    gpu_indices = tuple(value.strip() for value in arguments.gpu_indices.split(","))
    if len(gpu_indices) != 3 or len(set(gpu_indices)) != 3 or not all(
        value.isdigit() for value in gpu_indices
    ):
        raise ValueError("launcher requires exactly three distinct numeric GPU indices")
    script = arguments.root / "scripts/phase4/run_probe_training.py"
    common = [
        sys.executable,
        str(script),
        "--root",
        str(arguments.root),
        "--authorization",
        str(arguments.authorization),
        "--stage-a-publication",
        str(arguments.stage_a_publication),
        "--environment-lock",
        str(arguments.environment_lock),
        "--psd-root",
        str(arguments.psd_root),
        "--output-root",
        str(arguments.output_root),
        "--training-commit",
        arguments.training_commit,
        "--rung",
        str(arguments.rung),
        "--device",
        "cuda:0",
        "--execute",
    ]
    preparation_result = (
        arguments.output_root / "launcher-results" / f"rung-{arguments.rung}-preparation.json"
    )
    subprocess.run(
        common
        + [
            "--seed",
            "0",
            "--prepare-rung-only",
            "--result",
            str(preparation_result),
        ],
        check=True,
    )
    log_root = arguments.output_root / "logs" / f"rung-{arguments.rung}"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for seed, gpu_index in zip((0, 1, 2), gpu_indices):
        result_path = (
            arguments.output_root
            / "launcher-results"
            / f"rung-{arguments.rung}-seed-{seed}.json"
        )
        log_path = log_root / f"seed-{seed}.log"
        stream = log_path.open("ab")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu_index
        process = subprocess.Popen(
            common + ["--seed", str(seed), "--result", str(result_path)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((seed, gpu_index, process, log_path))
        streams.append(stream)
    return_codes = {}
    try:
        for seed, gpu_index, process, log_path in processes:
            return_codes[str(seed)] = {
                "gpu_index": gpu_index,
                "return_code": process.wait(),
                "log_path": str(log_path),
            }
    finally:
        for stream in streams:
            stream.close()
    summary = {
        "status": (
            "completed_all_three_seed_processes"
            if all(item["return_code"] == 0 for item in return_codes.values())
            else "one_or_more_seed_processes_failed"
        ),
        "rung": arguments.rung,
        "seed_processes": return_codes,
        "maximum_concurrent_fits": 3,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    summary_path = (
        arguments.output_root / "launcher-results" / f"rung-{arguments.rung}-summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed_all_three_seed_processes" else 1


if __name__ == "__main__":
    raise SystemExit(main())
