#!/usr/bin/env python3
"""Run three seeds in parallel for each of the three new architectures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ARCHITECTURES = ("nsf-t06-w128", "nsf-t06-w256", "nsf-t10-w128")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-publication", required=True, type=Path)
    parser.add_argument("--terminal-decision", required=True, type=Path)
    parser.add_argument("--probe-output-root", required=True, type=Path)
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
        raise ValueError("architecture launcher requires three distinct numeric GPUs")
    script = arguments.root / "scripts/phase5/run_architecture_fit.py"
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
        "--terminal-decision",
        str(arguments.terminal_decision),
        "--probe-output-root",
        str(arguments.probe_output_root),
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
    for architecture in ARCHITECTURES:
        processes = []
        streams = []
        for seed, gpu_index in zip((0, 1, 2), gpu_indices):
            log_path = log_root / architecture / f"seed-{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("ab")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu_index
            result_path = result_root / architecture / f"seed-{seed}.json"
            process = subprocess.Popen(
                common
                + [
                    "--architecture",
                    architecture,
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
            architecture_results = {}
            for seed, gpu_index, process, log_path in processes:
                architecture_results[str(seed)] = {
                    "gpu_index": gpu_index,
                    "return_code": process.wait(),
                    "log_path": str(log_path),
                }
        finally:
            for stream in streams:
                stream.close()
        all_results[architecture] = architecture_results
        if any(value["return_code"] != 0 for value in architecture_results.values()):
            break
    complete = len(all_results) == 3 and all(
        value["return_code"] == 0
        for architecture in all_results.values()
        for value in architecture.values()
    )
    summary = {
        "status": "completed_nine_new_architecture_fits" if complete else "architecture_fit_failed",
        "architectures": all_results,
        "maximum_concurrent_fits": 3,
        "probe_fits_retrained": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    summary_path = result_root / "architecture-launcher-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
