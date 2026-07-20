#!/usr/bin/env python3
"""Launch three seeds at a time for the nine terminal-rung grid fits."""

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
    for name in (
        "authorization",
        "stage-a-publication",
        "stage-b-publication",
        "combined-base-publication",
        "correction-publication",
        "train-increment-publication",
        "combined-131k-publication",
        "development-tail-publication",
        "terminal-decision",
        "probe-output-root",
        "environment-lock",
        "psd-root",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--gpu-indices", default="0,1,2")
    arguments = parser.parse_args(argv)
    gpu_indices = tuple(value.strip() for value in arguments.gpu_indices.split(","))
    if len(gpu_indices) != 3 or len(set(gpu_indices)) != 3 or not all(
        value.isdigit() for value in gpu_indices
    ):
        raise ValueError("terminal architecture launcher requires three numeric GPUs")
    script = arguments.root / "scripts/phase5/run_terminal_architecture_fit.py"
    common = [sys.executable, str(script), "--root", str(arguments.root)]
    for name in (
        "authorization",
        "stage-a-publication",
        "stage-b-publication",
        "combined-base-publication",
        "correction-publication",
        "train-increment-publication",
        "combined-131k-publication",
        "development-tail-publication",
        "terminal-decision",
        "probe-output-root",
        "environment-lock",
        "psd-root",
        "output-root",
    ):
        common.extend([f"--{name}", str(getattr(arguments, name.replace("-", "_")))])
    common.extend(
        [
            "--training-commit",
            arguments.training_commit,
            "--device",
            "cuda:0",
            "--execute",
        ]
    )
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
                [
                    *common,
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
        "status": (
            "completed_nine_terminal_architecture_fits"
            if complete
            else "terminal_architecture_fit_failed"
        ),
        "architectures": all_results,
        "maximum_concurrent_fits": 3,
        "probe_fits_retrained": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    summary_path = result_root / "terminal-architecture-launcher-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
