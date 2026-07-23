#!/usr/bin/env python3
"""Launch three seeds at a time for the nine terminal-rung grid fits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Optional, Sequence

ARCHITECTURES = ("nsf-t06-w128", "nsf-t06-w256", "nsf-t10-w128")
SEEDS = (0, 1, 2)


def _parse_gpu_indices(value: str) -> tuple[str, ...]:
    gpu_indices = tuple(item.strip() for item in value.split(","))
    if (
        not 1 <= len(gpu_indices) <= 3
        or len(set(gpu_indices)) != len(gpu_indices)
        or not all(item.isdigit() for item in gpu_indices)
    ):
        raise ValueError(
            "terminal architecture launcher requires one to three distinct "
            "numeric GPUs"
        )
    return gpu_indices


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
    gpu_indices = _parse_gpu_indices(arguments.gpu_indices)
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
        pending_seeds = list(SEEDS)
        available_gpus = list(gpu_indices)
        active: dict[
            int, tuple[str, subprocess.Popen[bytes], Path, BinaryIO]
        ] = {}
        architecture_results = {}
        failed = False
        while pending_seeds or active:
            while pending_seeds and available_gpus and not failed:
                seed = pending_seeds.pop(0)
                gpu_index = available_gpus.pop(0)
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
                active[seed] = (gpu_index, process, log_path, stream)
            completed = []
            for seed, (gpu_index, process, log_path, stream) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                stream.close()
                architecture_results[str(seed)] = {
                    "gpu_index": gpu_index,
                    "return_code": return_code,
                    "log_path": str(log_path),
                }
                available_gpus.append(gpu_index)
                completed.append(seed)
                failed = failed or return_code != 0
            for seed in completed:
                del active[seed]
            if active and not completed:
                time.sleep(1.0)
            if failed and not active:
                break
        all_results[architecture] = architecture_results
        if failed or len(architecture_results) != len(SEEDS):
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
        "configured_concurrent_fits": len(gpu_indices),
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
