#!/usr/bin/env python3
"""Prepare the terminal rung, evaluate retained fits, then launch three seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def _run_three(
    common: Sequence[str],
    *,
    gpu_indices: Sequence[str],
    result_root: Path,
    log_root: Path,
    extra: Sequence[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for seed, gpu_index in zip((0, 1, 2), gpu_indices):
        result_path = result_root / f"{label}-seed-{seed}.json"
        log_path = log_root / f"seed-{seed}.log"
        stream = log_path.open("ab")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu_index
        process = subprocess.Popen(
            [*common, "--seed", str(seed), "--result", str(result_path), *extra],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((seed, gpu_index, process, log_path))
        streams.append(stream)
    results: dict[str, dict[str, Any]] = {}
    try:
        for seed, gpu_index, process, log_path in processes:
            results[str(seed)] = {
                "gpu_index": gpu_index,
                "return_code": process.wait(),
                "log_path": str(log_path),
            }
    finally:
        for stream in streams:
            stream.close()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-base-publication", required=True, type=Path)
    parser.add_argument("--correction-publication", required=True, type=Path)
    parser.add_argument("--train-increment-publication", required=True, type=Path)
    parser.add_argument("--combined-131k-publication", required=True, type=Path)
    parser.add_argument("--development-tail-publication", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--psd-root", required=True, type=Path)
    parser.add_argument("--retained-65k-output-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--gpu-indices", default="0,1,2")
    arguments = parser.parse_args(argv)
    gpu_indices = tuple(value.strip() for value in arguments.gpu_indices.split(","))
    if len(gpu_indices) != 3 or len(set(gpu_indices)) != 3 or not all(
        value.isdigit() for value in gpu_indices
    ):
        raise ValueError("terminal launcher requires three distinct numeric GPUs")
    script = arguments.root / "scripts/phase4/run_probe_131k.py"
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
        "--combined-base-publication",
        str(arguments.combined_base_publication),
        "--correction-publication",
        str(arguments.correction_publication),
        "--train-increment-publication",
        str(arguments.train_increment_publication),
        "--combined-131k-publication",
        str(arguments.combined_131k_publication),
        "--development-tail-publication",
        str(arguments.development_tail_publication),
        "--environment-lock",
        str(arguments.environment_lock),
        "--psd-root",
        str(arguments.psd_root),
        "--retained-65k-output-root",
        str(arguments.retained_65k_output_root),
        "--output-root",
        str(arguments.output_root),
        "--training-commit",
        arguments.training_commit,
        "--device",
        "cuda:0",
        "--execute",
    ]
    results_root = arguments.output_root / "launcher-results"
    subprocess.run(
        [
            *common,
            "--seed",
            "0",
            "--prepare-rung-only",
            "--result",
            str(results_root / "rung-131072-preparation.json"),
        ],
        check=True,
    )
    retained_results = _run_three(
        common,
        gpu_indices=gpu_indices,
        result_root=results_root,
        log_root=arguments.output_root / "logs" / "terminal-tail" / "rung-65536",
        extra=("--evaluate-retained-65k-tail",),
        label="retained-65536-tail",
    )
    if not all(item["return_code"] == 0 for item in retained_results.values()):
        status = "one_or_more_retained_65k_tail_processes_failed"
        seed_results: dict[str, dict[str, Any]] = {}
    else:
        seed_results = _run_three(
            common,
            gpu_indices=gpu_indices,
            result_root=results_root,
            log_root=arguments.output_root / "logs" / "rung-131072",
            extra=(),
            label="rung-131072",
        )
        status = (
            "completed_all_terminal_probe_processes"
            if all(item["return_code"] == 0 for item in seed_results.values())
            else "one_or_more_terminal_probe_processes_failed"
        )
    summary = {
        "status": status,
        "rung": 131072,
        "retained_65k_tail_processes": retained_results,
        "seed_processes": seed_results,
        "maximum_concurrent_fits": 3,
        "architecture_selection_authorized": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    summary_path = results_root / "rung-131072-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "completed_all_terminal_probe_processes" else 1


if __name__ == "__main__":
    raise SystemExit(main())
