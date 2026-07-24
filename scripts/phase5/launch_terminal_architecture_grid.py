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
from typing import Any, BinaryIO, Mapping, Optional, Sequence

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


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"architecture result is not a JSON mapping: {path}")
    return value


def _completed_fit(
    *,
    output_root: Path,
    result_path: Path,
    architecture: str,
    seed: int,
) -> Mapping[str, Any] | None:
    run_directory = output_root / architecture / f"seed-{seed}"
    summary_path = run_directory / "run_summary.json"
    best_checkpoint = run_directory / "best.ckpt"
    existing = (result_path.exists(), summary_path.exists(), best_checkpoint.exists())
    if not any(existing):
        return None
    if not all(existing):
        return None
    result = _load_json(result_path)
    summary = _load_json(summary_path)
    if (
        result != summary
        or result.get("status")
        != "completed_terminal_architecture_fit_and_development_validation"
        or result.get("architecture_id") != architecture
        or int(result.get("seed", -1)) != seed
        or result.get("calibration_accessed") is not False
        or result.get("final_evaluation_accessed") is not False
        or result.get("extension_above_131072_authorized") is not False
    ):
        raise RuntimeError(
            f"completed architecture result identity is invalid: "
            f"{architecture} seed {seed}"
        )
    return result


def _resume_checkpoint(
    *,
    output_root: Path,
    result_path: Path,
    architecture: str,
    seed: int,
) -> Path | None:
    run_directory = output_root / architecture / f"seed-{seed}"
    if not run_directory.exists() and not result_path.exists():
        return None
    summary_path = run_directory / "run_summary.json"
    last_checkpoint = run_directory / "last.ckpt"
    if (
        run_directory.is_dir()
        and last_checkpoint.is_file()
        and not summary_path.exists()
        and not result_path.exists()
    ):
        return last_checkpoint
    raise RuntimeError(
        f"incomplete architecture output cannot be safely resumed: "
        f"{architecture} seed {seed}"
    )


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
    summary_path = result_root / "terminal-architecture-launcher-summary.json"
    if summary_path.exists():
        existing_summary = _load_json(summary_path)
        if existing_summary.get("status") == (
            "completed_nine_terminal_architecture_fits"
        ):
            for architecture in ARCHITECTURES:
                for seed in SEEDS:
                    result_path = (
                        result_root / architecture / f"seed-{seed}.json"
                    )
                    if (
                        _completed_fit(
                            output_root=arguments.output_root,
                            result_path=result_path,
                            architecture=architecture,
                            seed=seed,
                        )
                        is None
                    ):
                        raise RuntimeError(
                            "completed launcher summary has incomplete fit evidence"
                        )
            print(json.dumps(existing_summary, indent=2, sort_keys=True))
            return 0
        raise RuntimeError(
            "a failed terminal architecture launcher summary already exists"
        )
    all_results = {}
    completed_fit_reuse_count = 0
    resumed_fit_count = 0
    fresh_fit_count = 0
    for architecture in ARCHITECTURES:
        pending_seeds = []
        available_gpus = list(gpu_indices)
        active: dict[
            int, tuple[str, subprocess.Popen[bytes], Path, BinaryIO, bool]
        ] = {}
        architecture_results = {}
        failed = False
        for seed in SEEDS:
            result_path = result_root / architecture / f"seed-{seed}.json"
            completed = _completed_fit(
                output_root=arguments.output_root,
                result_path=result_path,
                architecture=architecture,
                seed=seed,
            )
            if completed is None:
                pending_seeds.append(seed)
                continue
            architecture_results[str(seed)] = {
                "gpu_index": None,
                "return_code": 0,
                "log_path": str(log_root / architecture / f"seed-{seed}.log"),
                "reused_completed_fit": True,
                "resumed_from_checkpoint": False,
            }
            completed_fit_reuse_count += 1
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
                resume_checkpoint = _resume_checkpoint(
                    output_root=arguments.output_root,
                    result_path=result_path,
                    architecture=architecture,
                    seed=seed,
                )
                command = [
                    *common,
                    "--architecture",
                    architecture,
                    "--seed",
                    str(seed),
                    "--result",
                    str(result_path),
                ]
                if resume_checkpoint is not None:
                    command.extend(
                        ["--resume-checkpoint", str(resume_checkpoint)]
                    )
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active[seed] = (
                    gpu_index,
                    process,
                    log_path,
                    stream,
                    resume_checkpoint is not None,
                )
                if resume_checkpoint is None:
                    fresh_fit_count += 1
                else:
                    resumed_fit_count += 1
            completed = []
            for seed, (
                gpu_index,
                process,
                log_path,
                stream,
                resumed,
            ) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                stream.close()
                result_path = result_root / architecture / f"seed-{seed}.json"
                if return_code == 0:
                    try:
                        completed_result = _completed_fit(
                            output_root=arguments.output_root,
                            result_path=result_path,
                            architecture=architecture,
                            seed=seed,
                        )
                    except (OSError, ValueError, RuntimeError):
                        completed_result = None
                    if completed_result is None:
                        return_code = 1
                architecture_results[str(seed)] = {
                    "gpu_index": gpu_index,
                    "return_code": return_code,
                    "log_path": str(log_path),
                    "reused_completed_fit": False,
                    "resumed_from_checkpoint": resumed,
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
        "completed_fit_reuse_count": completed_fit_reuse_count,
        "resumed_fit_count": resumed_fit_count,
        "fresh_fit_count": fresh_fit_count,
        "probe_fits_retrained": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "extension_above_131072_authorized": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
