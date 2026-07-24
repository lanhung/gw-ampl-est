#!/usr/bin/env python3
"""Assemble the non-authorizing RC.8 ablation-calibration release packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.ablation_evaluation_authorization import (
    build_ablation_calibration_release_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--ablation-training-authorization", type=Path, required=True)
    parser.add_argument("--primary-score-authorization", type=Path, required=True)
    parser.add_argument(
        "--primary-statistics-authorization", type=Path, required=True
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-test-result", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--calibration-output-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase7/ablation_calibration_release_packet.json"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (root / arguments.output).resolve()
    packet = build_ablation_calibration_release_packet(
        root,
        implementation_commit=arguments.implementation_commit,
        ablation_training_authorization_path=(
            arguments.ablation_training_authorization.resolve()
        ),
        primary_score_authorization_path=(
            arguments.primary_score_authorization.resolve()
        ),
        primary_statistics_authorization_path=(
            arguments.primary_statistics_authorization.resolve()
        ),
        wheel_path=arguments.wheel.resolve(),
        exact_wheel_test_result_path=arguments.wheel_test_result.resolve(),
        environment_lock_path=arguments.environment_lock.resolve(),
        calibration_output_root=arguments.calibration_output_root.resolve(),
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
