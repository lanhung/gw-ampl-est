#!/usr/bin/env python3
"""Assemble the non-authorizing six-fit ablation release packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.ablation_authorization import (
    build_ablation_release_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--architecture-authorization", type=Path, required=True)
    parser.add_argument("--architecture-decision", type=Path, required=True)
    parser.add_argument("--terminal-decision", type=Path, required=True)
    parser.add_argument("--primary-rung-preparation", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-test-result", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--ablation-output-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase7/ablation_release_packet.json"),
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    packet = build_ablation_release_packet(
        root,
        implementation_commit=arguments.implementation_commit,
        architecture_authorization_path=arguments.architecture_authorization.resolve(),
        architecture_decision_path=arguments.architecture_decision.resolve(),
        terminal_decision_path=arguments.terminal_decision.resolve(),
        primary_rung_preparation_path=arguments.primary_rung_preparation.resolve(),
        wheel_path=arguments.wheel.resolve(),
        exact_wheel_test_result_path=arguments.wheel_test_result.resolve(),
        environment_lock_path=arguments.environment_lock.resolve(),
        ablation_output_root=arguments.ablation_output_root.resolve(),
        output_path=(root / arguments.output).resolve(),
    )
    output = (root / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
