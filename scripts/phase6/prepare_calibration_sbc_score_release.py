#!/usr/bin/env python3
"""Build a non-authorizing release packet for six calibration/SBC scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.calibration_execution_authorization import (
    build_score_inference_release_packet,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--architecture-authorization", type=Path, required=True)
    parser.add_argument("--architecture-decision", type=Path, required=True)
    parser.add_argument("--materialization-authorization", type=Path, required=True)
    parser.add_argument("--materialization-result", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--exact-wheel-test-result", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--score-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    packet = build_score_inference_release_packet(
        arguments.root,
        implementation_commit=arguments.implementation_commit,
        architecture_authorization_path=arguments.architecture_authorization,
        architecture_decision_path=arguments.architecture_decision,
        materialization_authorization_path=arguments.materialization_authorization,
        materialization_result_path=arguments.materialization_result,
        publication_root=arguments.publication_root,
        wheel_path=arguments.wheel,
        exact_wheel_test_result_path=arguments.exact_wheel_test_result,
        environment_lock_path=arguments.environment_lock,
        score_output_root=arguments.score_output_root,
        output_path=arguments.output,
    )
    _atomic_json(arguments.output, packet)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
