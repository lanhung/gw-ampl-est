#!/usr/bin/env python3
"""Promote one six-fit ablation packet after delegated review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from gwlens_mm.training.ablation_authorization import (
    build_ablation_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--release-packet",
        type=Path,
        default=Path("results/phase7/ablation_release_packet.json"),
    )
    parser.add_argument(
        "--delegated-review",
        type=Path,
        default=Path("results/phase7/ablation_training_review.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "configs/execution/phase7_terminal_ablation_training_authorization.yaml"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (root / arguments.output).resolve()
    authorization = build_ablation_authorization(
        root,
        release_packet_path=(root / arguments.release_packet).resolve(),
        delegated_review_path=(root / arguments.delegated_review).resolve(),
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(authorization, sort_keys=False))
    os.replace(temporary, output)
    print(json.dumps(authorization, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
