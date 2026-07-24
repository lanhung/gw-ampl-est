#!/usr/bin/env python3
"""Aggregate six reviewed ablation IID comparisons without opening data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.ablation_evaluation_runtime import (
    dry_run_ablation_evaluation_runtime,
    run_authorized_ablation_iid_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print(
            json.dumps(
                dry_run_ablation_evaluation_runtime(Path.cwd()),
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_authorized_ablation_iid_aggregate(
        Path.cwd(),
        authorization_path=arguments.authorization,
        environment_lock_path=arguments.environment_lock,
        output_path=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
