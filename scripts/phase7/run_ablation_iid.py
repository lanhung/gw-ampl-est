#!/usr/bin/env python3
"""Run one reviewed calibrated ablation IID and paired-comparison job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.ablation_evaluation_runtime import (
    dry_run_ablation_evaluation_runtime,
    run_authorized_ablation_iid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--psd-root", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--view", choices=("gw_only", "em_only"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
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
    result = run_authorized_ablation_iid(
        Path.cwd(),
        authorization_path=arguments.authorization,
        publication_root=arguments.publication_root,
        environment_lock_path=arguments.environment_lock,
        psd_root=arguments.psd_root,
        score_output_path=arguments.score_output,
        comparison_output_path=arguments.comparison_output,
        view=arguments.view,
        seed=arguments.seed,
        device_name=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
