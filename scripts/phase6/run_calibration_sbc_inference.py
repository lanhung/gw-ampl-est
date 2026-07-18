#!/usr/bin/env python3
"""Extract reviewed calibration/SBC score artifacts from one selected seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.schema import SplitName
from gwlens_mm.training.calibration_inference import run_authorized_score_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--psd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("calibration_fit", "sbc_diagnostic"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "status": "checkpoint_score_inference_implementation_ready",
                    "score_artifact_created": False,
                    "calibration_map_fitted": False,
                    "sbc_statistical_test_executed": False,
                    "final_evaluation_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_authorized_score_inference(
        Path.cwd(),
        authorization_path=arguments.authorization,
        publication_root=arguments.publication_root,
        checkpoint_path=arguments.checkpoint,
        environment_lock_path=arguments.environment_lock,
        psd_root=arguments.psd_root,
        output_path=arguments.output,
        split=SplitName(arguments.split),
        seed=arguments.seed,
        device_name=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
