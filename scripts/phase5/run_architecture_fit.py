#!/usr/bin/env python3
"""Plan or execute one separately authorized post-lock architecture fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.architecture import run_authorized_architecture_fit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-publication", type=Path)
    parser.add_argument("--terminal-decision", type=Path)
    parser.add_argument("--probe-output-root", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-commit")
    parser.add_argument(
        "--architecture",
        choices=("nsf-t06-w128", "nsf-t06-w256", "nsf-t10-w128"),
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        result = {
            "status": "implementation_ready_architecture_fits_blocked",
            "new_architecture_fit_count": 9,
            "reused_probe_fit_count": 3,
            "architecture_selection_executed": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
        }
    else:
        required = (
            arguments.authorization,
            arguments.stage_a_publication,
            arguments.stage_b_publication,
            arguments.combined_publication,
            arguments.terminal_decision,
            arguments.probe_output_root,
            arguments.environment_lock,
            arguments.psd_root,
            arguments.output_root,
            arguments.training_commit,
            arguments.architecture,
            arguments.seed,
        )
        if any(value is None for value in required):
            raise ValueError("authorized architecture fit requires every identity argument")
        result = run_authorized_architecture_fit(
            arguments.root.resolve(),
            authorization_path=arguments.authorization,
            stage_a_publication_root=arguments.stage_a_publication,
            stage_b_publication_root=arguments.stage_b_publication,
            combined_publication_root=arguments.combined_publication,
            terminal_decision_path=arguments.terminal_decision,
            probe_output_root=arguments.probe_output_root,
            environment_lock_path=arguments.environment_lock,
            psd_root=arguments.psd_root,
            output_root=arguments.output_root,
            training_commit=arguments.training_commit,
            architecture=arguments.architecture,
            seed=arguments.seed,
            device_name=arguments.device,
            resume_checkpoint=arguments.resume_checkpoint,
        )
    if arguments.result is not None:
        arguments.result.parent.mkdir(parents=True, exist_ok=True)
        arguments.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
