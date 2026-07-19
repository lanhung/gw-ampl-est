#!/usr/bin/env python3
"""Plan or execute one separately authorized RC.6 ablation fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.ablations import run_authorized_ablation_fit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-publication", type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--terminal-size-decision", type=Path)
    parser.add_argument("--selected-architecture-decision", type=Path)
    parser.add_argument("--primary-rung-preparation", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-commit")
    parser.add_argument("--view", choices=("gw_only", "em_only"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    result: Mapping[str, Any]
    if not arguments.execute:
        result = {
            "status": "implementation_ready_ablation_fits_blocked",
            "ablation_views": ["gw_only", "em_only"],
            "seeds": [0, 1, 2],
            "maximum_future_fits": 6,
            "scientific_data_accessed": False,
            "optimizer_started": False,
            "calibration_or_sbc_accessed": False,
            "final_evaluation_accessed": False,
        }
    else:
        required = (
            arguments.authorization,
            arguments.stage_a_publication,
            arguments.stage_b_publication,
            arguments.combined_publication,
            arguments.correction_publication,
            arguments.terminal_size_decision,
            arguments.selected_architecture_decision,
            arguments.primary_rung_preparation,
            arguments.environment_lock,
            arguments.psd_root,
            arguments.output_root,
            arguments.training_commit,
            arguments.view,
            arguments.seed,
        )
        if any(value is None for value in required):
            raise ValueError("authorized ablation fit requires every frozen identity")
        result = run_authorized_ablation_fit(
            arguments.root.resolve(),
            authorization_path=arguments.authorization,
            stage_a_publication_root=arguments.stage_a_publication,
            stage_b_publication_root=arguments.stage_b_publication,
            combined_publication_root=arguments.combined_publication,
            correction_publication_root=arguments.correction_publication,
            terminal_size_decision_path=arguments.terminal_size_decision,
            selected_architecture_decision_path=arguments.selected_architecture_decision,
            primary_rung_preparation_path=arguments.primary_rung_preparation,
            environment_lock_path=arguments.environment_lock,
            psd_root=arguments.psd_root,
            output_root=arguments.output_root,
            training_commit=arguments.training_commit,
            view=arguments.view,
            seed=arguments.seed,
            device_name=arguments.device,
            resume_checkpoint=arguments.resume_checkpoint,
        )
    if arguments.result is not None:
        arguments.result.parent.mkdir(parents=True, exist_ok=True)
        arguments.result.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
