#!/usr/bin/env python3
"""Plan or execute one terminal 131k probe/tail operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.terminal131 import (
    evaluate_retained_65k_on_terminal_tail,
    run_authorized_131k_probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-base-publication", type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--train-increment-publication", type=Path)
    parser.add_argument("--combined-131k-publication", type=Path)
    parser.add_argument("--development-tail-publication", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--retained-65k-output-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-commit")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-rung-only", action="store_true")
    parser.add_argument("--evaluate-retained-65k-tail", action="store_true")
    return parser


def _write(path: Optional[Path], value: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.execute:
        result = {
            "status": "implementation_ready_terminal_131k_training_blocked",
            "planned_rung": 131072,
            "planned_seeds": [0, 1, 2],
            "retained_65k_tail_evaluation_planned": True,
            "architecture_selection_authorized": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
            "extension_above_131072_authorized": False,
        }
        _write(arguments.result, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    required = (
        arguments.authorization,
        arguments.stage_a_publication,
        arguments.stage_b_publication,
        arguments.combined_base_publication,
        arguments.correction_publication,
        arguments.train_increment_publication,
        arguments.combined_131k_publication,
        arguments.development_tail_publication,
        arguments.environment_lock,
        arguments.psd_root,
        arguments.output_root,
        arguments.training_commit,
        arguments.seed,
    )
    if any(value is None for value in required):
        raise ValueError("authorized terminal execution requires every identity argument")
    common = {
        "authorization_path": arguments.authorization,
        "stage_a_publication_root": arguments.stage_a_publication,
        "stage_b_publication_root": arguments.stage_b_publication,
        "combined_base_publication_root": arguments.combined_base_publication,
        "correction_publication_root": arguments.correction_publication,
        "train_parent_root": arguments.train_increment_publication,
        "combined_131k_publication_root": arguments.combined_131k_publication,
        "development_tail_parent_root": arguments.development_tail_publication,
        "environment_lock_path": arguments.environment_lock,
        "psd_root": arguments.psd_root,
        "output_root": arguments.output_root,
        "training_commit": arguments.training_commit,
        "seed": arguments.seed,
        "device_name": arguments.device,
    }
    if arguments.evaluate_retained_65k_tail:
        if arguments.retained_65k_output_root is None:
            raise ValueError("retained-tail evaluation requires its 65k output root")
        summary = evaluate_retained_65k_on_terminal_tail(
            arguments.root.resolve(),
            retained_65k_output_root=arguments.retained_65k_output_root,
            **common,
        )
    else:
        summary = run_authorized_131k_probe(
            arguments.root.resolve(),
            resume_checkpoint=arguments.resume_checkpoint,
            execute_optimizer=not arguments.prepare_rung_only,
            **common,
        )
    _write(arguments.result, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
