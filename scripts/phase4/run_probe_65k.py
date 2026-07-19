#!/usr/bin/env python3
"""Plan or execute one separately authorized 65k probe fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.rung65 import run_authorized_65k_probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-publication", type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-commit")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-rung-only", action="store_true")
    return parser


def _write(path: Optional[Path], value: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.execute:
        result = {
            "status": "implementation_ready_65k_training_blocked",
            "planned_rung": 65536,
            "planned_seeds": [0, 1, 2],
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
            "extension_above_65536_authorized": False,
        }
        _write(arguments.result, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    required = (
        arguments.authorization,
        arguments.stage_a_publication,
        arguments.stage_b_publication,
        arguments.combined_publication,
        arguments.environment_lock,
        arguments.psd_root,
        arguments.output_root,
        arguments.training_commit,
        arguments.seed,
    )
    if any(value is None for value in required):
        raise ValueError("authorized 65k execution requires every identity argument")
    summary = run_authorized_65k_probe(
        arguments.root.resolve(),
        authorization_path=arguments.authorization,
        stage_a_publication_root=arguments.stage_a_publication,
        stage_b_publication_root=arguments.stage_b_publication,
        combined_publication_root=arguments.combined_publication,
        correction_publication_root=arguments.correction_publication,
        environment_lock_path=arguments.environment_lock,
        psd_root=arguments.psd_root,
        output_root=arguments.output_root,
        training_commit=arguments.training_commit,
        seed=arguments.seed,
        device_name=arguments.device,
        resume_checkpoint=arguments.resume_checkpoint,
        execute_optimizer=not arguments.prepare_rung_only,
    )
    _write(arguments.result, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
