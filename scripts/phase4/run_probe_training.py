#!/usr/bin/env python3
"""Plan or execute one separately authorized Phase 4 probe fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.contracts import load_training_stack_contract, model_configuration_hash
from gwlens_mm.training.runner import run_authorized_probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-base-publication", type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-commit")
    parser.add_argument("--rung", type=int, choices=(16384, 32768))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-rung-only", action="store_true")
    return parser


def _write_result(path: Optional[Path], result: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    authorization, model = load_training_stack_contract(root)
    result = {
        "status": "implementation_ready_scientific_training_blocked",
        "implementation_authorization_status": authorization["authorization_status"],
        "model_configuration_hash": model_configuration_hash(model),
        "planned_rungs": [16384, 32768],
        "planned_seeds": [0, 1, 2],
        "maximum_concurrent_fits": 3,
        "stage_a_accessed": False,
        "optimizer_started": False,
        "final_evaluation_accessed": False,
    }
    if not arguments.execute:
        _write_result(arguments.result, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    required = (
        arguments.authorization,
        arguments.stage_a_publication,
        arguments.environment_lock,
        arguments.psd_root,
        arguments.output_root,
        arguments.training_commit,
        arguments.rung,
        arguments.seed,
    )
    if any(value is None for value in required):
        raise ValueError("authorized execution requires every identity and data argument")
    summary = run_authorized_probe(
        root,
        authorization_path=arguments.authorization,
        stage_a_publication_root=arguments.stage_a_publication,
        stage_b_publication_root=arguments.stage_b_publication,
        combined_base_publication_root=arguments.combined_base_publication,
        correction_publication_root=arguments.correction_publication,
        environment_lock_path=arguments.environment_lock,
        psd_root=arguments.psd_root,
        output_root=arguments.output_root,
        training_commit=arguments.training_commit,
        rung_count=arguments.rung,
        seed=arguments.seed,
        device_name=arguments.device,
        resume_checkpoint=arguments.resume_checkpoint,
        execute_optimizer=not arguments.prepare_rung_only,
    )
    _write_result(arguments.result, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
