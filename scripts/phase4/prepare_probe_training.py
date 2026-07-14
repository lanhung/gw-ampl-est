#!/usr/bin/env python3
"""Fail-closed Phase 4 probe-training planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.training.contracts import (
    TrainingGateError,
    load_training_stack_contract,
    model_configuration_hash,
    validate_scientific_training_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    authorization, model = load_training_stack_contract(root)
    result = {
        "status": "implementation_ready_scientific_training_blocked",
        "model_implementation_id": model["implementation_id"],
        "model_configuration_hash": model_configuration_hash(model),
        "implementation_authorization_status": authorization["authorization_status"],
        "scientific_probe_training_authorized": False,
        "stage_a_accessed": False,
        "final_evaluation_accessed": False,
    }
    if arguments.execute:
        if not all((arguments.authorization, arguments.stage_a_publication)):
            raise TrainingGateError("execution requires authorization and Stage A publication")
        evidence = validate_scientific_training_gate(
            root,
            authorization_path=arguments.authorization,
            stage_a_publication_root=arguments.stage_a_publication,
        )
        publication = evidence["publication"]
        result.update(
            status="ready_to_construct_authorized_training_run",
            scientific_probe_training_authorized=True,
            training_authorization_status=evidence["authorization_status"],
            stage_a_parent_manifest_sha256=publication.manifest_sha256,
        )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
