#!/usr/bin/env python3
"""Render the frozen final-evaluation plan without generating any pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.production.final_evaluation import (
    FINAL_EVALUATION_COMMITMENT_HASH,
    NUMERICAL_VALIDITY_ADDENDUM_HASH,
    derive_final_evaluation_identities,
    dry_run_plan,
    load_final_evaluation_contract,
    resolve_bound_published_reference_dataset,
    validate_future_final_evaluation_authorization,
)
from gwlens_mm.production.run_control import verify_psd_files
from gwlens_mm.production.stage_a import verify_generator_commit
from gwlens_mm.provenance import configuration_hash

_ALLOWED_RELEASE_ONLY_PATHS = (
    "AGENTS.md",
    "configs/execution/phase4_final_evaluation_materialization_authorization.yaml",
    "docs/DECISIONS.md",
    "docs/PROJECT_STATE.md",
    "docs/reports/PHASE7_FINAL_EVALUATION_MATERIALIZATION_REPORT.md",
    "results/experiment_registry.csv",
    "results/phase7/final_reference_catalog.json",
    "results/phase7/final_materialization_release_packet.json",
    "results/phase7/final_materialization_review.json",
)
_REFERENCE_COUNTS = {
    "train": (131072, 5),
    "validation": (6144, 1),
    "calibration_fit": (4096, 1),
    "sbc_diagnostic": (2048, 1),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_future_release_gate(
    *,
    root: Path,
    authorization_path: Path,
    generator_commit: str,
    commitment_path: Path,
    addendum_path: Path,
) -> Mapping[str, Any]:
    config, _ = load_final_evaluation_contract(root)
    authorization = load_yaml(authorization_path)
    commitment_sha256 = _sha256(commitment_path)
    addendum_sha256 = _sha256(addendum_path)
    validate_future_final_evaluation_authorization(
        authorization,
        config=config,
        generator_commit=generator_commit,
        commitment_sha256=commitment_sha256,
        numerical_validity_addendum_sha256=addendum_sha256,
    )
    if (
        commitment_sha256 != FINAL_EVALUATION_COMMITMENT_HASH
        or addendum_sha256 != NUMERICAL_VALIDITY_ADDENDUM_HASH
    ):
        raise ValueError("final-evaluation release commitment identity mismatch")
    verify_generator_commit(
        root,
        generator_commit,
        allowed_postfreeze_paths=_ALLOWED_RELEASE_ONLY_PATHS,
    )
    immutable = authorization["immutable_generator"]
    wheel = Path(str(immutable["wheel_path"]))
    environment = Path(str(immutable["environment_lock_path"]))
    checks: dict[str, Any] = {
        "wheel_sha256": _sha256(wheel),
        "environment_lock_sha256": _sha256(environment),
    }
    if (
        checks["wheel_sha256"] != immutable.get("wheel_sha256")
        or checks["environment_lock_sha256"]
        != immutable.get("environment_lock_sha256")
    ):
        raise ValueError("final-evaluation immutable environment identity mismatch")
    base = load_yaml(root / str(config["base_data_config"]))
    checks["baseline_psd_files"] = verify_psd_files(base["gw"]["psd_curves"])
    references = authorization.get("published_reference_datasets")
    if not isinstance(references, dict) or set(references) != {
        "train",
        "validation",
        "calibration_fit",
        "sbc_diagnostic",
    }:
        raise ValueError("final-evaluation release lacks all reference datasets")
    approved = Path("/root/autodl-tmp/lensing-4")
    for role, specification in references.items():
        datasets = (
            specification.get("datasets")
            if isinstance(specification, dict)
            else None
        )
        expected_count, expected_dataset_count = _REFERENCE_COUNTS[role]
        exclusions = (
            specification.get("excluded_physical_system_ids", ())
            if isinstance(specification, dict)
            else ()
        )
        if (
            not isinstance(datasets, list)
            or len(datasets) != expected_dataset_count
            or int(specification.get("accepted_system_count", -1))
            != expected_count
            or not isinstance(exclusions, list)
            or (role == "train" and len(exclusions) != 5)
            or (role != "train" and exclusions)
        ):
            raise ValueError(
                f"final-evaluation {role} structured reference contract is invalid"
            )
        resolved = [
            resolve_bound_published_reference_dataset(
                item,
                approved_root=approved,
            )
            for item in datasets
        ]
        if len({item.dataset_root for item in resolved}) != len(resolved):
            raise ValueError(f"final-evaluation {role} dataset roots collide")
    staging = Path(str(config["paths"]["staging_root"]))
    staging.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging.parent).free
    checks["free_bytes"] = free
    if free < int(config["resource_gates"]["minimum_prelaunch_free_bytes"]):
        raise RuntimeError("final-evaluation release free-space gate failed")
    identities = derive_final_evaluation_identities(root, config, generator_commit)
    publication = Path(str(config["paths"]["publication_root"]))
    if (publication / identities.parent_run_id).exists():
        raise FileExistsError("final-evaluation official parent identity already exists")
    return {
        "status": "ready_for_sealed_final_evaluation_materialization",
        "generator_commit": generator_commit,
        "configuration_hash": configuration_hash(config),
        "commitment_sha256": commitment_sha256,
        "numerical_validity_addendum_sha256": addendum_sha256,
        "official_identities": {
            "parent_run_id": identities.parent_run_id,
            "namespace_dataset_ids": dict(identities.namespace_dataset_ids),
        },
        "disjoint_reference_dataset_roots": references,
        "checks": checks,
        "accepted_target": 20480,
        "sealed": True,
        "unsealing_authorized": False,
        "scientific_analysis_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--generator-commit")
    parser.add_argument("--commitment", type=Path)
    parser.add_argument("--numerical-validity-addendum", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    result: Mapping[str, Any]
    if arguments.execute:
        required = (
            arguments.authorization,
            arguments.generator_commit,
            arguments.commitment,
            arguments.numerical_validity_addendum,
            arguments.output,
        )
        if any(value is None for value in required):
            raise ValueError("future final-evaluation release gate requires all identities")
        result = _evaluate_future_release_gate(
            root=root,
            authorization_path=arguments.authorization,
            generator_commit=arguments.generator_commit,
            commitment_path=arguments.commitment,
            addendum_path=arguments.numerical_validity_addendum,
        )
    else:
        result = dry_run_plan(root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            position = sys.argv.index("--output") + 1
            if position < len(sys.argv):
                output = Path(sys.argv[position])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "status": "blocked_preexecution",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "official_identities": None,
                            "materialization_authorized": False,
                            "unsealing_authorized": False,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        raise
