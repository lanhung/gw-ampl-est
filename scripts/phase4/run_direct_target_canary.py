#!/usr/bin/env python3
"""Run the disposable 8+8 direct-target execution canary after human authorization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.production.ab_qualification import validate_first_block_health
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.stage_a import (
    CANARY_NAMESPACES,
    PHASE4_CONFIG,
    build_namespace_config,
    derive_canary_identity,
    load_phase4_contract,
    validate_direct_target_shard,
    verify_generator_commit,
)
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.schema import SplitName

ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_canary_authorization(
    path: Path, generator_commit: str, preregistration_hash: str
) -> Mapping[str, Any]:
    authorization_path = path if path.is_absolute() else ROOT / path
    if not authorization_path.is_file():
        raise PermissionError("disposable canary execution authorization is absent")
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != "authorized_disposable_canary_only":
        raise PermissionError("authorization is not canary-only")
    preregistration = authorization.get("preregistration", {})
    if preregistration.get("canonical_hash") != preregistration_hash:
        raise ValueError("canary authorization preregistration hash mismatch")
    immutable = authorization.get("immutable_generator", {})
    if immutable.get("git_commit") != generator_commit:
        raise ValueError("canary authorization generator commit mismatch")
    wheel_path = Path(str(immutable.get("wheel_path", "")))
    if not wheel_path.is_file() or _sha256(wheel_path) != immutable.get("wheel_sha256"):
        raise ValueError("canary generator wheel identity mismatch")
    lock_path = ROOT / "configs/environment/phase4-autodl-requirements.lock.txt"
    if _sha256(lock_path) != immutable.get("environment_lock_sha256"):
        raise ValueError("canary environment lock identity mismatch")
    if immutable.get("editable_install_authorized") is not False:
        raise PermissionError("editable project installation is forbidden")
    flags = authorization.get("authorization", {})
    if flags.get("disposable_canary_execution_authorized") is not True:
        raise PermissionError("disposable canary execution is not authorized")
    if flags.get("accepted_pair_generator_authorized_within_canary_only") is not True:
        raise PermissionError("accepted-pair generator is not authorized for canary")
    for key in (
        "scientific_data_generation_authorized",
        "stage_a_materialization_authorized",
        "model_training_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "scientific_evaluation_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"canary authorization requires {key}=false")
    contract = authorization.get("canary_contract", {})
    if (
        contract.get("train_namespace_accepted_pairs"),
        contract.get("validation_namespace_accepted_pairs"),
        contract.get("total_accepted_pairs"),
    ) != (8, 8, 16):
        raise ValueError("canary authorization count mismatch")
    if contract.get("throughput_inspection_authorized") is not False:
        raise PermissionError("canary throughput inspection must remain forbidden")
    if contract.get("ess_inspection_authorized") is not False:
        raise PermissionError("canary ESS inspection must remain forbidden")
    policy = authorization.get("use_policy", {})
    if any(
        policy.get(key) is not False
        for key in (
            "scientific_use_authorized",
            "training_use_authorized",
            "calibration_use_authorized",
            "test_use_authorized",
        )
    ):
        raise PermissionError("canary use policy must remain entirely false")
    if policy.get("permanent_exclusion_from_all_scientific_splits") is not True:
        raise PermissionError("canary permanent scientific exclusion is absent")
    return authorization


def _verify_noneditable_install() -> None:
    distribution = importlib.metadata.distribution("gwlens-mm")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return
    metadata = json.loads(direct_url)
    if metadata.get("dir_info", {}).get("editable") is True:
        raise PermissionError("canary execution cannot use an editable project install")


def _environment_identity() -> Dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "zarr",
                "bilby",
                "lalsuite",
                "lenstronomy",
                "astropy",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--config", default=PHASE4_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-after-first-namespace", action="store_true")
    args = parser.parse_args()
    config, _, _ = load_phase4_contract(ROOT, args.config)
    authorization = _verify_canary_authorization(
        args.authorization,
        args.generator_commit,
        str(config["preregistration"]["canonical_hash"]),
    )
    authorization_path = args.authorization
    if authorization_path.is_absolute():
        try:
            allowed_authorization = str(authorization_path.relative_to(ROOT))
        except ValueError as error:
            raise ValueError("authorization file must be inside the repository") from error
    else:
        allowed_authorization = str(authorization_path)
    verify_generator_commit(
        ROOT,
        args.generator_commit,
        allowed_postfreeze_paths=(allowed_authorization, "AGENTS.md"),
    )
    _verify_noneditable_install()
    identity = derive_canary_identity(config, args.generator_commit)
    staging_root = Path(config["paths"]["canary_staging_root"])
    evidence_root = Path(config["paths"]["canary_evidence_root"])
    approved = Path("/root/autodl-tmp/lensing-4")
    for path in (staging_root, evidence_root):
        if not path.is_absolute() or not path.is_relative_to(approved):
            raise ValueError("canary path escaped the AutoDL project root")
        path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_root).free
    required = int(config["resource_gates"]["minimum_canary_prelaunch_free_bytes"])
    if free < required:
        raise RuntimeError(f"canary free-space gate failed: {free} < {required}")
    parent = staging_root / identity.parent_run_id
    parent.mkdir(parents=True, exist_ok=True)
    environment_path = parent / "environment.json"
    current_environment = _environment_identity()
    current_environment.update(
        {
            "generator_commit": args.generator_commit,
            "wheel_sha256": authorization["immutable_generator"]["wheel_sha256"],
            "editable_install": False,
            "python_executable": sys.executable,
        }
    )
    if environment_path.exists():
        if json.loads(environment_path.read_text()) != current_environment:
            raise ValueError("canary resume environment identity changed")
    else:
        _atomic_json(environment_path, current_environment)
    datasets = {
        "train_namespace": identity.train_dataset_id,
        "validation_namespace": identity.validation_dataset_id,
    }
    proposal = load_yaml(ROOT / config["direct_target_density_implementation"]["path"])
    parent_preregistration = load_yaml(ROOT / config["parent_scientific_preregistration"]["path"])
    checkpoint = evidence_root / f"{identity.parent_run_id}.first_namespace.json"
    results: Dict[str, Any] = {}
    for position, namespace in enumerate(CANARY_NAMESPACES):
        dataset = datasets[namespace]
        stage = parent / dataset
        for child in ("attempts", "shards", "validation", "environment"):
            (stage / child).mkdir(parents=True, exist_ok=True)
        namespace_config = build_namespace_config(ROOT, config, namespace, canary=True)
        complete = stage / "shards" / "shard-00000"
        if not complete.exists():
            generate_qualification_shard(
                shard_index=0,
                stage=stage,
                config=namespace_config,
                preregistration=parent_preregistration,
                generator_git_commit=identity.generator_commit,
                dataset_id=dataset,
                proposal_config=proposal,
            )
        verify_complete_shard(complete, 8)
        health = validate_first_block_health(stage, dataset, expected_pairs=8)
        direct = validate_direct_target_shard(
            complete,
            expected_split=SplitName.GENERATOR_QUALIFICATION,
            expected_dataset=dataset,
            expected_pairs=8,
        )
        digest, byte_count = tree_checksum(complete)
        results[namespace] = {
            "dataset_id": dataset,
            "tree_sha256": digest,
            "bytes": byte_count,
            "health": health,
            "direct_target": direct,
        }
        if position == 0 and not checkpoint.exists():
            _atomic_json(
                checkpoint,
                {
                    "generator_commit": identity.generator_commit,
                    "dataset_id": dataset,
                    "tree_sha256": digest,
                },
            )
        if position == 0 and args.stop_after_first_namespace:
            _atomic_json(
                args.output,
                {
                    "status": "intentional_canary_interruption",
                    "generator_commit": identity.generator_commit,
                    "parent_run_id": identity.parent_run_id,
                    "accepted_pair_count": 8,
                    "scientific_use_authorized": False,
                    "training_use_authorized": False,
                    "throughput_or_ess_inspected": False,
                },
            )
            return
    first_checkpoint = json.loads(checkpoint.read_text())
    first_unchanged = (
        first_checkpoint["tree_sha256"] == results["train_namespace"]["tree_sha256"]
    )
    if not first_unchanged:
        raise ValueError("completed first canary namespace changed across resume")
    total_bytes = sum(int(value["bytes"]) for value in results.values())
    if total_bytes > int(config["resource_gates"]["maximum_canary_output_bytes"]):
        raise RuntimeError("canary output exceeded its hard byte cap")
    manifest = {
        "status": "passed",
        "generator_commit": identity.generator_commit,
        "configuration_hash": identity.configuration_hash,
        "parent_run_id": identity.parent_run_id,
        "namespaces": results,
        "accepted_pair_count": 16,
        "resume_first_namespace_byte_identical": True,
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "calibration_use_authorized": False,
        "test_use_authorized": False,
        "throughput_or_ess_inspected": False,
        "stage_a_identity_created": False,
        "gwosc_gwtc_accessed": False,
        "environment": current_environment,
        "total_bytes": total_bytes,
    }
    _atomic_json(args.output, manifest)
    evidence_manifest = evidence_root / f"{identity.parent_run_id}.manifest.json"
    _atomic_json(evidence_manifest, manifest)
    print(
        json.dumps(
            {**manifest, "manifest_sha256": _sha256(evidence_manifest)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
