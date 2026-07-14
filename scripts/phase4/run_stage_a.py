#!/usr/bin/env python3
"""Materialize direct-target Stage A only after a ready release certificate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.stage_a import (
    OFFICIAL_NAMESPACES,
    PHASE4_CONFIG,
    build_namespace_config,
    load_phase4_contract,
    validate_stage_a_namespace,
    verify_generator_commit,
)
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.schema import SplitName

ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify_execution_authorization(
    config: Mapping[str, Any], generator_commit: str
) -> Mapping[str, Any]:
    path = config["authorization"].get("future_execution_path")
    if not path:
        raise PermissionError("Stage A execution authorization is absent")
    authorization = load_yaml(ROOT / str(path))
    if authorization.get("authorization_status") != (
        "authorized_scientific_materialization_only"
    ):
        raise PermissionError("authorization is not Stage A materialization-only")
    flags = authorization.get("authorization", {})
    for key in (
        "scientific_data_generation_authorized",
        "stage_a_materialization_authorized",
        "disposable_canary_accepted",
    ):
        if flags.get(key) is not True:
            raise PermissionError(f"Stage A authorization requires {key}=true")
    for key in (
        "model_training_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "iid_ood_mismatch_evaluation_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"Stage A authorization requires {key}=false")
    immutable = authorization.get("immutable_generator", {})
    if immutable.get("git_commit") != generator_commit:
        raise ValueError("Stage A authorization generator commit mismatch")
    if (
        authorization.get("preregistration", {}).get("canonical_hash")
        != config["preregistration"]["canonical_hash"]
    ):
        raise ValueError("Stage A authorization preregistration hash mismatch")
    counts = authorization.get("stage_a_contract", {})
    if (
        counts.get("train_accepted_count"),
        counts.get("validation_accepted_count"),
        counts.get("total_accepted_count"),
        counts.get("total_shard_count"),
    ) != (32768, 6144, 38912, 304):
        raise ValueError("Stage A authorization count contract mismatch")
    return authorization


def _generate_pending(
    *,
    stage: Path,
    namespace_config: Mapping[str, Any],
    parent_preregistration: Mapping[str, Any],
    proposal: Mapping[str, Any],
    generator_commit: str,
    dataset_id: str,
) -> None:
    pending: list[int] = []
    for shard_index in range(int(namespace_config["shard_count"])):
        complete = stage / "shards" / f"shard-{shard_index:05d}"
        if complete.exists():
            verify_complete_shard(complete, int(namespace_config["pairs_per_shard"]))
        else:
            pending.append(shard_index)
    workers = min(
        int(namespace_config["execution"]["qualification_worker_processes"]),
        len(pending),
    )
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_qualification_shard,
                shard_index=shard_index,
                stage=stage,
                config=namespace_config,
                preregistration=parent_preregistration,
                generator_git_commit=generator_commit,
                dataset_id=dataset_id,
                proposal_config=proposal,
            ): shard_index
            for shard_index in pending
        }
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--release-certificate", required=True, type=Path)
    parser.add_argument("--config", default=PHASE4_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config, _, _ = load_phase4_contract(ROOT, args.config)
    authorization = _verify_execution_authorization(config, args.generator_commit)
    verify_generator_commit(ROOT, args.generator_commit)
    certificate = json.loads(args.release_certificate.read_text())
    if certificate.get("status") != "ready_for_official_execution":
        raise PermissionError("release certificate is not ready for official execution")
    if certificate.get("generator_commit") != args.generator_commit:
        raise ValueError("release certificate generator commit mismatch")
    if certificate.get("preregistration_hash") != config["preregistration"]["canonical_hash"]:
        raise ValueError("release certificate preregistration hash mismatch")
    identities = certificate.get("official_identities")
    if not isinstance(identities, dict):
        raise ValueError("release certificate lacks official identities")
    staging_root = Path(config["paths"]["stage_a_staging_root"])
    publication_root = Path(config["paths"]["stage_a_publication_root"])
    approved = Path("/root/autodl-tmp/lensing-4")
    for path in (staging_root, publication_root):
        if not path.is_absolute() or not path.is_relative_to(approved):
            raise ValueError("Stage A path escaped the AutoDL project root")
        path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_root.parent).free
    required = int(config["resource_gates"]["minimum_prelaunch_free_bytes"])
    if free < required:
        raise RuntimeError(f"Stage A free-space gate failed: {free} < {required}")
    parent_id = str(identities["parent_run_id"])
    parent_stage = staging_root / parent_id
    parent_publication = publication_root / parent_id
    if parent_publication.exists():
        raise FileExistsError("official Stage A publication identity already exists")
    parent_stage.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        parent_stage / "run_manifest.json",
        {
            "status": "generating_or_resuming",
            "parent_run_id": parent_id,
            "generator_commit": args.generator_commit,
            "preregistration_version": config["preregistration"]["version"],
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            "configuration_hash": configuration_hash(config),
            "authorization_commit": authorization["authorizing_commit"],
            "accepted_target": 38912,
            "model_training_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    )
    parent_preregistration = load_yaml(ROOT / config["parent_scientific_preregistration"]["path"])
    proposal = load_yaml(ROOT / config["direct_target_density_implementation"]["path"])
    dataset_ids = {
        "train": str(identities["train_dataset_id"]),
        "validation": str(identities["validation_dataset_id"]),
    }
    namespace_configs: Dict[str, Mapping[str, Any]] = {}
    for namespace in OFFICIAL_NAMESPACES:
        namespace_config = build_namespace_config(ROOT, config, namespace, canary=False)
        namespace_configs[namespace] = namespace_config
        stage = parent_stage / dataset_ids[namespace]
        for child in ("attempts", "shards", "validation", "environment"):
            (stage / child).mkdir(parents=True, exist_ok=True)
        _atomic_json(
            stage / "run_manifest.json",
            {
                "status": "generating_or_resuming",
                "split": namespace,
                "dataset_id": dataset_ids[namespace],
                "generator_commit": args.generator_commit,
                "configuration_hash": configuration_hash(namespace_config),
                "accepted_target": namespace_config["accepted_pair_count"],
                "proposal_distribution_id": config["direct_target_density_implementation"][
                    "evaluation_target_id"
                ],
                "evaluation_distribution_id": config["direct_target_density_implementation"][
                    "evaluation_target_id"
                ],
                "all_importance_weights_one": True,
            },
        )
        _generate_pending(
            stage=stage,
            namespace_config=namespace_config,
            parent_preregistration=parent_preregistration,
            proposal=proposal,
            generator_commit=args.generator_commit,
            dataset_id=dataset_ids[namespace],
        )
    validations: Dict[str, Any] = {}
    grouped_ids: Dict[str, Dict[str, set[str]]] = {}
    for namespace in OFFICIAL_NAMESPACES:
        summary, identifiers = validate_stage_a_namespace(
            parent_stage / dataset_ids[namespace],
            namespace_config=namespace_configs[namespace],
            expected_split=SplitName(namespace),
            expected_dataset=dataset_ids[namespace],
            generator_commit=args.generator_commit,
        )
        validations[namespace] = summary
        grouped_ids[namespace] = identifiers
    for key in ("pair", "source", "lens", "system", "noise"):
        if grouped_ids["train"][key] & grouped_ids["validation"][key]:
            raise ValueError(f"Stage A train/validation {key} leakage")
    manifest = {
        "status": "passed",
        "parent_run_id": parent_id,
        "generator_commit": args.generator_commit,
        "preregistration_version": config["preregistration"]["version"],
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "accepted_pair_count": 38912,
        "train_accepted_pair_count": 32768,
        "validation_accepted_pair_count": 6144,
        "validations": validations,
        "train_validation_group_disjoint": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "scientific_data_generation_authorized": True,
        "model_training_authorized": False,
        "calibration_authorized": False,
        "final_evaluation_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(parent_stage / "dataset_manifest.json", manifest)
    publication_root.mkdir(parents=True, exist_ok=True)
    os.replace(parent_stage, parent_publication)
    digest, byte_count = tree_checksum(parent_publication)
    manifest.update(
        {
            "publication_path": str(parent_publication),
            "publication_tree_sha256": digest,
            "publication_bytes": byte_count,
            "remaining_free_bytes": shutil.disk_usage(parent_publication).free,
        }
    )
    _atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            index = sys.argv.index("--output") + 1
            if index < len(sys.argv):
                _atomic_json(
                    Path(sys.argv[index]),
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "scientific_data_generation_authorized": False,
                        "model_training_authorized": False,
                        "gwosc_gwtc_accessed": False,
                    },
                )
        raise
