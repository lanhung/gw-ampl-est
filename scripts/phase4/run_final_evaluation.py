#!/usr/bin/env python3
"""Materialize the frozen final pool only through a future sealed-data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping

from gwlens_mm.config import load_yaml
from gwlens_mm.production.final_evaluation import (
    FINAL_EVALUATION_CONFIG,
    build_final_evaluation_namespace_config,
    collect_published_group_identifiers,
    final_evaluation_namespaces,
    load_final_evaluation_contract,
    validate_final_evaluation_namespace,
)
from gwlens_mm.production.qualification import generate_qualification_shard
from gwlens_mm.production.stage_a import verify_generator_commit
from gwlens_mm.production.storage import tree_checksum, verify_complete_shard
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_POSTFREEZE_PATHS = (
    "AGENTS.md",
    "configs/execution/phase4_final_evaluation_materialization_authorization.yaml",
    "docs/DECISIONS.md",
    "docs/PROJECT_STATE.md",
    "docs/reports/PHASE4_FINAL_EVALUATION_GENERATOR_IMPLEMENTATION_REPORT.md",
    "results/experiment_registry.csv",
    "results/phase4/final_evaluation_commitment.json",
    "results/phase4/final_evaluation_commitment.sha256",
    "tests/test_phase4_direct_target.py",
    "tests/test_phase4_final_evaluation.py",
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _future_authorization(
    path: Path,
    *,
    config: Mapping[str, Any],
    generator_commit: str,
    commitment_sha256: str,
) -> Mapping[str, Any]:
    authorization = load_yaml(path)
    if authorization.get("authorization_status") != (
        "authorized_sealed_final_evaluation_materialization_only"
    ):
        raise PermissionError("sealed final-evaluation authorization is absent")
    immutable = authorization.get("immutable_generator", {})
    if immutable.get("git_commit") != generator_commit:
        raise ValueError("final-evaluation authorization generator mismatch")
    frozen = authorization.get("frozen_contract", {})
    if frozen.get("configuration_hash") != configuration_hash(config):
        raise ValueError("final-evaluation authorization config hash mismatch")
    if frozen.get("commitment_sha256") != commitment_sha256:
        raise ValueError("final-evaluation authorization commitment mismatch")
    contract = authorization.get("materialization_contract", {})
    if (
        int(contract.get("accepted_pair_count", -1)),
        int(contract.get("shard_count", -1)),
        int(contract.get("namespace_count", -1)),
    ) != (20480, 160, 15):
        raise ValueError("final-evaluation authorization count mismatch")
    flags = authorization.get("authorization", {})
    if flags.get("sealed_materialization_authorized") is not True:
        raise PermissionError("sealed final-evaluation materialization is closed")
    for key in (
        "unsealing_authorized",
        "scientific_analysis_authorized",
        "model_training_authorized",
        "calibration_fit_authorized",
        "learning_curve_use_authorized",
        "architecture_selection_use_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(key) is not False:
            raise PermissionError(f"future materialization requires {key}=false")
    if contract.get("training_size_and_architecture_locked") is not True:
        raise PermissionError("final pool cannot materialize before model design lock")
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
    pending = []
    for shard_index in range(int(namespace_config["shard_count"])):
        complete = stage / "shards" / f"shard-{shard_index:05d}"
        if complete.exists():
            verify_complete_shard(complete, int(namespace_config["pairs_per_shard"]))
        else:
            pending.append(shard_index)
    if not pending:
        return
    workers = min(
        int(namespace_config["execution"]["qualification_worker_processes"]),
        len(pending),
    )
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
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--release-certificate", required=True, type=Path)
    parser.add_argument("--commitment", required=True, type=Path)
    parser.add_argument("--config", default=FINAL_EVALUATION_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config, _ = load_final_evaluation_contract(ROOT)
    if args.config != FINAL_EVALUATION_CONFIG:
        raise ValueError("only the frozen final-evaluation config is supported")
    commitment = json.loads(args.commitment.read_text())
    commitment_sha256 = _sha256(args.commitment)
    if commitment.get("commitment_status") != "finalized_before_training":
        raise PermissionError("final-evaluation generation commitment is not finalized")
    if commitment.get("future_scientific_generator_commit") != args.generator_commit:
        raise ValueError("commitment generator identity mismatch")
    authorization = _future_authorization(
        args.authorization,
        config=config,
        generator_commit=args.generator_commit,
        commitment_sha256=commitment_sha256,
    )
    verify_generator_commit(
        ROOT,
        args.generator_commit,
        allowed_postfreeze_paths=_ALLOWED_POSTFREEZE_PATHS,
    )
    certificate = json.loads(args.release_certificate.read_text())
    if certificate.get("status") != "ready_for_sealed_final_evaluation_materialization":
        raise PermissionError("release certificate is not ready for sealed materialization")
    if certificate.get("generator_commit") != args.generator_commit:
        raise ValueError("release certificate generator mismatch")
    if certificate.get("configuration_hash") != configuration_hash(config):
        raise ValueError("release certificate configuration mismatch")
    if certificate.get("commitment_sha256") != commitment_sha256:
        raise ValueError("release certificate commitment mismatch")
    identities = certificate.get("official_identities")
    if not isinstance(identities, dict) or not isinstance(
        identities.get("namespace_dataset_ids"), dict
    ):
        raise ValueError("release certificate lacks sealed official identities")
    namespaces = final_evaluation_namespaces(config)
    dataset_ids = identities["namespace_dataset_ids"]
    if set(dataset_ids) != {item.namespace_id for item in namespaces}:
        raise ValueError("release certificate namespace identity set mismatch")
    if len(set(dataset_ids.values())) != len(namespaces):
        raise ValueError("final-evaluation dataset identities collide")
    reference_roots_value = certificate.get("disjoint_reference_dataset_roots")
    required_reference_splits = {"train", "validation", "calibration_fit", "sbc_diagnostic"}
    if not isinstance(reference_roots_value, dict) or set(reference_roots_value) != (
        required_reference_splits
    ):
        raise ValueError("release certificate lacks all published reference pools")
    reference_roots = tuple(Path(str(reference_roots_value[key])) for key in sorted(
        required_reference_splits
    ))
    parent_id = str(identities["parent_run_id"])
    if any(
        str(value).startswith(("qualification-", "phase3ca", "phase4-canary"))
        for value in (parent_id, *dataset_ids.values())
    ):
        raise ValueError("engineering identity entered final evaluation")

    staging_root = Path(config["paths"]["staging_root"])
    publication_root = Path(config["paths"]["publication_root"])
    approved = Path("/root/autodl-tmp/lensing-4")
    for path in (staging_root, publication_root):
        if not path.is_absolute() or not path.is_relative_to(approved):
            raise ValueError("final-evaluation path escaped the AutoDL project root")
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in reference_roots:
        if (
            not path.is_absolute()
            or not path.is_relative_to(approved)
            or "staging" in path.parts
            or path.is_relative_to(staging_root)
            or path.is_relative_to(publication_root)
        ):
            raise ValueError("final-evaluation reference is not a published external pool")
    free = shutil.disk_usage(staging_root.parent).free
    required = int(config["resource_gates"]["minimum_prelaunch_free_bytes"])
    if free < required:
        raise RuntimeError(f"final-evaluation free-space gate failed: {free} < {required}")
    parent_stage = staging_root / parent_id
    parent_publication = publication_root / parent_id
    if parent_publication.exists():
        raise FileExistsError("sealed final-evaluation identity already exists")
    parent_stage.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        parent_stage / "run_manifest.json",
        {
            "status": "generating_or_resuming_sealed",
            "parent_run_id": parent_id,
            "generator_commit": args.generator_commit,
            "configuration_hash": configuration_hash(config),
            "commitment_sha256": commitment_sha256,
            "authorization_identity": authorization.get("authorizing_commit"),
            "accepted_target": 20480,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
        },
    )
    parent_preregistration = load_yaml(
        ROOT / config["parent_scientific_preregistration"]["path"]
    )
    proposal = load_yaml(ROOT / config["target_proposal_config"])
    namespace_configs: Dict[str, Mapping[str, Any]] = {}
    for namespace in namespaces:
        namespace_config = build_final_evaluation_namespace_config(
            ROOT, config, namespace
        )
        namespace_configs[namespace.namespace_id] = namespace_config
        dataset_id = str(dataset_ids[namespace.namespace_id])
        stage = parent_stage / dataset_id
        for child in ("attempts", "shards", "validation", "environment"):
            (stage / child).mkdir(parents=True, exist_ok=True)
        _atomic_json(
            stage / "run_manifest.json",
            {
                "status": "generating_or_resuming_sealed",
                "namespace_id": namespace.namespace_id,
                "split": namespace.split.value,
                "diagnostic_context_id": namespace.diagnostic_context_id,
                "dataset_id": dataset_id,
                "generator_commit": args.generator_commit,
                "configuration_hash": configuration_hash(namespace_config),
                "accepted_target": namespace.accepted_count,
                "unsealing_authorized": False,
            },
        )
        _generate_pending(
            stage=stage,
            namespace_config=namespace_config,
            parent_preregistration=parent_preregistration,
            proposal=proposal,
            generator_commit=args.generator_commit,
            dataset_id=dataset_id,
        )

    validations: Dict[str, Any] = {}
    global_ids: Dict[str, set[str]] = {
        key: set()
        for key in (
            "pair",
            "source",
            "lens",
            "system",
            "noise",
            "augmentation_parent",
        )
    }
    for namespace in namespaces:
        dataset_id = str(dataset_ids[namespace.namespace_id])
        summary, identifiers = validate_final_evaluation_namespace(
            parent_stage / dataset_id,
            namespace_config=namespace_configs[namespace.namespace_id],
            namespace=namespace,
            expected_dataset=dataset_id,
            generator_commit=args.generator_commit,
        )
        validations[namespace.namespace_id] = summary
        for key in global_ids:
            if global_ids[key] & identifiers[key]:
                raise ValueError(f"cross-namespace final-evaluation {key} leakage")
            global_ids[key].update(identifiers[key])
    reference_ids = collect_published_group_identifiers(reference_roots)
    for key in global_ids:
        if global_ids[key] & reference_ids[key]:
            raise ValueError(f"final evaluation leaks into a reference {key} group")
    manifest = {
        "status": "passed_sealed",
        "parent_run_id": parent_id,
        "generator_commit": args.generator_commit,
        "configuration_hash": configuration_hash(config),
        "commitment_sha256": commitment_sha256,
        "accepted_pair_count": 20480,
        "complete_shard_count": 160,
        "namespace_count": 15,
        "validations": validations,
        "all_namespaces_group_disjoint": True,
        "sealed": True,
        "unsealing_authorized": False,
        "learning_curve_use_authorized": False,
        "architecture_selection_use_authorized": False,
        "calibration_fit_use_authorized": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(parent_stage / "dataset_manifest.json", manifest)
    digest, byte_count = tree_checksum(parent_stage)
    if byte_count > int(config["resource_gates"]["maximum_published_bytes"]):
        raise RuntimeError("sealed final-evaluation staging exceeded byte gate")
    remaining = shutil.disk_usage(parent_stage).free
    if remaining < int(config["resource_gates"]["minimum_post_publication_free_bytes"]):
        raise RuntimeError("sealed final-evaluation free-space gate failed before publication")
    publication_root.mkdir(parents=True, exist_ok=True)
    os.replace(parent_stage, parent_publication)
    result = {
        **manifest,
        "publication_path": str(parent_publication),
        "publication_tree_sha256": digest,
        "publication_bytes": byte_count,
        "remaining_free_bytes": remaining,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            position = sys.argv.index("--output") + 1
            if position < len(sys.argv):
                _atomic_json(
                    Path(sys.argv[position]),
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "unsealing_authorized": False,
                        "scientific_analysis_authorized": False,
                        "gwosc_gwtc_accessed": False,
                    },
                )
        raise
