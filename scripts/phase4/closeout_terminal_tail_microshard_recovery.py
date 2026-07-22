#!/usr/bin/env python3
"""Independently close out the atomic terminal-tail microshard recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.production.storage import tree_checksum
from gwlens_mm.production.terminal131 import (
    TERMINAL_131K_CONFIG,
    load_terminal_131k_contract,
)
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.terminal131 import (
    TAIL_COUNT,
    TRAIN_131K_COUNT,
    TRAIN_INCREMENT_COUNT,
    VALIDATION_COUNT,
    resolve_terminal_131k_training_publication,
)
from scripts.phase4.run_terminal_tail_microshard_recovery import (
    AUTHORIZATION,
    BASE_CONFIG_HASH,
    GENERATOR_COMMIT,
    TRAIN_DATASET_ID,
    TRAIN_PARENT_ID,
    _recovery_identities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def validate_microshard_execution_result(
    root: Path,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate exact identities, layout, counts, and closed safety gates."""

    authorization = load_yaml(root / AUTHORIZATION)
    frozen = authorization.get("frozen_scientific_contract", {})
    orchestration = authorization.get("immutable_orchestration", {})
    execution = authorization.get("microshard_execution_contract", {})
    flags = authorization.get("authorization", {})
    orchestration_commit = str(result.get("orchestration_commit", ""))
    if not (
        authorization.get("authorization_status")
        == "authorized_engineering_microshard_recovery_only"
        and frozen.get("generator_commit") == GENERATOR_COMMIT
        and frozen.get("base_configuration_hash") == BASE_CONFIG_HASH
        and configuration_hash(config) == BASE_CONFIG_HASH
        and orchestration.get("git_commit") == orchestration_commit
        and int(execution.get("worker_processes", -1)) == 32
        and int(execution.get("total_shard_count", -1)) == 512
        and int(execution.get("total_accepted_pairs", -1)) == TAIL_COUNT
        and flags.get("tail_microshard_recovery_authorized") is True
        and flags.get("training_authorized") is False
        and flags.get("final_evaluation_authorized") is False
        and flags.get("gwosc_gwtc_access_authorized") is False
    ):
        raise ValueError("terminal microshard authorization changed")
    identities = _recovery_identities(config, orchestration_commit, authorization)
    expected = {
        "tail_parent_id": identities["tail_parent_id"],
        "tail_dataset_ids": identities["tail_dataset_ids"],
        "combined_train_id": identities["combined_train_id"],
        "identity_payload_sha256": identities["identity_payload_sha256"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("terminal microshard official identity mismatch")
    observed = (
        result.get("status"),
        result.get("phase"),
        result.get("generator_commit"),
        result.get("train_parent_id"),
        result.get("train_dataset_id"),
        result.get("configuration_hash"),
        int(result.get("new_train_accepted_count", -1)),
        int(result.get("development_tail_accepted_count", -1)),
        int(result.get("development_tail_namespace_count", -1)),
        int(result.get("development_tail_shard_count", -1)),
        int(result.get("development_tail_shards_per_namespace", -1)),
        int(result.get("development_tail_pairs_per_shard", -1)),
        int(result.get("terminal_train_accepted_count", -1)),
        result.get("proposal_equals_evaluation"),
        result.get("all_importance_weights_one"),
        result.get("failed_parallel32_evidence_reused"),
        result.get("train_131k_probe_authorized"),
        result.get("architecture_selection_authorized"),
        result.get("calibration_authorized"),
        result.get("sbc_authorized"),
        result.get("final_evaluation_authorized"),
        result.get("extension_above_131072_authorized"),
        result.get("gwosc_gwtc_accessed"),
    )
    expected_values = (
        "passed",
        "4-terminal-tail-microshard-recovery",
        GENERATOR_COMMIT,
        TRAIN_PARENT_ID,
        TRAIN_DATASET_ID,
        BASE_CONFIG_HASH,
        TRAIN_INCREMENT_COUNT,
        TAIL_COUNT,
        4,
        512,
        128,
        1,
        TRAIN_131K_COUNT,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    if observed != expected_values:
        raise ValueError("terminal microshard count or safety contract failed")
    if int(result.get("remaining_free_bytes", -1)) < int(
        authorization["resource_gates"]["minimum_post_run_free_bytes"]
    ):
        raise ValueError("terminal microshard result recorded insufficient free space")
    return expected, authorization


def closeout_microshard_publication(
    root: Path,
    *,
    config_path: str,
    execution_result_path: Path,
    recompute_tree: bool,
) -> Mapping[str, Any]:
    """Resolve every parent and independently replay large publication hashes."""

    config = load_terminal_131k_contract(root, config_path)
    execution = _load_json(execution_result_path)
    identities, authorization = validate_microshard_execution_result(
        root, config, execution
    )
    reference = config["corrected_65k_reference"]
    roots = reference["publication_roots"]
    train_parent = Path(str(authorization["published_train_increment"]["parent_root"]))
    tail_parent = Path(str(authorization["paths"]["tail_publication_root"])) / str(
        identities["tail_parent_id"]
    )
    combined_root = Path(
        str(authorization["paths"]["combined_publication_root"])
    ) / str(identities["combined_train_id"])
    train_manifest_hash = _sha256(train_parent / "dataset_manifest.json")
    tail_manifest_hash = _sha256(tail_parent / "dataset_manifest.json")
    publication_authorization = {
        "corrected_65k_publication": {
            "base_generator_commit": reference["base_generator_commit"],
            "base_preregistration_hash": reference["base_preregistration_hash"],
            "correction_generator_commit": reference["correction_generator_commit"],
            "correction_preregistration_hash": reference[
                "correction_preregistration_hash"
            ],
            "correction_parent_manifest_sha256": reference[
                "correction_parent_manifest_sha256"
            ],
            "correction_publication_tree_sha256": reference[
                "correction_publication_tree_sha256"
            ],
            "combined_base_manifest_sha256": reference[
                "combined_base_manifest_sha256"
            ],
        },
        "terminal_publication": {
            "combined_manifest_sha256": execution["combined_manifest_sha256"],
            "train_parent_manifest_sha256": train_manifest_hash,
            "development_tail_manifest_sha256": tail_manifest_hash,
        },
    }
    publication = resolve_terminal_131k_training_publication(
        publication_authorization,
        stage_a_publication_root=Path(str(roots["stage_a"])),
        stage_b_publication_root=Path(str(roots["stage_b"])),
        combined_base_publication_root=Path(str(roots["combined_base"])),
        correction_publication_root=Path(str(roots["correction"])),
        train_parent_root=train_parent,
        combined_131k_publication_root=combined_root,
        development_tail_parent_root=tail_parent,
    )
    if (
        publication.development_tail_shards_per_namespace != 128
        or publication.development_tail_pairs_per_shard != 1
    ):
        raise ValueError("terminal microshard reader resolved the wrong layout")
    tree_evidence: Dict[str, Any] = {"recomputed": recompute_tree}
    if recompute_tree:
        train_tree, train_bytes = tree_checksum(train_parent)
        tail_tree, tail_bytes = tree_checksum(tail_parent)
        if (
            train_tree != execution.get("train_publication_tree_sha256")
            or train_bytes != int(execution.get("train_publication_bytes", -1))
            or tail_tree != execution.get("development_tail_publication_tree_sha256")
            or tail_bytes
            != int(execution.get("development_tail_publication_bytes", -1))
        ):
            raise ValueError("terminal microshard independent tree checksum failed")
        tree_evidence.update(
            {
                "train_publication_tree_sha256": train_tree,
                "train_publication_bytes": train_bytes,
                "development_tail_publication_tree_sha256": tail_tree,
                "development_tail_publication_bytes": tail_bytes,
            }
        )
    observed_free = shutil.disk_usage(train_parent).free
    minimum_free = int(authorization["resource_gates"]["minimum_post_run_free_bytes"])
    if observed_free < minimum_free:
        raise ValueError("terminal microshard closeout free-space gate failed")
    return {
        "status": "terminal_131k_independent_closeout_passed",
        "execution_mode": "dynamic_microshard_tail_recovery",
        "execution_result_sha256": _sha256(execution_result_path),
        "configuration_hash": configuration_hash(config),
        "generator_commit": execution["generator_commit"],
        "orchestration_commit": execution["orchestration_commit"],
        "parent_run_id": TRAIN_PARENT_ID,
        "train_dataset_id": TRAIN_DATASET_ID,
        "development_tail_parent_id": identities["tail_parent_id"],
        "development_tail_dataset_ids": identities["tail_dataset_ids"],
        "development_tail_shards_per_namespace": 128,
        "development_tail_pairs_per_shard": 1,
        "combined_train_id": identities["combined_train_id"],
        "identity_payload_sha256": identities["identity_payload_sha256"],
        "new_train_accepted_count": TRAIN_INCREMENT_COUNT,
        "new_train_shard_count": 512,
        "development_tail_accepted_count": TAIL_COUNT,
        "development_tail_namespace_count": 4,
        "development_tail_shard_count": 512,
        "logical_train_accepted_count": TRAIN_131K_COUNT,
        "validation_accepted_count": VALIDATION_COUNT,
        "combined_manifest_sha256": publication.combined_manifest_sha256,
        "train_parent_manifest_sha256": publication.train_parent_manifest_sha256,
        "development_tail_manifest_sha256": (
            publication.development_tail_manifest_sha256
        ),
        "publication_roots": {
            "terminal_train_increment": str(train_parent),
            "terminal_combined_131k": str(combined_root),
            "development_tail": str(tail_parent),
        },
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "failed_parallel32_evidence_reused": False,
        "observed_remaining_free_bytes": observed_free,
        "minimum_post_peak_free_bytes": minimum_free,
        "tree_evidence": tree_evidence,
        "training_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=TERMINAL_131K_CONFIG)
    parser.add_argument("--execution-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-tree-recompute", action="store_true")
    arguments = parser.parse_args(argv)
    result = closeout_microshard_publication(
        arguments.root,
        config_path=arguments.config,
        execution_result_path=arguments.execution_result,
        recompute_tree=not arguments.skip_tree_recompute,
    )
    _atomic_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
