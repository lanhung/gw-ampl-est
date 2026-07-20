#!/usr/bin/env python3
"""Independently close out one atomically published terminal-131k release."""

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
    derive_terminal_identities,
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


def validate_terminal_execution_result(
    root: Path,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate exact identities and fail-closed flags without opening data."""

    authorization = load_yaml(
        root / str(config["future_execution_authorization_path"])
    )
    generator_commit = str(result.get("generator_commit", ""))
    immutable = authorization.get("immutable_generator", {})
    if not (
        authorization.get("authorization_status")
        == "authorized_exact_terminal_131k_materialization_only"
        and authorization.get("implementation_commit") == generator_commit
        and immutable.get("git_commit") == generator_commit
    ):
        raise ValueError("terminal closeout generator authorization changed")
    identities = derive_terminal_identities(root, config, generator_commit)
    expected = {
        "parent_run_id": identities.parent_run_id,
        "train_dataset_id": identities.train_dataset_id,
        "development_tail_parent_id": identities.development_tail_parent_id,
        "development_tail_dataset_ids": dict(
            identities.development_tail_dataset_ids
        ),
        "combined_train_id": identities.combined_train_id,
        "configuration_hash": identities.configuration_hash,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("terminal closeout official identity mismatch")
    scheduler = result.get("scheduler")
    if not isinstance(scheduler, Mapping) or not (
        int(scheduler.get("scheduler_worker_processes", -1)) == 32
        and int(scheduler.get("configured_worker_processes", -1)) == 16
        and scheduler.get("worker_64_authorized") is False
        and result.get("orchestration_commit")
        == scheduler.get("orchestration_commit")
    ):
        raise ValueError("terminal closeout scheduler identity mismatch")
    if (
        result.get("status"),
        int(result.get("new_train_accepted_count", -1)),
        int(result.get("new_train_shard_count", -1)),
        int(result.get("development_tail_accepted_count", -1)),
        int(result.get("development_tail_namespace_count", -1)),
        int(result.get("terminal_train_accepted_count", -1)),
        result.get("proposal_equals_evaluation"),
        result.get("all_importance_weights_one"),
        result.get("train_131k_probe_authorized"),
        result.get("architecture_selection_authorized"),
        result.get("calibration_authorized"),
        result.get("sbc_authorized"),
        result.get("final_evaluation_authorized"),
        result.get("extension_above_131072_authorized"),
        result.get("gwosc_gwtc_accessed"),
    ) != (
        "passed",
        TRAIN_INCREMENT_COUNT,
        512,
        TAIL_COUNT,
        4,
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
    ):
        raise ValueError("terminal closeout count or safety contract failed")
    minimum_free = int(config["resource_gates"]["minimum_post_peak_free_bytes"])
    if int(result.get("remaining_free_bytes", -1)) < minimum_free:
        raise ValueError("terminal closeout recorded insufficient free space")
    if result.get("configuration_hash") != configuration_hash(config):
        raise ValueError("terminal closeout configuration hash changed")
    return expected


def closeout_terminal_publication(
    root: Path,
    *,
    config_path: str,
    execution_result_path: Path,
    recompute_tree: bool,
) -> Mapping[str, Any]:
    """Resolve all parents and optionally recompute both large tree hashes."""

    config = load_terminal_131k_contract(root, config_path)
    execution = _load_json(execution_result_path)
    identities = validate_terminal_execution_result(root, config, execution)
    paths = config["paths"]
    reference = config["corrected_65k_reference"]
    reference_roots = reference["publication_roots"]
    train_parent = Path(str(paths["train_publication_root"])) / str(
        identities["parent_run_id"]
    )
    tail_parent = Path(str(paths["tail_publication_root"])) / str(
        identities["development_tail_parent_id"]
    )
    combined_root = Path(str(paths["combined_publication_root"])) / str(
        identities["combined_train_id"]
    )
    train_manifest_hash = _sha256(train_parent / "dataset_manifest.json")
    tail_manifest_hash = _sha256(tail_parent / "dataset_manifest.json")
    publication_authorization = {
        "corrected_65k_publication": {
            "base_generator_commit": reference["base_generator_commit"],
            "base_preregistration_hash": reference["base_preregistration_hash"],
            "correction_generator_commit": reference[
                "correction_generator_commit"
            ],
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
        stage_a_publication_root=Path(str(reference_roots["stage_a"])),
        stage_b_publication_root=Path(str(reference_roots["stage_b"])),
        combined_base_publication_root=Path(str(reference_roots["combined_base"])),
        correction_publication_root=Path(str(reference_roots["correction"])),
        train_parent_root=train_parent,
        combined_131k_publication_root=combined_root,
        development_tail_parent_root=tail_parent,
    )
    if publication.combined_manifest_sha256 != execution["combined_manifest_sha256"]:
        raise ValueError("terminal closeout combined manifest mismatch")

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
            raise ValueError("terminal closeout independent tree checksum failed")
        tree_evidence.update(
            {
                "train_publication_tree_sha256": train_tree,
                "train_publication_bytes": train_bytes,
                "development_tail_publication_tree_sha256": tail_tree,
                "development_tail_publication_bytes": tail_bytes,
            }
        )
    observed_free = shutil.disk_usage(train_parent).free
    minimum_free = int(config["resource_gates"]["minimum_post_peak_free_bytes"])
    if observed_free < minimum_free:
        raise ValueError("terminal closeout current free-space gate failed")
    return {
        "status": "terminal_131k_independent_closeout_passed",
        "execution_result_sha256": _sha256(execution_result_path),
        "configuration_hash": configuration_hash(config),
        "generator_commit": execution["generator_commit"],
        "orchestration_commit": execution["orchestration_commit"],
        **identities,
        "new_train_accepted_count": TRAIN_INCREMENT_COUNT,
        "new_train_shard_count": 512,
        "development_tail_accepted_count": TAIL_COUNT,
        "development_tail_namespace_count": 4,
        "logical_train_accepted_count": TRAIN_131K_COUNT,
        "validation_accepted_count": VALIDATION_COUNT,
        "combined_manifest_sha256": publication.combined_manifest_sha256,
        "train_parent_manifest_sha256": publication.train_parent_manifest_sha256,
        "development_tail_manifest_sha256": (
            publication.development_tail_manifest_sha256
        ),
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
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
    args = parser.parse_args(argv)
    result = closeout_terminal_publication(
        args.root,
        config_path=args.config,
        execution_result_path=args.execution_result,
        recompute_tree=not args.skip_tree_recompute,
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
