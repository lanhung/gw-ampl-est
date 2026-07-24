from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production import final_evaluation_authorization as release_module
from gwlens_mm.production.final_evaluation import (
    FINAL_EVALUATION_COMMITMENT_HASH,
    NUMERICAL_VALIDITY_ADDENDUM_HASH,
    validate_future_final_evaluation_authorization,
)
from gwlens_mm.production.final_evaluation_authorization import (
    AUTHORIZATION_STATUS,
    REFERENCE_CATALOG_STATUS,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_final_materialization_authorization,
    load_final_materialization_release_stack_contract,
    validate_final_reference_catalog,
)
from gwlens_mm.production.waveform_correction import (
    CORRECTION_PREREGISTRATION_HASH,
)
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def _parent_and_child(
    approved: Path,
    name: str,
) -> tuple[Path, dict[str, str]]:
    parent = approved / "published" / f"{name}-parent"
    child = parent / f"{name}-dataset"
    records = child / "shards" / "shard-00000" / "records.parquet"
    records.parent.mkdir(parents=True)
    records.write_bytes(b"fixture")
    manifest = parent / "dataset_manifest.json"
    manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
    return child, {
        "dataset_id": child.name,
        "dataset_root": str(child),
        "parent_root": str(parent),
        "parent_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def _catalog(approved: Path) -> dict:
    entries = [_parent_and_child(approved, f"dataset-{index}")[1] for index in range(8)]
    return {
        "status": REFERENCE_CATALOG_STATUS,
        "training_reference_mode": "terminal_131k",
        "roles": {
            "train": {
                "accepted_system_count": 131072,
                "datasets": entries[:5],
                "excluded_physical_system_ids": [f"excluded-{index}" for index in range(5)],
                "logical_manifest_sha256": "1" * 64,
            },
            "validation": {
                "accepted_system_count": 6144,
                "datasets": entries[5:6],
                "excluded_physical_system_ids": [],
                "logical_manifest_sha256": "2" * 64,
            },
            "calibration_fit": {
                "accepted_system_count": 4096,
                "datasets": entries[6:7],
                "excluded_physical_system_ids": [],
                "logical_manifest_sha256": "3" * 64,
            },
            "sbc_diagnostic": {
                "accepted_system_count": 2048,
                "datasets": entries[7:8],
                "excluded_physical_system_ids": [],
                "logical_manifest_sha256": "3" * 64,
            },
        },
        "scientific_data_opened": False,
        "final_evaluation_materialized": False,
    }


def _packet() -> dict:
    config = load_yaml(ROOT / "configs/data/phase4_final_evaluation.yaml")
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "materialization_execution_authorized": False,
        "official_identities": None,
        "implementation_commit": "a" * 40,
        "frozen_contract": {
            "configuration_path": "configs/data/phase4_final_evaluation.yaml",
            "configuration_hash": configuration_hash(config),
            "commitment_sha256": FINAL_EVALUATION_COMMITMENT_HASH,
            "numerical_validity_addendum_sha256": (
                NUMERICAL_VALIDITY_ADDENDUM_HASH
            ),
            "waveform_numerical_validity_preregistration_hash": (
                CORRECTION_PREREGISTRATION_HASH
            ),
        },
        "prospective_generator_revision": {
            "original_committed_generator": (
                "bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac"
            ),
            "scope": "waveform_numerical_validity_implementation_only",
            "counts_seeds_distributions_changed": False,
            "original_commitment_mutated": False,
        },
        "training_size_decision": {
            "path": "/root/autodl-tmp/lensing-4/results/terminal.json",
            "sha256": "4" * 64,
            "decision": "lock_train_131k_saturated",
            "selected_training_count": 131072,
        },
        "architecture_decision": {
            "path": "/root/autodl-tmp/lensing-4/results/architecture.json",
            "sha256": "5" * 64,
            "selected_architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "6" * 64,
            "locked_training_rung": 131072,
            "result_count": 12,
            "three_model_seeds_retained": True,
        },
        "published_reference_contract": {
            "training_reference_mode": "terminal_131k",
            "corrected_combined_train_manifest_sha256": "7" * 64,
            "correction_parent_manifest_sha256": (
                "0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2"
            ),
            "correction_publication_tree_sha256": (
                "a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12"
            ),
            "terminal_preregistration_hash": (
                "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
            ),
            "terminal_combined_train_manifest_sha256": "8" * 64,
            "terminal_train_increment_parent_manifest_sha256": "9" * 64,
            "development_tail_manifest_sha256": "a" * 64,
            "validation_manifest_sha256": "b" * 64,
            "strict_corrected_65k_subset": True,
            "development_tail_excluded_from_final_reference": True,
            "extension_above_131072_authorized": False,
            "terminal_size_decision": "lock_train_131k_saturated",
            "terminal_size_decision_sha256": "4" * 64,
            "selected_architecture_locked_rung": 131072,
            "selected_architecture_decision_sha256": "5" * 64,
            "logical_system_counts": {
                "train": 131072,
                "validation": 6144,
                "calibration_fit": 4096,
                "sbc_diagnostic": 2048,
            },
        },
        "published_reference_datasets": {
            role: {
                "accepted_system_count": count,
                "datasets": [],
                "excluded_physical_system_ids": [],
            }
            for role, count in (
                ("train", 131072),
                ("validation", 6144),
                ("calibration_fit", 4096),
                ("sbc_diagnostic", 2048),
            )
        },
        "immutable_generator": {
            "git_commit": "a" * 40,
            "wheel_path": "/root/autodl-tmp/lensing-4/review/final.whl",
            "wheel_sha256": "c" * 64,
            "environment_lock_path": (
                "/root/autodl-tmp/lensing-4/review/environment.txt"
            ),
            "environment_lock_sha256": "d" * 64,
            "editable_install_authorized": False,
        },
        "materialization_contract": {
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "training_size_and_architecture_locked": True,
        },
        "future_authorization_path": (
            "configs/execution/"
            "phase7_final_evaluation_materialization_authorization.yaml"
        ),
        "future_delegated_review_path": (
            "results/phase7/final_materialization_review.json"
        ),
        "release_packet_path": (
            "results/phase7/final_materialization_release_packet.json"
        ),
        "review_scope": {
            "training_reference_mode": "terminal_131k",
            "locked_training_rung": 131072,
            "selected_architecture_id": "nsf-t10-w256",
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "sealed_materialization_authorized": True,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
        },
    }


def test_release_stack_is_implementation_only() -> None:
    authorization = load_final_materialization_release_stack_contract(ROOT)
    flags = authorization["authorization"]
    assert flags["nonauthorizing_release_packet_implementation_authorized"] is True
    assert flags["final_evaluation_materialization_authorized"] is False
    assert flags["final_evaluation_unsealing_authorized"] is False
    assert flags["checkpoint_access_authorized"] is False


def test_structured_reference_catalog_binds_atomic_parents(tmp_path: Path) -> None:
    approved = tmp_path / "project"
    catalog = _catalog(approved)
    assert validate_final_reference_catalog(
        catalog,
        approved_root=approved,
    ) == catalog
    changed = deepcopy(catalog)
    changed["roles"]["train"]["datasets"][1] = changed["roles"]["train"][
        "datasets"
    ][0]
    with pytest.raises(TrainingGateError, match="reused"):
        validate_final_reference_catalog(changed, approved_root=approved)
    changed = deepcopy(catalog)
    changed["roles"]["train"]["datasets"][0]["parent_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="atomic parent"):
        validate_final_reference_catalog(changed, approved_root=approved)


def test_reviewed_packet_builds_only_sealed_materialization_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_module,
        "load_final_materialization_release_stack_contract",
        lambda root: {},
    )
    packet = _packet()
    release_path = (
        tmp_path / "results/phase7/final_materialization_release_packet.json"
    )
    release_path.parent.mkdir(parents=True)
    release_path.write_text(
        json.dumps(packet, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review = {
        "status": REVIEW_STATUS,
        "reviewed_by": (
            "codex_as_delegated_scientific_and_engineering_reviewer"
        ),
        "review_date": "2026-07-24",
        "release_packet_sha256": hashlib.sha256(
            release_path.read_bytes()
        ).hexdigest(),
        **packet["review_scope"],
    }
    review_path = tmp_path / "results/phase7/final_materialization_review.json"
    review_path.write_text(
        json.dumps(review, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = (
        tmp_path
        / "configs/execution/phase7_final_evaluation_materialization_authorization.yaml"
    )
    authorization = build_final_materialization_authorization(
        tmp_path,
        release_packet_path=release_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    assert authorization["authorization"]["sealed_materialization_authorized"] is True
    assert authorization["authorization"]["unsealing_authorized"] is False
    assert authorization["authorization"]["scientific_analysis_authorized"] is False
    config = load_yaml(ROOT / "configs/data/phase4_final_evaluation.yaml")
    validate_future_final_evaluation_authorization(
        authorization,
        config=config,
        generator_commit="a" * 40,
        commitment_sha256=FINAL_EVALUATION_COMMITMENT_HASH,
        numerical_validity_addendum_sha256=NUMERICAL_VALIDITY_ADDENDUM_HASH,
    )

    review["accepted_pair_count"] = 20479
    review_path.write_text(
        json.dumps(review, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingGateError, match="scope changed"):
        build_final_materialization_authorization(
            tmp_path,
            release_packet_path=release_path,
            delegated_review_path=review_path,
            output_path=output,
        )
