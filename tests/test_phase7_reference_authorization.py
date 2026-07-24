from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gwlens_mm.training import reference_authorization as release_module
from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.reference_authorization import (
    AUTHORIZATION_STATUS,
    CATALOG_STATUS,
    CLOSED_BOUNDARIES,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_reference_query_authorization,
    build_reference_query_release_packet,
    validate_reference_query_catalog,
)
from gwlens_mm.training.reference_baseline import REFERENCE_CONFIG_HASH
from gwlens_mm.training.reference_execution import (
    validate_reference_execution_stack_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _dataset_spec(
    project: Path,
    *,
    parent_name: str,
    dataset_name: str,
    accepted_count: int,
) -> dict[str, object]:
    parent = project / "published" / parent_name
    dataset = parent / dataset_name
    (dataset / "shards/shard-00000").mkdir(parents=True)
    (dataset / "shards/shard-00000/records.parquet").write_bytes(b"fixture")
    manifest = parent / "dataset_manifest.json"
    if not manifest.exists():
        _write_json(manifest, {"status": "passed"})
    assert not (dataset / "dataset_manifest.json").exists()
    return {
        "dataset_id": dataset_name,
        "dataset_root": str(dataset),
        "parent_root": str(parent),
        "parent_manifest_sha256": _sha256(manifest),
        "accepted_count": accepted_count,
    }


def _review_scope(role: str) -> dict[str, object]:
    final = role != "validation"
    count = {
        "validation": 6144,
        "iid_test": 8192,
        "balanced_tail_diagnostic": 4096,
    }[role]
    dataset_count = 4 if role == "balanced_tail_diagnostic" else 1
    return {
        "locked_training_rung": 131072,
        "selected_architecture_id": "nsf-t10-w256",
        "query_role": role,
        "query_count": count,
        "query_dataset_count": dataset_count,
        "scientific_reference_bank_access_authorized": True,
        "reference_query_execution_authorized": True,
        "validation_reference_execution_authorized": not final,
        "final_evaluation_unsealing_authorized": final,
        "final_reference_execution_authorized": final,
    }


def _packet(role: str = "validation") -> dict[str, object]:
    counts = {
        "validation": (6144,),
        "iid_test": (8192,),
        "balanced_tail_diagnostic": (1024, 1024, 1024, 1024),
    }[role]
    corrected = {
        "base_generator_commit": "1" * 40,
        "base_preregistration_hash": "2" * 64,
        "correction_generator_commit": "3" * 40,
        "correction_parent_manifest_sha256": "4" * 64,
        "correction_publication_tree_sha256": "5" * 64,
        "combined_base_manifest_sha256": "6" * 64,
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "reference_query_execution_authorized": False,
        "implementation_commit": "7" * 40,
        "terminal_decision": {
            "path": "/root/autodl-tmp/lensing-4/terminal.json",
            "sha256": "8" * 64,
            "decision": "lock_train_131k_saturated",
        },
        "selected_architecture": {
            "decision_path": "/root/autodl-tmp/lensing-4/architecture.json",
            "decision_sha256": "9" * 64,
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "a" * 64,
            "locked_training_rung": 131072,
        },
        "primary_rung_preparation": {
            "path": "/root/autodl-tmp/lensing-4/preparation.json",
            "sha256": "b" * 64,
        },
        "query_catalog": {
            "path": "/root/autodl-tmp/lensing-4/query-catalog.json",
            "sha256": "c" * 64,
            "query_role": role,
            "datasets": [
                {
                    "dataset_id": f"{role}-child-{index}",
                    "dataset_root": (
                        f"/root/autodl-tmp/lensing-4/{role}-child-{index}"
                    ),
                    "parent_root": "/root/autodl-tmp/lensing-4/parent",
                    "parent_manifest_sha256": "d" * 64,
                    "accepted_count": count,
                }
                for index, count in enumerate(counts)
            ],
        },
        "corrected_65k_publication": corrected,
        "terminal_publication": {
            "combined_manifest_sha256": "e" * 64,
            "train_parent_manifest_sha256": "f" * 64,
            "development_tail_manifest_sha256": "0" * 64,
        },
        "publication_roots": {
            "stage_a": "/root/autodl-tmp/lensing-4/stage-a",
            "stage_b": "/root/autodl-tmp/lensing-4/stage-b",
            "combined_base": "/root/autodl-tmp/lensing-4/combined",
            "correction": "/root/autodl-tmp/lensing-4/correction",
            "terminal_train_increment": "/root/autodl-tmp/lensing-4/increment",
            "terminal_combined_131k": "/root/autodl-tmp/lensing-4/terminal",
            "development_tail": "/root/autodl-tmp/lensing-4/tail",
        },
        "immutable_execution": {
            "git_commit": "7" * 40,
            "wheel_path": "/root/autodl-tmp/lensing-4/reference.whl",
            "wheel_filename": "reference.whl",
            "wheel_sha256": "1" * 64,
            "environment_lock_path": "/root/autodl-tmp/lensing-4/env.txt",
            "environment_lock_sha256": "2" * 64,
            "editable_install_authorized": False,
        },
        "reference_output_root": "/root/autodl-tmp/lensing-4/reference-validation",
        "review_scope": _review_scope(role),
        "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        "future_authorization_path": (
            f"configs/execution/phase7_reference_{role}_authorization.yaml"
        ),
        "future_review_path": (
            f"results/phase7/reference_{role}_delegated_review.json"
        ),
        "release_packet_repository_path": (
            f"results/phase7/reference_{role}_release_packet.json"
        ),
    }


def test_reference_release_stack_remains_implementation_only() -> None:
    contract = validate_reference_execution_stack_contract(ROOT)
    authorization = contract["authorization"]
    assert authorization["frozen_addendum"]["canonical_hash"] == REFERENCE_CONFIG_HASH
    flags = authorization["authorization"]
    assert flags["nonauthorizing_release_packet_implementation_authorized"] is True
    assert flags["delegated_review_builder_implementation_authorized"] is True
    assert flags["scientific_reference_bank_access_authorized"] is False
    assert flags["final_evaluation_unsealing_authorized"] is False


def test_catalog_binds_children_to_atomic_parent_without_child_manifest(
    tmp_path: Path,
) -> None:
    validation = {
        "status": CATALOG_STATUS,
        "query_role": "validation",
        "scientific_query_opened": False,
        "reference_executed": False,
        "datasets": [
            _dataset_spec(
                tmp_path,
                parent_name="stage-a",
                dataset_name="validation-child",
                accepted_count=6144,
            )
        ],
    }
    role, datasets = validate_reference_query_catalog(
        validation,
        approved_root=tmp_path,
    )
    assert role == "validation"
    assert len(datasets) == 1

    tail = {
        "status": CATALOG_STATUS,
        "query_role": "balanced_tail_diagnostic",
        "scientific_query_opened": False,
        "reference_executed": False,
        "datasets": [
            _dataset_spec(
                tmp_path,
                parent_name="sealed-final",
                dataset_name=f"tail-{index}",
                accepted_count=1024,
            )
            for index in range(4)
        ],
    }
    role, datasets = validate_reference_query_catalog(tail, approved_root=tmp_path)
    assert role == "balanced_tail_diagnostic"
    assert len(datasets) == 4


def test_release_packet_uses_exact_query_catalog_and_fresh_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    (root / "results/phase7").mkdir(parents=True)
    decision = project / "architecture.json"
    terminal = project / "terminal.json"
    _write_json(decision, {"selected_architecture_id": "nsf-t10-w256"})
    _write_json(terminal, {"decision": "lock_train_131k_saturated"})
    preparation = project / "training/rung-131072/rung_preparation.json"
    _write_json(
        preparation,
        {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": 3,
            "member_ids": ["a", "b", "c"],
        },
    )
    architecture_authorization = root / "configs/execution/architecture.yaml"
    _write_yaml(
        architecture_authorization,
        {
            "authorization_status": (
                "authorized_terminal_131k_architecture_selection_only"
            ),
            "architecture_selection_output_path": str(decision),
            "terminal_decision_path": str(terminal),
            "terminal_decision_sha256": _sha256(terminal),
            "reused_probe_output_root": str(preparation.parents[1]),
            "reused_probe_rung_preparation_sha256": _sha256(preparation),
            "corrected_65k_publication": _packet()["corrected_65k_publication"],
            "terminal_publication": _packet()["terminal_publication"],
            "publication_roots": _packet()["publication_roots"],
        },
    )
    catalog_path = project / "review/query.json"
    _write_json(
        catalog_path,
        {
            "status": CATALOG_STATUS,
            "query_role": "validation",
            "scientific_query_opened": False,
            "reference_executed": False,
            "datasets": [
                _dataset_spec(
                    project,
                    parent_name="stage-a",
                    dataset_name="validation-child",
                    accepted_count=6144,
                )
            ],
        },
    )
    wheel = project / "review/reference.whl"
    wheel.write_bytes(b"wheel")
    wheel_result = project / "review/wheel.json"
    _write_json(wheel_result, {})
    environment = project / "review/environment.txt"
    environment.write_text("environment\n")

    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(release_module, "TRAIN_131K_COUNT", 3)
    monkeypatch.setattr(
        release_module,
        "validate_reference_execution_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(release_module, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        release_module,
        "validate_hashed_terminal_decisions",
        lambda *args, **kwargs: {
            "terminal": {"decision": "lock_train_131k_saturated"},
            "architecture": {
                "architecture_id": "nsf-t10-w256",
                "model_configuration_hash": "3" * 64,
            },
        },
    )
    monkeypatch.setattr(
        release_module,
        "selected_model_configuration",
        lambda *args: {"model": "fixture"},
    )
    monkeypatch.setattr(
        release_module,
        "model_configuration_hash",
        lambda value: "3" * 64,
    )
    monkeypatch.setattr(
        release_module,
        "_immutable_execution",
        lambda **kwargs: {
            "git_commit": kwargs["implementation_commit"],
            "wheel_filename": "reference.whl",
            "wheel_sha256": "4" * 64,
        },
    )
    output_root = project / "outputs/reference"
    packet = build_reference_query_release_packet(
        root,
        implementation_commit="5" * 40,
        architecture_authorization_path=architecture_authorization,
        architecture_decision_path=decision,
        terminal_decision_path=terminal,
        primary_rung_preparation_path=preparation,
        query_catalog_path=catalog_path,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        reference_output_root=output_root,
        output_path=root / "results/phase7/reference_validation_release_packet.json",
    )
    assert packet["query_catalog"]["query_role"] == "validation"
    assert len(packet["query_catalog"]["datasets"]) == 1
    assert packet["reference_query_execution_authorized"] is False
    assert not output_root.exists()


@pytest.mark.parametrize(
    "role",
    ["validation", "iid_test", "balanced_tail_diagnostic"],
)
def test_review_creates_exact_role_specific_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    monkeypatch.setattr(
        release_module,
        "validate_reference_execution_stack_contract",
        lambda value: {},
    )
    packet = _packet(role)
    packet_path = tmp_path / f"results/phase7/reference_{role}_release_packet.json"
    _write_json(packet_path, packet)
    review_path = tmp_path / f"results/phase7/reference_{role}_delegated_review.json"
    _write_json(
        review_path,
        {
            "status": REVIEW_STATUS,
            "reviewed_release_packet_sha256": _sha256(packet_path),
            "reviewed_by": (
                "codex_as_delegated_scientific_and_engineering_reviewer"
            ),
            "review_date": "2026-07-24",
            "authorization_scope": packet["review_scope"],
            "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        },
    )
    output = tmp_path / f"configs/execution/phase7_reference_{role}_authorization.yaml"
    authorization = build_reference_query_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    assert authorization["query_role"] == role
    flags = authorization["authorization"]
    assert flags["validation_reference_execution_authorized"] is (
        role == "validation"
    )
    assert flags["final_evaluation_unsealing_authorized"] is (
        role != "validation"
    )
    assert flags["checkpoint_access_authorized"] is False
    assert authorization["reference_is_exact_likelihood_or_gold"] is False

    changed = deepcopy(json.loads(review_path.read_text()))
    changed["authorization_scope"]["query_count"] += 1
    _write_json(review_path, changed)
    with pytest.raises(TrainingGateError, match="review is not exact"):
        build_reference_query_authorization(
            tmp_path,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output,
        )
