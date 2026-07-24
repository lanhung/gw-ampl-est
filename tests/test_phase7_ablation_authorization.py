from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gwlens_mm.training import ablation_authorization as release_module
from gwlens_mm.training.ablation_authorization import (
    AUTHORIZATION_STATUS,
    CLOSED_BOUNDARIES,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_ablation_authorization,
    build_ablation_release_packet,
    load_ablation_release_stack_contract,
)
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _scope() -> dict[str, object]:
    return {
        "locked_training_rung": 131072,
        "selected_architecture_id": "nsf-t10-w256",
        "ablation_views": ["gw_only", "em_only"],
        "model_seeds": [0, 1, 2],
        "fit_count": 6,
        "maximum_concurrent_fits": 3,
        "data_loader_worker_processes": 4,
        "corrected_65k_data_access_authorized": True,
        "terminal_train_increment_data_access_authorized": True,
        "terminal_131k_combined_reference_access_authorized": True,
        "development_validation_authorized": True,
        "ablation_fit_execution_authorized": True,
    }


def _packet() -> dict[str, object]:
    corrected = {
        "base_generator_commit": "1" * 40,
        "base_preregistration_hash": "2" * 64,
        "correction_generator_commit": "3" * 40,
        "correction_preregistration_hash": "4" * 64,
        "correction_parent_manifest_sha256": "5" * 64,
        "correction_publication_tree_sha256": "6" * 64,
        "combined_base_manifest_sha256": "7" * 64,
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "ablation_fit_execution_authorized": False,
        "implementation_commit": "8" * 40,
        "terminal_decision": {
            "path": "/root/autodl-tmp/lensing-4/terminal.json",
            "sha256": "9" * 64,
            "decision": "lock_train_131k_saturated",
        },
        "selected_architecture": {
            "decision_path": "/root/autodl-tmp/lensing-4/architecture.json",
            "decision_sha256": "a" * 64,
            "architecture_id": "nsf-t10-w256",
            "model_configuration_hash": "b" * 64,
            "locked_training_rung": 131072,
        },
        "ablation_model_configuration_hashes": {
            "gw_only": "c" * 64,
            "em_only": "d" * 64,
        },
        "primary_rung_preparation": {
            "path": "/root/autodl-tmp/lensing-4/preparation.json",
            "sha256": "e" * 64,
        },
        "corrected_65k_publication": corrected,
        "terminal_publication": {
            "combined_manifest_sha256": "f" * 64,
            "train_parent_manifest_sha256": "0" * 64,
            "development_tail_manifest_sha256": "1" * 64,
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
        "final_evaluation_commitment_sha256": "2" * 64,
        "immutable_training": {
            "git_commit": "8" * 40,
            "wheel_path": "/root/autodl-tmp/lensing-4/ablation.whl",
            "wheel_filename": "ablation.whl",
            "wheel_sha256": "3" * 64,
            "environment_lock_path": "/root/autodl-tmp/lensing-4/env.txt",
            "environment_lock_sha256": "4" * 64,
            "editable_install_authorized": False,
        },
        "ablation_output_root": "/root/autodl-tmp/lensing-4/ablations",
        "fit_output_identities": {
            view: {
                str(seed): f"/root/autodl-tmp/lensing-4/ablations/{view}/seed-{seed}"
                for seed in (0, 1, 2)
            }
            for view in ("gw_only", "em_only")
        },
        "review_scope": _scope(),
        "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        "future_authorization_path": (
            "configs/execution/phase7_terminal_ablation_training_authorization.yaml"
        ),
        "future_review_path": "results/phase7/ablation_training_review.json",
        "release_packet_repository_path": (
            "results/phase7/ablation_release_packet.json"
        ),
    }


def test_ablation_release_stack_is_implementation_only() -> None:
    authorization = load_ablation_release_stack_contract(ROOT)
    assert authorization["frozen_contracts"]["exact_fit_count"] == 6
    flags = authorization["authorization"]
    assert flags["nonauthorizing_release_packet_implementation_authorized"] is True
    assert flags["scientific_data_access_authorized"] is False
    assert flags["ablation_fit_execution_authorized"] is False
    assert flags["final_evaluation_unsealing_authorized"] is False


def test_release_packet_allocates_exact_six_fresh_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    (root / "results/phase7").mkdir(parents=True)
    (root / "configs/execution").mkdir(parents=True)
    decision = project / "results/architecture.json"
    terminal = project / "results/terminal.json"
    preparation = project / "training/rung-131072/rung_preparation.json"
    members = [f"system-{index:06d}" for index in range(131072)]
    _write_json(decision, {"selected_architecture_id": "nsf-t10-w256"})
    _write_json(terminal, {"decision": "lock_train_131k_saturated"})
    _write_json(
        preparation,
        {
            "status": "ready_for_authorized_probe_fits",
            "rung_count": 131072,
            "member_ids": members,
            "final_evaluation_commitment_sha256": "5" * 64,
            "membership_sha256": "6" * 64,
            "input_standardizer_sha256": "7" * 64,
            "target_standardizer_sha256": "8" * 64,
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
            "final_evaluation_commitment_sha256": "5" * 64,
            "corrected_65k_publication": _packet()[
                "corrected_65k_publication"
            ],
            "terminal_publication": _packet()["terminal_publication"],
            "publication_roots": _packet()["publication_roots"],
        },
    )
    wheel = project / "review/ablation.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    wheel_result = project / "review/wheel-result.json"
    _write_json(wheel_result, {})
    environment = project / "review/environment.txt"
    environment.write_text("environment\n")

    monkeypatch.setattr(release_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "load_ablation_release_stack_contract",
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
                "model_configuration_hash": "9" * 64,
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
        lambda value: "9" * 64
        if "ablation" not in value
        else ("a" * 64 if value["ablation"] == "gw_only" else "b" * 64),
    )
    monkeypatch.setattr(
        release_module,
        "ablation_model_configuration",
        lambda *args, **kwargs: {"ablation": kwargs["view"]},
    )
    monkeypatch.setattr(
        release_module,
        "_immutable_training",
        lambda **kwargs: {
            "git_commit": kwargs["implementation_commit"],
            "wheel_filename": "ablation.whl",
            "wheel_sha256": "c" * 64,
        },
    )
    output_root = project / "training/ablations"
    packet = build_ablation_release_packet(
        root,
        implementation_commit="d" * 40,
        architecture_authorization_path=architecture_authorization,
        architecture_decision_path=decision,
        terminal_decision_path=terminal,
        primary_rung_preparation_path=preparation,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        ablation_output_root=output_root,
        output_path=root / "results/phase7/ablation_release_packet.json",
    )
    identities = packet["fit_output_identities"]
    assert isinstance(identities, dict)
    assert sum(len(value) for value in identities.values()) == 6
    assert len(
        {
            path
            for seed_outputs in identities.values()
            for path in seed_outputs.values()
        }
    ) == 6
    assert packet["ablation_fit_execution_authorized"] is False
    assert not output_root.exists()


def test_reviewed_packet_creates_only_terminal_six_fit_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_module,
        "load_ablation_release_stack_contract",
        lambda value: {},
    )
    packet_path = tmp_path / "results/phase7/ablation_release_packet.json"
    _write_json(packet_path, _packet())
    review_path = tmp_path / "results/phase7/ablation_training_review.json"
    _write_json(
        review_path,
        {
            "status": REVIEW_STATUS,
            "reviewed_release_packet_sha256": _sha256(packet_path),
            "reviewed_by": (
                "codex_as_delegated_scientific_and_engineering_reviewer"
            ),
            "review_date": "2026-07-24",
            "authorization_scope": _scope(),
            "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        },
    )
    output = (
        tmp_path
        / "configs/execution/phase7_terminal_ablation_training_authorization.yaml"
    )
    authorization = build_ablation_authorization(
        tmp_path,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    assert authorization["locked_training_rung"] == 131072
    assert authorization["authorized_ablation_views"] == ["gw_only", "em_only"]
    assert authorization["authorized_training_seeds"] == [0, 1, 2]
    assert authorization["maximum_fit_count"] == 6
    assert authorization["data_loader_worker_processes"] == 4
    flags = authorization["authorization"]
    assert flags["ablation_fit_execution_authorized"] is True
    assert flags["final_evaluation_unsealing_authorized"] is False
    assert flags["gwosc_gwtc_access_authorized"] is False

    changed = deepcopy(json.loads(review_path.read_text()))
    changed["authorization_scope"]["fit_count"] = 5
    _write_json(review_path, changed)
    with pytest.raises(TrainingGateError, match="review is not exact"):
        build_ablation_authorization(
            tmp_path,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output,
        )


def test_builder_status_matches_existing_terminal_runtime() -> None:
    source = (ROOT / "src/gwlens_mm/training/ablations.py").read_text()
    assert f'"{AUTHORIZATION_STATUS}"' in source
