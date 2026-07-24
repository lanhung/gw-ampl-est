from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwlens_mm.training import legacy_sis_authorization as release_module
from gwlens_mm.training import legacy_sis_stress as stress_module
from gwlens_mm.training.legacy_sis_authorization import (
    AUTHORIZATION_STATUS,
    RELEASE_STATUS,
    REVIEW_STATUS,
    build_legacy_sis_authorization,
    build_legacy_sis_release_packet,
    run_authorized_legacy_sis_reproduction,
)
from gwlens_mm.training.legacy_sis_stress import LegacySISStressGateError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _legacy_fixture(legacy_root: Path) -> tuple[Path, Path, dict[str, object]]:
    checkpoint = legacy_root / "runs/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque-checkpoint-never-deserialized")
    predictions = legacy_root / "runs/predictions.csv"
    rows = (
        ("0", 3.0, 3.2, -1.0, -1.2),
        ("1", 5.0, 4.5, -3.0, -2.5),
        ("2", 7.0, 7.1, -5.0, -5.1),
    )
    with predictions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event_id", "mu0_true", "mu0_pred", "mu1_true", "mu1_pred"))
        writer.writerows(rows)
    values = np.asarray(rows, dtype=object)
    truth = values[:, 1].astype(np.float64)
    prediction = values[:, 2].astype(np.float64)
    error = prediction - truth
    identity = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_predictions_path": str(predictions),
        "validation_predictions_sha256": _sha256(predictions),
        "validation_rows": 3,
        "training_entry_sha256": "1" * 64,
        "generator_sha256": "2" * 64,
    }
    metric = {
        "target": "mu0_absolute_magnification",
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_percent": float(np.mean(np.abs(error / truth)) * 100.0),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]),
        "absolute_numeric_tolerance": 1.0e-12,
        "sis_identity_absolute_tolerance": 1.0e-12,
    }
    return checkpoint, predictions, {"identity": identity, "metric": metric}


def _implementation_authorization(
    checkpoint: Path,
    predictions: Path,
    fixture: dict[str, object],
) -> dict[str, object]:
    return {
        "frozen_legacy_identity": fixture["identity"],
        "descriptive_metric_contract": fixture["metric"],
        "claim_boundary": {
            "model_selected_validation_not_independent_test": True,
            "point_prediction_not_posterior": True,
            "sis_only": True,
            "et_single_detector": True,
            "synthetic_gaussian_design_noise": True,
            "incompatible_with_h1_l1_v1_final_input": True,
            "matched_competitor_claim_authorized": False,
            "calibrated_coverage_claim_authorized": False,
            "v2_final_data_application_authorized": False,
        },
    }


def _packet(
    checkpoint: Path,
    predictions: Path,
    fixture: dict[str, object],
    *,
    project_root: Path,
) -> dict[str, object]:
    review_scope = {
        "legacy_asset_read_authorized": True,
        "baseline_execution_authorized": True,
        "legacy_checkpoint_deserialization_authorized": False,
        "legacy_asset_write_authorized": False,
        "scientific_data_access_authorized": False,
        "final_evaluation_access_authorized": False,
        "v2_final_data_application_authorized": False,
        "manuscript_claim_finalization_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "legacy_asset_read_authorized": False,
        "implementation_commit": "3" * 40,
        "frozen_legacy_identity": fixture["identity"],
        "descriptive_metric_contract": fixture["metric"],
        "claim_boundary": _implementation_authorization(
            checkpoint, predictions, fixture
        )["claim_boundary"],
        "immutable_execution": {
            "git_commit": "3" * 40,
            "wheel_path": str(project_root / "review/release.whl"),
            "wheel_filename": "release.whl",
            "wheel_sha256": "4" * 64,
            "environment_lock_path": str(project_root / "review/environment.txt"),
            "environment_lock_sha256": "5" * 64,
            "editable_install_authorized": False,
        },
        "evidence_output_path": str(project_root / "results/phase7/legacy.json"),
        "review_scope": review_scope,
        "read_only_evidence_contract": {
            "checkpoint_deserialization_forbidden": True,
            "before_after_inode_size_mtime_identity_required": True,
            "legacy_write_forbidden": True,
            "output_must_be_new_project_only": True,
        },
        "future_authorization_path": (
            "configs/execution/phase7_legacy_sis_read_only_authorization.yaml"
        ),
        "future_review_path": "results/phase7/legacy_sis_delegated_review.json",
        "release_packet_repository_path": (
            "results/phase7/legacy_sis_release_packet.json"
        ),
    }


def test_release_packet_is_nonauthorizing_and_does_not_read_legacy_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    legacy = tmp_path / "legacy"
    (root / "results/phase7").mkdir(parents=True)
    checkpoint, predictions, fixture = _legacy_fixture(legacy)
    wheel = project / "review/release.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    wheel_result = project / "review/wheel.json"
    _write_json(wheel_result, {})
    environment = project / "review/environment.txt"
    environment.write_text("environment\n")
    evidence = project / "results/phase7/legacy.json"

    monkeypatch.setattr(release_module, "NEW_PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "validate_legacy_sis_stack_contract",
        lambda value: _implementation_authorization(
            checkpoint, predictions, fixture
        ),
    )
    monkeypatch.setattr(release_module, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(
        release_module,
        "_immutable_execution",
        lambda **kwargs: {
            "git_commit": kwargs["implementation_commit"],
            "wheel_sha256": "6" * 64,
        },
    )
    packet = build_legacy_sis_release_packet(
        root,
        implementation_commit="7" * 40,
        wheel_path=wheel,
        exact_wheel_test_result_path=wheel_result,
        environment_lock_path=environment,
        evidence_output_path=evidence,
        output_path=root / "results/phase7/legacy_sis_release_packet.json",
    )
    assert packet["status"] == RELEASE_STATUS
    assert packet["legacy_asset_read_authorized"] is False
    assert packet["review_scope"]["legacy_asset_read_authorized"] is True
    assert not evidence.exists()


def test_delegated_review_creates_exact_read_only_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    project = tmp_path / "project"
    legacy = tmp_path / "legacy"
    checkpoint, predictions, fixture = _legacy_fixture(legacy)
    packet = _packet(
        checkpoint,
        predictions,
        fixture,
        project_root=project,
    )
    packet_path = root / "results/phase7/legacy_sis_release_packet.json"
    _write_json(packet_path, packet)
    review_path = root / "results/phase7/legacy_sis_delegated_review.json"
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
        },
    )
    monkeypatch.setattr(
        release_module,
        "validate_legacy_sis_stack_contract",
        lambda value: {},
    )
    output = (
        root / "configs/execution/phase7_legacy_sis_read_only_authorization.yaml"
    )
    authorization = build_legacy_sis_authorization(
        root,
        release_packet_path=packet_path,
        delegated_review_path=review_path,
        output_path=output,
    )
    assert authorization["authorization_status"] == AUTHORIZATION_STATUS
    flags = authorization["authorization"]
    assert flags["legacy_asset_read_authorized"] is True
    assert flags["legacy_checkpoint_deserialization_authorized"] is False
    assert flags["v2_final_data_application_authorized"] is False

    changed = deepcopy(json.loads(review_path.read_text()))
    changed["authorization_scope"]["legacy_asset_write_authorized"] = True
    _write_json(review_path, changed)
    with pytest.raises(LegacySISStressGateError, match="review is not exact"):
        build_legacy_sis_authorization(
            root,
            release_packet_path=packet_path,
            delegated_review_path=review_path,
            output_path=output,
        )


def test_exact_gate_runs_read_only_and_writes_only_new_project_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    project = tmp_path / "project"
    legacy = tmp_path / "legacy"
    checkpoint, predictions, fixture = _legacy_fixture(legacy)
    checkpoint_before = checkpoint.read_bytes()
    wheel = project / "review/release.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel")
    environment = project / "review/environment.txt"
    environment.write_text("environment\n")
    evidence = project / "results/phase7/legacy.json"
    authorization_path = (
        root / "configs/execution/phase7_legacy_sis_read_only_authorization.yaml"
    )
    _write_yaml(
        authorization_path,
        {
            "authorization_status": AUTHORIZATION_STATUS,
            "frozen_legacy_identity": fixture["identity"],
            "descriptive_metric_contract": fixture["metric"],
            "authorization": _packet(
                checkpoint,
                predictions,
                fixture,
                project_root=project,
            )["review_scope"],
            "immutable_execution": {
                "git_commit": "8" * 40,
                "wheel_path": str(wheel),
                "wheel_sha256": _sha256(wheel),
                "environment_lock_path": str(environment),
                "environment_lock_sha256": _sha256(environment),
                "editable_install_authorized": False,
            },
            "evidence_output_path": str(evidence),
            "post_freeze_allowed_paths": [],
            "stop_after_read_only_reproduction": True,
        },
    )
    monkeypatch.setattr(release_module, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(release_module, "NEW_PROJECT_ROOT", project)
    monkeypatch.setattr(stress_module, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(stress_module, "NEW_PROJECT_ROOT", project)
    monkeypatch.setattr(
        release_module,
        "validate_legacy_sis_stack_contract",
        lambda value: {},
    )
    monkeypatch.setattr(
        release_module,
        "_verify_training_checkout",
        lambda *args: None,
    )
    result = run_authorized_legacy_sis_reproduction(
        root,
        authorization_path=authorization_path,
        checkpoint_path=checkpoint,
        predictions_path=predictions,
        evidence_output_path=evidence,
        execution_commit="8" * 40,
    )
    assert result["checkpoint_deserialized"] is False
    assert result["legacy_assets_byte_identity_preserved"] is True
    assert checkpoint.read_bytes() == checkpoint_before
    assert json.loads(evidence.read_text())["status"] == (
        "legacy_sis_point_regression_descriptive_stress_control_reproduced"
    )
