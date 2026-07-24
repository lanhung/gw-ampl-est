from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.training import legacy_sis_stress as stress_module
from gwlens_mm.training.legacy_sis_stress import (
    LegacySISStressContract,
    LegacySISStressGateError,
    validate_legacy_sis_stack_contract,
    verify_legacy_sis_stress_control,
    write_legacy_sis_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, LegacySISStressContract]:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"opaque-checkpoint-never-deserialized")
    predictions = tmp_path / "predictions.csv"
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
    return checkpoint, predictions, LegacySISStressContract(
        checkpoint_sha256=_sha256(checkpoint),
        predictions_sha256=_sha256(predictions),
        validation_rows=3,
        expected_mae=float(np.mean(np.abs(error))),
        expected_rmse=float(np.sqrt(np.mean(np.square(error)))),
        expected_mape_percent=float(np.mean(np.abs(error / truth)) * 100.0),
        expected_pearson=float(np.corrcoef(truth, prediction)[0, 1]),
        numeric_tolerance=1.0e-12,
        sis_identity_tolerance=1.0e-12,
    )


def test_legacy_sis_implementation_gate_keeps_execution_closed() -> None:
    authorization = validate_legacy_sis_stack_contract(ROOT)
    assert authorization["authorization_status"] == "authorized_implementation_only"
    flags = authorization["authorization"]
    allowed = {
        "verifier_implementation_authorized",
        "exact_read_only_runtime_gate_implementation_authorized",
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    assert all(flags[name] is True for name in allowed)
    assert all(value is False for name, value in flags.items() if name not in allowed)
    boundary = authorization["claim_boundary"]
    assert boundary["point_prediction_not_posterior"] is True
    assert boundary["matched_competitor_claim_authorized"] is False
    assert boundary["v2_final_data_application_authorized"] is False


def test_legacy_sis_verifier_reproduces_metrics_without_loading_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, predictions, contract = _fixture(tmp_path)
    monkeypatch.setattr(stress_module, "LEGACY_ROOT", tmp_path)
    before = checkpoint.read_bytes()
    result = verify_legacy_sis_stress_control(checkpoint, predictions, contract)
    assert result["validation_rows"] == 3
    assert result["unique_event_ids"] == 3
    assert result["checkpoint_deserialized"] is False
    assert result["legacy_assets_byte_identity_preserved"] is True
    assert result["point_prediction_not_posterior"] is True
    assert result["matched_competitor"] is False
    assert checkpoint.read_bytes() == before


def test_legacy_sis_verifier_rejects_hash_and_duplicate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, predictions, contract = _fixture(tmp_path)
    monkeypatch.setattr(stress_module, "LEGACY_ROOT", tmp_path)
    bad_hash = LegacySISStressContract(
        **{**contract.__dict__, "predictions_sha256": "0" * 64}
    )
    with pytest.raises(LegacySISStressGateError, match="prediction hash"):
        verify_legacy_sis_stress_control(checkpoint, predictions, bad_hash)
    text = predictions.read_text(encoding="utf-8")
    predictions.write_text(text.replace("1,5.0", "0,5.0"), encoding="utf-8")
    duplicate = LegacySISStressContract(
        **{**contract.__dict__, "predictions_sha256": _sha256(predictions)}
    )
    with pytest.raises(LegacySISStressGateError, match="duplicated"):
        verify_legacy_sis_stress_control(checkpoint, predictions, duplicate)


def test_legacy_sis_evidence_writer_is_new_root_only_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new_root = tmp_path / "new"
    legacy_root = tmp_path / "legacy"
    monkeypatch.setattr(stress_module, "NEW_PROJECT_ROOT", new_root)
    monkeypatch.setattr(stress_module, "LEGACY_ROOT", legacy_root)
    output = new_root / "evidence/result.json"
    write_legacy_sis_evidence(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    assert not output.with_name("result.json.partial").exists()
    with pytest.raises(FileExistsError):
        write_legacy_sis_evidence(output, {"status": "again"})
    with pytest.raises(LegacySISStressGateError, match="escaped"):
        write_legacy_sis_evidence(legacy_root / "forbidden.json", {"status": "bad"})


def test_legacy_sis_command_defaults_to_asset_read_blocked() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase7/verify_legacy_sis_stress_control.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "implementation_ready_legacy_asset_read_blocked"
    assert result["legacy_asset_read"] is False
    assert result["checkpoint_deserialized"] is False
    assert result["v2_final_data_accessed"] is False
