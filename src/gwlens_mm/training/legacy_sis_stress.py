"""Read-only verifier for the frozen legacy SIS point-regression stress control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np

from ..config import load_yaml

IMPLEMENTATION_AUTHORIZATION = (
    "configs/execution/phase7_legacy_sis_stress_control_authorization.yaml"
)
LEGACY_ROOT = Path("/root/autodl-tmp/tmp")
NEW_PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")


class LegacySISStressGateError(ValueError):
    """Raised when legacy stress-control evidence crosses its frozen boundary."""


@dataclass(frozen=True)
class LegacySISStressContract:
    checkpoint_sha256: str
    predictions_sha256: str
    validation_rows: int
    expected_mae: float
    expected_rmse: float
    expected_mape_percent: float
    expected_pearson: float
    numeric_tolerance: float
    sis_identity_tolerance: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(path: Path) -> Tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def legacy_sis_contract(
    authorization: Mapping[str, Any],
) -> LegacySISStressContract:
    identity = authorization.get("frozen_legacy_identity", {})
    metric = authorization.get("descriptive_metric_contract", {})
    return LegacySISStressContract(
        checkpoint_sha256=str(identity.get("checkpoint_sha256", "")),
        predictions_sha256=str(identity.get("validation_predictions_sha256", "")),
        validation_rows=int(identity.get("validation_rows", -1)),
        expected_mae=float(metric.get("mae", math.nan)),
        expected_rmse=float(metric.get("rmse", math.nan)),
        expected_mape_percent=float(metric.get("mape_percent", math.nan)),
        expected_pearson=float(metric.get("pearson", math.nan)),
        numeric_tolerance=float(metric.get("absolute_numeric_tolerance", math.nan)),
        sis_identity_tolerance=float(
            metric.get("sis_identity_absolute_tolerance", math.nan)
        ),
    )


def validate_legacy_sis_stack_contract(root: Path) -> Mapping[str, Any]:
    """Keep the implementation checkpoint non-executable and claim-safe."""

    authorization = load_yaml(root / IMPLEMENTATION_AUTHORIZATION)
    if authorization.get("authorization_status") != "authorized_implementation_only":
        raise LegacySISStressGateError("legacy SIS implementation gate is absent")
    flags = authorization.get("authorization", {})
    allowed = {
        "verifier_implementation_authorized",
        "exact_read_only_runtime_gate_implementation_authorized",
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed):
        raise LegacySISStressGateError("legacy SIS verifier implementation is incomplete")
    if any(value is not False for name, value in flags.items() if name not in allowed):
        raise LegacySISStressGateError("legacy SIS implementation opened execution")
    boundary = authorization.get("claim_boundary", {})
    required_true = {
        "model_selected_validation_not_independent_test",
        "point_prediction_not_posterior",
        "sis_only",
        "et_single_detector",
        "synthetic_gaussian_design_noise",
        "incompatible_with_h1_l1_v1_final_input",
    }
    required_false = {
        "matched_competitor_claim_authorized",
        "calibrated_coverage_claim_authorized",
        "v2_final_data_application_authorized",
    }
    if any(boundary.get(name) is not True for name in required_true) or any(
        boundary.get(name) is not False for name in required_false
    ):
        raise LegacySISStressGateError("legacy SIS claim boundary drifted")
    contract = legacy_sis_contract(authorization)
    numeric = (
        contract.expected_mae,
        contract.expected_rmse,
        contract.expected_mape_percent,
        contract.expected_pearson,
        contract.numeric_tolerance,
        contract.sis_identity_tolerance,
    )
    if (
        contract.validation_rows != 500
        or any(not math.isfinite(value) for value in numeric)
        or contract.numeric_tolerance <= 0.0
        or contract.sis_identity_tolerance <= 0.0
    ):
        raise LegacySISStressGateError("legacy SIS metric contract is invalid")
    return authorization


def verify_legacy_sis_stress_control(
    checkpoint_path: Path,
    predictions_path: Path,
    contract: LegacySISStressContract,
) -> Mapping[str, Any]:
    """Hash the checkpoint and recompute descriptive metrics from saved predictions."""

    for path in (checkpoint_path, predictions_path):
        resolved = path.resolve()
        if not resolved.is_relative_to(LEGACY_ROOT) or not resolved.is_file():
            raise LegacySISStressGateError("legacy path escaped the immutable root")
    before = {
        "checkpoint": _stat_identity(checkpoint_path),
        "predictions": _stat_identity(predictions_path),
    }
    checkpoint_hash = _sha256(checkpoint_path)
    predictions_hash = _sha256(predictions_path)
    if checkpoint_hash != contract.checkpoint_sha256:
        raise LegacySISStressGateError("legacy checkpoint hash changed")
    if predictions_hash != contract.predictions_sha256:
        raise LegacySISStressGateError("legacy prediction hash changed")
    required = ("event_id", "mu0_true", "mu0_pred", "mu1_true", "mu1_pred")
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise LegacySISStressGateError("legacy prediction columns changed")
        rows = list(reader)
    if len(rows) != contract.validation_rows:
        raise LegacySISStressGateError("legacy validation row count changed")
    identifiers = tuple(row["event_id"] for row in rows)
    if any(not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise LegacySISStressGateError("legacy validation IDs are empty or duplicated")
    values = np.asarray(
        [
            [float(row[name]) for name in required[1:]]
            for row in rows
        ],
        dtype=np.float64,
    )
    if values.shape != (contract.validation_rows, 4) or not np.all(np.isfinite(values)):
        raise LegacySISStressGateError("legacy prediction values are invalid")
    mu0_true, mu0_pred, mu1_true, mu1_pred = values.T
    error = mu0_pred - mu0_true
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_percent": float(np.mean(np.abs(error / mu0_true)) * 100.0),
        "pearson": float(np.corrcoef(mu0_true, mu0_pred)[0, 1]),
    }
    expected = {
        "mae": contract.expected_mae,
        "rmse": contract.expected_rmse,
        "mape_percent": contract.expected_mape_percent,
        "pearson": contract.expected_pearson,
    }
    if any(
        not math.isclose(metrics[name], value, abs_tol=contract.numeric_tolerance)
        for name, value in expected.items()
    ):
        raise LegacySISStressGateError("legacy descriptive metrics did not reproduce")
    true_residual = float(np.max(np.abs(mu0_true + mu1_true - 2.0)))
    predicted_residual = float(np.max(np.abs(mu0_pred + mu1_pred - 2.0)))
    if max(true_residual, predicted_residual) > contract.sis_identity_tolerance:
        raise LegacySISStressGateError("legacy SIS signed-magnification identity failed")
    after = {
        "checkpoint": _stat_identity(checkpoint_path),
        "predictions": _stat_identity(predictions_path),
    }
    if before != after:
        raise LegacySISStressGateError("legacy artifact changed during verification")
    return {
        "status": "legacy_sis_point_regression_descriptive_stress_control_reproduced",
        "validation_rows": len(rows),
        "unique_event_ids": len(set(identifiers)),
        "checkpoint_sha256": checkpoint_hash,
        "predictions_sha256": predictions_hash,
        "metrics": metrics,
        "metric_absolute_tolerance": contract.numeric_tolerance,
        "maximum_true_sis_identity_residual": true_residual,
        "maximum_predicted_sis_identity_residual": predicted_residual,
        "legacy_assets_byte_identity_preserved": True,
        "checkpoint_deserialized": False,
        "model_selected_validation_not_independent_test": True,
        "point_prediction_not_posterior": True,
        "matched_competitor": False,
        "v2_final_data_accessed": False,
        "gwosc_gwtc_accessed": False,
    }


def write_legacy_sis_evidence(output_path: Path, result: Mapping[str, Any]) -> None:
    """Atomically write one small result outside every legacy root."""

    resolved = output_path.resolve()
    if not resolved.is_relative_to(NEW_PROJECT_ROOT) or resolved.is_relative_to(
        LEGACY_ROOT
    ):
        raise LegacySISStressGateError("legacy evidence output escaped the new project")
    if output_path.exists():
        raise FileExistsError("legacy stress-control output identity already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output_path)
