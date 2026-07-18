from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gwlens_mm.schema import SplitName
from gwlens_mm.training.calibration_inference import (
    score_inference_seed,
    validate_score_inference_authorization,
)
from gwlens_mm.training.contracts import TrainingGateError

ROOT = Path(__file__).resolve().parents[1]


def test_score_inference_uses_six_disjoint_split_seed_namespaces() -> None:
    contract = {
        "root_seed_by_split_and_model_seed": {
            "calibration_fit": {"0": 101, "1": 102, "2": 103},
            "sbc_diagnostic": {"0": 201, "1": 202, "2": 203},
        }
    }
    assert (
        score_inference_seed(
            contract, split=SplitName.CALIBRATION_FIT, model_seed=1
        )
        == 102
    )
    changed = json.loads(json.dumps(contract))
    changed["root_seed_by_split_and_model_seed"]["sbc_diagnostic"]["0"] = 101
    with pytest.raises(TrainingGateError, match="collide"):
        score_inference_seed(
            changed, split=SplitName.SBC_DIAGNOSTIC, model_seed=0
        )


def test_current_implementation_gate_cannot_open_checkpoint_inference(
    tmp_path: Path,
) -> None:
    with pytest.raises(TrainingGateError, match="not authorized"):
        validate_score_inference_authorization(
            ROOT,
            authorization_path=ROOT
            / "configs/execution/phase6_calibration_sbc_materialization_stack_authorization.yaml",
            split=SplitName.CALIBRATION_FIT,
            seed=0,
            checkpoint_path=tmp_path / "checkpoint.ckpt",
            publication_root=tmp_path / "publication",
            output_path=tmp_path / "scores.npz",
        )


def test_score_inference_cli_is_dry_by_default(tmp_path: Path) -> None:
    output = tmp_path / "scores.npz"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase6/run_calibration_sbc_inference.py"),
            "--authorization",
            str(tmp_path / "missing.yaml"),
            "--publication-root",
            str(tmp_path / "publication"),
            "--checkpoint",
            str(tmp_path / "checkpoint.ckpt"),
            "--environment-lock",
            str(tmp_path / "environment.json"),
            "--psd-root",
            str(tmp_path / "psd"),
            "--output",
            str(output),
            "--split",
            "calibration_fit",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "checkpoint_score_inference_implementation_ready"
    assert result["score_artifact_created"] is False
    assert result["calibration_map_fitted"] is False
    assert result["sbc_statistical_test_executed"] is False
    assert not output.exists()
