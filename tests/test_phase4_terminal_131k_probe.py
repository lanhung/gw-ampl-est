from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError
from gwlens_mm.training.engine import TrainingRunIdentity
from gwlens_mm.training.terminal131 import (
    TAIL_STRATA,
    validate_terminal_131k_training_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_training_identity_accepts_only_the_frozen_terminal_rung() -> None:
    identity = TrainingRunIdentity(
        model_configuration_hash="0" * 64,
        training_code_commit="1" * 40,
        training_environment_sha256="2" * 64,
        train_manifest_sha256="3" * 64,
        validation_manifest_sha256="4" * 64,
        final_evaluation_commitment_sha256="5" * 64,
        membership_sha256="6" * 64,
        input_standardizer_sha256="7" * 64,
        target_standardizer_sha256="8" * 64,
        training_rung_count=131072,
        seed=0,
    )
    identity.validate()
    invalid = TrainingRunIdentity(**{**identity.__dict__, "training_rung_count": 262144})
    with pytest.raises(ValueError, match="unregistered rung"):
        invalid.validate()


def test_terminal_probe_execution_remains_closed_without_exact_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(TrainingGateError, match="authorization is absent"):
        validate_terminal_131k_training_gate(
            ROOT,
            authorization_path=(
                ROOT
                / "configs/execution/phase4_terminal_131k_preregistration_authorization.yaml"
            ),
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_base_publication_root=tmp_path / "combined-base",
            correction_publication_root=tmp_path / "correction",
            train_parent_root=tmp_path / "train-increment",
            combined_131k_publication_root=tmp_path / "combined-131k",
            development_tail_parent_root=tmp_path / "tail",
        )


def test_terminal_tail_contract_has_four_distinct_strata() -> None:
    assert len(TAIL_STRATA) == 4
    assert len(set(TAIL_STRATA)) == 4


def test_terminal_runner_dry_plan_is_execution_disabled(tmp_path: Path) -> None:
    from scripts.phase4.run_probe_131k import main

    result_path = tmp_path / "plan.json"
    assert main(["--result", str(result_path)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "implementation_ready_terminal_131k_training_blocked"
    assert result["planned_rung"] == 131072
    assert result["architecture_selection_authorized"] is False
    assert result["extension_above_131072_authorized"] is False
