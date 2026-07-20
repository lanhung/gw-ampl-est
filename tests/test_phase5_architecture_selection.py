from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training import architecture as architecture_module
from gwlens_mm.training.architecture import (
    ALL_ARCHITECTURE_IDS,
    NEW_ARCHITECTURE_IDS,
    PROBE_ARCHITECTURE_ID,
    _locked_preparation_manifest_sha256,
    _locked_train_manifest_sha256,
    _locked_training_dataset,
    architecture_execution_evidence,
    candidate_model_configuration,
    load_architecture_specs,
    select_architecture,
    selected_model_configuration,
    validate_architecture_execution_gate,
)
from gwlens_mm.training.contracts import TrainingGateError, model_configuration_hash
from gwlens_mm.training.data import (
    CombinedTrainingPublication,
    CorrectedTrainingPublication,
    StageAPublication,
)
from gwlens_mm.training.engine import TrainingRunIdentity, _validate_execution_evidence
from gwlens_mm.training.terminal_architecture import (
    validate_terminal_architecture_execution_gate,
)

ROOT = Path(__file__).resolve().parents[1]
GRID_HASH = "abb3ef575e0f37a8f0150169391efb350b1c53893508bf8ba2505f9219075355"


def test_architecture_grid_is_exact_and_candidate_hashes_are_distinct() -> None:
    assert configuration_hash(
        load_yaml(ROOT / "configs/models/phase5_architecture_grid.yaml")
    ) == GRID_HASH
    specifications = load_architecture_specs(ROOT)
    assert tuple(spec.architecture_id for spec in specifications) == (
        "nsf-t06-w128",
        "nsf-t06-w256",
        "nsf-t10-w128",
        "nsf-t10-w256",
    )
    assert sum(spec.reused_probe for spec in specifications) == 1
    assert next(spec for spec in specifications if spec.reused_probe).architecture_id == (
        PROBE_ARCHITECTURE_ID
    )
    hashes = {
        model_configuration_hash(candidate_model_configuration(ROOT, specification))
        for specification in specifications
    }
    assert len(hashes) == 4
    assert set(NEW_ARCHITECTURE_IDS) == {
        spec.architecture_id for spec in specifications if not spec.reused_probe
    }
    for specification in specifications:
        selected = selected_model_configuration(ROOT, specification.architecture_id)
        expected = (
            load_yaml(ROOT / "configs/models/phase4_probe_nsf.yaml")
            if specification.reused_probe
            else candidate_model_configuration(ROOT, specification)
        )
        assert model_configuration_hash(selected) == model_configuration_hash(expected)


def test_candidate_configuration_changes_only_frozen_grid_axes_and_identity() -> None:
    base = load_yaml(ROOT / "configs/models/phase4_probe_nsf.yaml")
    specification = next(
        spec
        for spec in load_architecture_specs(ROOT)
        if spec.architecture_id == "nsf-t06-w128"
    )
    candidate = candidate_model_configuration(ROOT, specification)
    assert candidate["architecture"]["flow"]["transforms"] == 6
    assert candidate["architecture"]["flow"]["conditioner_width"] == 128
    assert candidate["implementation_id"] == "phase5_nsf-t06-w128_v1"
    restored = dict(candidate)
    restored["implementation_id"] = base["implementation_id"]
    restored["architecture"] = dict(candidate["architecture"])
    restored["architecture"]["flow"] = dict(candidate["architecture"]["flow"])
    restored["architecture"]["flow"]["transforms"] = 10
    restored["architecture"]["flow"]["conditioner_width"] = 256
    assert restored == base


def test_architecture_fit_uses_common_engine_authorization_envelope() -> None:
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
        training_rung_count=65536,
        seed=0,
    )
    evidence = architecture_execution_evidence(
        identity,
        architecture="nsf-t06-w128",
        immutable_wheel_sha256="9" * 64,
    )
    _validate_execution_evidence(evidence, identity)
    assert evidence["status"] == "authorized_probe_training"
    assert evidence["fit_role"] == "authorized_architecture_fit"
    assert evidence["reused_probe_retraining"] is False


def _results(*, tie: bool = False) -> list[dict[str, object]]:
    rows = []
    means = {
        "nsf-t06-w128": 0.20,
        "nsf-t06-w256": 0.19,
        "nsf-t10-w128": 0.18,
        "nsf-t10-w256": 0.17,
    }
    parameters = {
        "nsf-t06-w128": 100,
        "nsf-t06-w256": 200,
        "nsf-t10-w128": 150,
        "nsf-t10-w256": 300,
    }
    for architecture in ALL_ARCHITECTURE_IDS:
        for seed in (0, 1, 2):
            rows.append(
                {
                    "architecture_id": architecture,
                    "seed": seed,
                    "trainable_parameter_count": parameters[architecture],
                    "development_mean_nlp_nat_per_target_dimension": (
                        0.2 if tie else means[architecture] + seed * 0.001
                    ),
                }
            )
    return rows


def test_selection_uses_three_seed_mean_and_never_selects_a_seed() -> None:
    result = select_architecture(_results())
    assert result["selected_architecture_id"] == PROBE_ARCHITECTURE_ID
    assert result["best_seed_selected"] is False
    assert result["new_fit_count"] == 9
    assert result["reused_probe_fit_count"] == 3
    assert result["calibration_accessed"] is False
    assert result["final_evaluation_accessed"] is False


def test_exact_tie_uses_lower_parameter_count() -> None:
    result = select_architecture(_results(tie=True))
    assert result["selected_architecture_id"] == "nsf-t06-w128"


def test_selection_rejects_missing_seed_or_best_seed_shortcut() -> None:
    rows = _results()
    rows.pop()
    with pytest.raises(TrainingGateError, match="exactly twelve"):
        select_architecture(rows)


def test_implementation_gate_cannot_execute_architecture_selection(tmp_path: Path) -> None:
    authorization_path = (
        ROOT / "configs/execution/phase5_architecture_selection_stack_authorization.yaml"
    )
    authorization = load_yaml(authorization_path)
    assert authorization["authorization_status"] == "authorized_implementation_only"
    assert authorization["authorization"]["scientific_data_access_authorized"] is False
    assert authorization["authorization"]["architecture_fit_execution_authorized"] is False
    assert (
        authorization["authorization"]["architecture_selection_execution_authorized"]
        is False
    )
    with pytest.raises(TrainingGateError, match="execution authorization is absent"):
        validate_architecture_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_publication_root=tmp_path / "combined",
            terminal_decision_path=tmp_path / "decision.json",
            probe_output_root=tmp_path / "probe",
        )


def test_terminal_architecture_implementation_gate_cannot_open_data(
    tmp_path: Path,
) -> None:
    authorization_path = (
        ROOT
        / "configs/execution/phase5_terminal_131k_architecture_stack_authorization.yaml"
    )
    authorization = load_yaml(authorization_path)
    assert authorization["authorization_status"] == "authorized_implementation_only"
    assert all(value is False for value in authorization["authorization"].values())
    with pytest.raises(TrainingGateError, match="execution authorization is absent"):
        validate_terminal_architecture_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_base_publication_root=tmp_path / "combined-base",
            correction_publication_root=tmp_path / "correction",
            train_parent_root=tmp_path / "train-increment",
            combined_131k_publication_root=tmp_path / "combined-131k",
            development_tail_parent_root=tmp_path / "tail",
            terminal_decision_path=tmp_path / "decision.json",
            probe_output_root=tmp_path / "probe",
        )


def test_terminal_architecture_runner_defaults_to_blocked(tmp_path: Path) -> None:
    from scripts.phase5.run_terminal_architecture_fit import main

    result_path = tmp_path / "result.json"
    assert main(["--result", str(result_path)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["locked_rung"] == 131072
    assert result["new_fit_count"] == 9
    assert result["reused_probe_fit_count"] == 3
    assert result["architecture_selection_executed"] is False
    assert result["extension_above_131072_authorized"] is False


def _publication_identities(tmp_path: Path) -> tuple[
    CombinedTrainingPublication, CorrectedTrainingPublication
]:
    stage_a = StageAPublication(
        parent_root=tmp_path / "stage-a",
        manifest_path=tmp_path / "stage-a" / "dataset_manifest.json",
        manifest_sha256="a" * 64,
        generator_commit="b" * 40,
        preregistration_hash="c" * 64,
        train_dataset_id="train-a",
        validation_dataset_id="validation-a",
        train_root=tmp_path / "train-a",
        validation_root=tmp_path / "validation-a",
        namespace_manifest_sha256={"train": "d" * 64, "validation": "e" * 64},
    )
    combined = CombinedTrainingPublication(
        combined_root=tmp_path / "combined",
        combined_manifest_path=tmp_path / "combined" / "dataset_manifest.json",
        combined_manifest_sha256="f" * 64,
        stage_a=stage_a,
        stage_b_parent_root=tmp_path / "stage-b",
        stage_b_parent_manifest_path=tmp_path / "stage-b" / "dataset_manifest.json",
        stage_b_parent_manifest_sha256="1" * 64,
        stage_b_dataset_id="train-b",
        stage_b_train_root=tmp_path / "train-b",
        train_manifest_sha256="2" * 64,
    )
    corrected = CorrectedTrainingPublication(
        correction_root=tmp_path / "correction",
        correction_manifest_path=tmp_path / "correction" / "dataset_manifest.json",
        correction_manifest_sha256="3" * 64,
        correction_tree_sha256="4" * 64,
        stage_a=stage_a,
        combined_base=combined,
        stage_a_excluded_ids=("bad-a",),
        stage_b_excluded_ids=("bad-b",),
        stage_a_replacement_root=tmp_path / "replacement-a",
        stage_b_replacement_root=tmp_path / "replacement-b",
        stage_a_replacement_ids=("replacement-a",),
        stage_b_replacement_ids=("replacement-b",),
        corrected_stage_a_train_manifest_sha256="5" * 64,
        corrected_combined_train_manifest_sha256="6" * 64,
    )
    return combined, corrected


def test_architecture_identity_helpers_never_fall_back_to_bad_base_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined, corrected = _publication_identities(tmp_path)
    assert _locked_train_manifest_sha256(combined) == "2" * 64
    assert _locked_preparation_manifest_sha256(combined) == "f" * 64
    assert _locked_train_manifest_sha256(corrected) == "6" * 64
    assert _locked_preparation_manifest_sha256(corrected) == "3" * 64

    sentinel = object()
    monkeypatch.setattr(
        architecture_module,
        "corrected_65k_training_dataset",
        lambda publication, curves: sentinel,
    )
    assert _locked_training_dataset(corrected, {}) is sentinel


def test_corrected_architecture_gate_requires_explicit_overlay_path(
    tmp_path: Path,
) -> None:
    decision = {
        "decision": "lock_train_65k",
        "comparison": "train_32k_to_train_65k",
        "extension_above_65536_authorized": False,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision) + "\n")
    flags = {
        "stage_a_data_access_authorized": True,
        "stage_b_data_access_authorized": True,
        "replacement_data_access_authorized": True,
        "new_architecture_fit_execution_authorized": True,
        "architecture_selection_execution_authorized": True,
        "probe_architecture_retraining_authorized": False,
        "best_seed_selection_authorized": False,
        "model_tuning_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_65536_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }
    authorization = {
        "authorization_status": "authorized_corrected_architecture_selection_only",
        "authorization": flags,
        "locked_training_rung": 65536,
        "authorized_new_architecture_ids": list(NEW_ARCHITECTURE_IDS),
        "authorized_training_seeds": [0, 1, 2],
        "terminal_decision_path": str(decision_path),
        "terminal_decision_sha256": hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest(),
    }
    authorization_path = tmp_path / "authorization.yaml"
    authorization_path.write_text(json.dumps(authorization) + "\n")
    with pytest.raises(TrainingGateError, match="requires the correction publication"):
        validate_architecture_execution_gate(
            ROOT,
            authorization_path=authorization_path,
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_publication_root=tmp_path / "combined",
            terminal_decision_path=decision_path,
            probe_output_root=tmp_path / "probe",
        )
