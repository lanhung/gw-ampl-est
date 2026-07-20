from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gwlens_mm.training.contracts import TrainingGateError, model_configuration_hash
from gwlens_mm.training.terminal_downstream import (
    checkpoint_training_rung_is_authorized,
    dry_run_terminal_downstream_plan,
    validate_hashed_terminal_decisions,
    validate_terminal_architecture_decision,
    validate_terminal_downstream_stack_contract,
    validate_terminal_reference_descriptor,
    validate_terminal_size_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def _terminal_decision(label: str = "lock_train_131k_saturated") -> dict[str, object]:
    return {
        "status": "terminal_learning_curve_decision_complete",
        "comparison": "corrected_train_65k_to_train_131k_terminal",
        "decision": label,
        "selected_training_count": 131072,
        "architecture_selection_review_allowed": True,
        "extension_above_131072_authorized": False,
        "all_three_probe_seeds_retained": True,
        "best_seed_selected": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }


def _architecture_decision() -> dict[str, object]:
    return {
        "status": "architecture_locked_on_development_validation",
        "selected_architecture_id": "nsf-t10-w256",
        "selection_metric": "mean_validation_nlp_across_three_seeds",
        "best_seed_selected": False,
        "total_result_count": 12,
        "new_fit_count": 9,
        "reused_probe_fit_count": 3,
        "calibration_accessed": False,
        "sbc_accessed": False,
        "final_evaluation_accessed": False,
        "opens_later_gate_automatically": False,
    }


def test_terminal_downstream_stack_is_implementation_only() -> None:
    gate = validate_terminal_downstream_stack_contract(ROOT)
    assert gate["authorization"]["authorization_status"] == (
        "authorized_implementation_only"
    )
    assert all(
        value is False
        for value in gate["authorization"]["authorization"].values()
    )
    plan = dry_run_terminal_downstream_plan(ROOT)
    assert plan["locked_training_rung"] == 131072
    assert plan["checkpoint_inference_authorized"] is False
    assert plan["final_evaluation_unsealing_authorized"] is False


@pytest.mark.parametrize(
    "label",
    [
        "lock_train_131k_saturated",
        "lock_train_131k_resource_capped_data_limited",
    ],
)
def test_both_terminal_resource_cap_labels_are_valid(label: str) -> None:
    assert validate_terminal_size_decision(_terminal_decision(label))["decision"] == label


def test_65k_or_automatic_extension_cannot_enter_terminal_downstream() -> None:
    old = _terminal_decision()
    old["decision"] = "lock_train_65k"
    old["selected_training_count"] = 65536
    with pytest.raises(TrainingGateError, match="exact 131k lock"):
        validate_terminal_size_decision(old)
    extension = _terminal_decision()
    extension["extension_above_131072_authorized"] = True
    with pytest.raises(TrainingGateError, match="exact 131k lock"):
        validate_terminal_size_decision(extension)


def test_terminal_architecture_lock_is_twelve_results_and_no_best_seed() -> None:
    result = validate_terminal_architecture_decision(ROOT, _architecture_decision())
    assert result["architecture_id"] == "nsf-t10-w256"
    assert result["model_configuration_hash"] == model_configuration_hash(
        result["model"]
    )
    shortcut = _architecture_decision()
    shortcut["best_seed_selected"] = True
    with pytest.raises(TrainingGateError, match="architecture lock"):
        validate_terminal_architecture_decision(ROOT, shortcut)


def test_131k_checkpoint_requires_an_explicit_131k_authorization() -> None:
    historical = {"selected_architecture": {}}
    terminal = {"selected_architecture": {"locked_training_rung": 131072}}
    assert checkpoint_training_rung_is_authorized(
        {"training_rung_count": 65536}, historical
    )
    assert not checkpoint_training_rung_is_authorized(
        {"training_rung_count": 131072}, historical
    )
    assert checkpoint_training_rung_is_authorized(
        {"training_rung_count": 131072}, terminal
    )
    assert not checkpoint_training_rung_is_authorized(
        {"training_rung_count": 65536}, terminal
    )
    assert not checkpoint_training_rung_is_authorized(
        {"training_rung_count": 262144},
        {"selected_architecture": {"locked_training_rung": 262144}},
    )


def test_terminal_reference_descriptor_binds_counts_and_hashes() -> None:
    descriptor: dict[str, object] = {
        "terminal_combined_manifest_sha256": "1" * 64,
        "terminal_train_increment_parent_manifest_sha256": "2" * 64,
        "corrected_65k_manifest_sha256": "3" * 64,
        "validation_manifest_sha256": "4" * 64,
        "development_tail_manifest_sha256": "5" * 64,
        "logical_train_system_count": 131072,
        "validation_system_count": 6144,
        "development_tail_system_count": 512,
        "strict_corrected_65k_subset": True,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "extension_above_131072_authorized": False,
    }
    assert validate_terminal_reference_descriptor(descriptor) is descriptor
    descriptor["logical_train_system_count"] = 65536
    with pytest.raises(TrainingGateError, match="reference contract"):
        validate_terminal_reference_descriptor(descriptor)


def test_decision_files_are_hash_bound(tmp_path: Path) -> None:
    terminal_path = tmp_path / "terminal.json"
    architecture_path = tmp_path / "architecture.json"
    terminal_path.write_text(json.dumps(_terminal_decision()) + "\n")
    architecture_path.write_text(json.dumps(_architecture_decision()) + "\n")
    result = validate_hashed_terminal_decisions(
        ROOT,
        terminal_decision_path=terminal_path,
        terminal_decision_sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        architecture_decision_path=architecture_path,
        architecture_decision_sha256=hashlib.sha256(
            architecture_path.read_bytes()
        ).hexdigest(),
    )
    assert result["terminal"]["selected_training_count"] == 131072
    assert result["architecture"]["architecture_id"] == "nsf-t10-w256"


def test_terminal_downstream_script_writes_only_a_blocked_plan(tmp_path: Path) -> None:
    from scripts.phase5.prepare_terminal_downstream import main

    result_path = tmp_path / "plan.json"
    assert main(["--root", str(ROOT), "--result", str(result_path)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == (
        "implementation_ready_terminal_downstream_execution_closed"
    )
    assert result["calibration_sbc_materialization_authorized"] is False
    assert result["final_evaluation_materialization_authorized"] is False
