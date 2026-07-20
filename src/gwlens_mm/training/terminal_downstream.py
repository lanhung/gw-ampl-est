"""Shared fail-closed contracts for post-terminal-131k scientific software.

This module is deliberately data-free.  It validates the terminal size and
architecture decisions and prevents a 131k checkpoint from being accepted by
an old 32k/65k authorization merely because the model architecture matches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..provenance import configuration_hash
from .architecture import selected_model_configuration
from .contracts import TrainingGateError, model_configuration_hash
from .terminal131 import TRAIN_131K_COUNT, VALIDATION_COUNT

IMPLEMENTATION_AUTHORIZATION_PATH = (
    "configs/execution/phase5_terminal_downstream_stack_authorization.yaml"
)
TERMINAL_PREREGISTRATION_PATH = (
    "configs/statistics/terminal_131k_preregistration.yaml"
)
TERMINAL_PREREGISTRATION_HASH = (
    "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
)
TERMINAL_DECISIONS = frozenset(
    {
        "lock_train_131k_saturated",
        "lock_train_131k_resource_capped_data_limited",
    }
)
HISTORICAL_LOCKED_RUNGS = frozenset({32768, 65536})
SUPPORTED_LOCKED_RUNGS = frozenset({*HISTORICAL_LOCKED_RUNGS, TRAIN_131K_COUNT})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_terminal_downstream_stack_contract(root: Path) -> Mapping[str, Any]:
    """Prove that terminal downstream work is still implementation-only."""

    preregistration = load_yaml(root / TERMINAL_PREREGISTRATION_PATH)
    authorization = load_yaml(root / IMPLEMENTATION_AUTHORIZATION_PATH)
    frozen = authorization.get("frozen_preregistration", {})
    if (
        authorization.get("authorization_status")
        != "authorized_implementation_only"
        or frozen.get("version") != "1.2.0-rc.1"
        or frozen.get("path") != TERMINAL_PREREGISTRATION_PATH
        or frozen.get("canonical_hash") != TERMINAL_PREREGISTRATION_HASH
        or configuration_hash(preregistration) != TERMINAL_PREREGISTRATION_HASH
    ):
        raise TrainingGateError("terminal downstream implementation gate drifted")
    contract = authorization.get("implementation_contract", {})
    if not (
        contract.get("locked_training_rung") == TRAIN_131K_COUNT
        and contract.get("terminal_decisions") == sorted(TERMINAL_DECISIONS)
        and contract.get("architecture_result_count") == 12
        and contract.get("retained_model_seed_count") == 3
        and contract.get("validation_system_count") == VALIDATION_COUNT
        and contract.get("extension_above_131072_forbidden") is True
    ):
        raise TrainingGateError("terminal downstream implementation contract changed")
    flags = authorization.get("authorization", {})
    if not flags or any(value is not False for value in flags.values()):
        raise TrainingGateError("terminal downstream implementation opened execution")
    return {"preregistration": preregistration, "authorization": authorization}


def validate_terminal_size_decision(
    decision: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Accept only one of the two preregistered terminal resource-cap labels."""

    if (
        decision.get("status") != "terminal_learning_curve_decision_complete"
        or decision.get("comparison")
        != "corrected_train_65k_to_train_131k_terminal"
        or decision.get("decision") not in TERMINAL_DECISIONS
        or int(decision.get("selected_training_count", -1)) != TRAIN_131K_COUNT
        or decision.get("architecture_selection_review_allowed") is not True
        or decision.get("extension_above_131072_authorized") is not False
        or decision.get("all_three_probe_seeds_retained") is not True
        or decision.get("best_seed_selected") is not False
        or decision.get("calibration_accessed") is not False
        or decision.get("final_evaluation_accessed") is not False
    ):
        raise TrainingGateError("terminal downstream gate lacks an exact 131k lock")
    return decision


def validate_terminal_architecture_decision(
    root: Path,
    decision: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the frozen twelve-result, three-seed development-only selection."""

    if (
        decision.get("status") != "architecture_locked_on_development_validation"
        or int(decision.get("total_result_count", -1)) != 12
        or int(decision.get("new_fit_count", -1)) != 9
        or int(decision.get("reused_probe_fit_count", -1)) != 3
        or decision.get("selection_metric")
        != "mean_validation_nlp_across_three_seeds"
        or decision.get("best_seed_selected") is not False
        or decision.get("calibration_accessed") is not False
        or decision.get("sbc_accessed") is not False
        or decision.get("final_evaluation_accessed") is not False
        or decision.get("opens_later_gate_automatically") is not False
    ):
        raise TrainingGateError(
            "terminal downstream gate lacks the exact architecture lock"
        )
    architecture_id = str(decision.get("selected_architecture_id", ""))
    model = selected_model_configuration(root, architecture_id)
    return {
        "decision": decision,
        "architecture_id": architecture_id,
        "model": model,
        "model_configuration_hash": model_configuration_hash(model),
    }


def checkpoint_training_rung_is_authorized(
    identity: Mapping[str, Any], authorization: Mapping[str, Any]
) -> bool:
    """Bind checkpoint rung to an exact authorization.

    Historical authorizations did not carry ``locked_training_rung`` and are
    intentionally limited to their original 32k/65k range.  A 131k checkpoint
    is accepted only when a later authorization explicitly binds 131072.
    """

    try:
        observed = int(identity.get("training_rung_count", -1))
    except (TypeError, ValueError):
        return False
    selected = authorization.get("selected_architecture", {})
    if not isinstance(selected, Mapping):
        return False
    expected_value = selected.get("locked_training_rung")
    if expected_value is None:
        return observed in HISTORICAL_LOCKED_RUNGS
    try:
        expected = int(expected_value)
    except (TypeError, ValueError):
        return False
    return expected in SUPPORTED_LOCKED_RUNGS and observed == expected


def validate_terminal_reference_descriptor(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the small manifest identity block used by future exact gates."""

    hash_fields = (
        "terminal_combined_manifest_sha256",
        "terminal_train_increment_parent_manifest_sha256",
        "corrected_65k_manifest_sha256",
        "validation_manifest_sha256",
        "development_tail_manifest_sha256",
    )
    if any(
        len(str(descriptor.get(name, ""))) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(descriptor.get(name, "")).lower()
        )
        for name in hash_fields
    ):
        raise TrainingGateError("terminal downstream reference hash is invalid")
    if (
        int(descriptor.get("logical_train_system_count", -1)) != TRAIN_131K_COUNT
        or int(descriptor.get("validation_system_count", -1)) != VALIDATION_COUNT
        or int(descriptor.get("development_tail_system_count", -1)) != 512
        or descriptor.get("strict_corrected_65k_subset") is not True
        or descriptor.get("proposal_equals_evaluation") is not True
        or descriptor.get("all_importance_weights_one") is not True
        or descriptor.get("extension_above_131072_authorized") is not False
    ):
        raise TrainingGateError("terminal downstream reference contract changed")
    return descriptor


def validate_hashed_terminal_decisions(
    root: Path,
    *,
    terminal_decision_path: Path,
    terminal_decision_sha256: str,
    architecture_decision_path: Path,
    architecture_decision_sha256: str,
) -> Mapping[str, Any]:
    """Load and hash-bind both future decision artifacts without opening data."""

    import json

    if (
        not terminal_decision_path.is_file()
        or _sha256(terminal_decision_path) != terminal_decision_sha256
        or not architecture_decision_path.is_file()
        or _sha256(architecture_decision_path) != architecture_decision_sha256
    ):
        raise TrainingGateError("terminal downstream decision identity mismatch")
    terminal = json.loads(terminal_decision_path.read_text(encoding="utf-8"))
    architecture = json.loads(architecture_decision_path.read_text(encoding="utf-8"))
    if not isinstance(terminal, dict) or not isinstance(architecture, dict):
        raise TrainingGateError("terminal downstream decision is not a mapping")
    return {
        "terminal": validate_terminal_size_decision(terminal),
        "architecture": validate_terminal_architecture_decision(root, architecture),
    }


def dry_run_terminal_downstream_plan(root: Path) -> Mapping[str, Any]:
    """Describe the post-lock handoff while keeping every scientific gate closed."""

    validate_terminal_downstream_stack_contract(root)
    return {
        "status": "implementation_ready_terminal_downstream_execution_closed",
        "locked_training_rung": TRAIN_131K_COUNT,
        "allowed_terminal_decisions": sorted(TERMINAL_DECISIONS),
        "architecture_result_count": 12,
        "retained_model_seed_count": 3,
        "calibration_sbc_materialization_authorized": False,
        "checkpoint_inference_authorized": False,
        "final_evaluation_materialization_authorized": False,
        "final_evaluation_unsealing_authorized": False,
        "ablation_training_authorized": False,
        "reference_execution_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }
