"""Machine-readable training gates and deterministic rung membership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ..config import load_yaml
from ..provenance import configuration_hash

RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"
RC3_HASH = "6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927"
PROBE_SEED_DOMAIN = "adaptive_scientific_split_assignment_v1"
TRAIN_32K_COUNT = 32768
PROBE_16K_COUNT = 16384


class TrainingGateError(ValueError):
    """Raised when implementation or scientific training is not authorized."""


def load_training_stack_contract(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and validate the implementation-only authorization and model candidate."""

    authorization = load_yaml(
        root / "configs/execution/phase4_probe_training_stack_authorization.yaml"
    )
    model = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_engineering_smoke_only"
    ):
        raise TrainingGateError("training-stack implementation authorization is absent")
    flags = authorization["authorization"]
    required_true = {
        "training_stack_implementation_authorized",
        "lazy_reader_implementation_authorized",
        "whitening_implementation_authorized",
        "model_implementation_authorized",
        "unit_and_integration_tests_authorized",
    }
    if any(flags.get(name) is not True for name in required_true):
        raise TrainingGateError("training-stack implementation authorization is incomplete")
    required_false = {
        "stage_a_data_access_authorized",
        "scientific_probe_training_authorized",
        "model_selection_authorized",
        "hyperparameter_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "scientific_evaluation_authorized",
        "final_evaluation_access_authorized",
        "gwosc_gwtc_access_authorized",
        "real_noise_authorized",
    }
    if any(flags.get(name) is not False for name in required_false):
        raise TrainingGateError("implementation-only authorization opened a scientific gate")
    frozen = authorization["frozen_scientific_contracts"]
    if frozen["direct_target_preregistration_hash"] != RC4_HASH:
        raise TrainingGateError("implementation authorization references the wrong RC.4 hash")
    if frozen["adaptive_preregistration_hash"] != RC3_HASH:
        raise TrainingGateError("implementation authorization references the wrong RC.3 hash")
    contract = model["scientific_contract"]
    if contract["direct_target_preregistration_hash"] != RC4_HASH:
        raise TrainingGateError("model candidate references the wrong RC.4 hash")
    if contract["adaptive_preregistration_hash"] != RC3_HASH:
        raise TrainingGateError("model candidate references the wrong RC.3 hash")
    execution = model["execution"]
    if any(
        execution.get(name) is not False
        for name in (
            "scientific_training_enabled",
            "model_selection_enabled",
            "stage_a_access_enabled",
        )
    ):
        raise TrainingGateError("model candidate must remain execution-disabled")
    return authorization, model


def _rank_digest(root_seed: int, physical_system_id: str) -> bytes:
    if root_seed < 0 or not physical_system_id:
        raise ValueError("rank inputs must contain a nonnegative seed and nonempty system ID")
    message = f"{PROBE_SEED_DOMAIN}\0{root_seed}\0{physical_system_id}".encode()
    return hashlib.sha256(message).digest()


def deterministic_probe_subset(
    physical_system_ids: Sequence[str],
    *,
    root_seed: int,
    full_rung_count: int = TRAIN_32K_COUNT,
    probe_count: int = PROBE_16K_COUNT,
) -> Tuple[str, ...]:
    """Choose the frozen 16k subset only after all 32k IDs are available.

    Returning a subset from a partial generation prefix would make the membership depend
    on publication order, which is explicitly forbidden by RC.3/RC.4.
    """

    ids = tuple(physical_system_ids)
    if len(ids) != full_rung_count:
        raise TrainingGateError(
            f"probe membership requires exactly {full_rung_count} completed train IDs"
        )
    if len(set(ids)) != len(ids):
        raise TrainingGateError("training rung contains duplicate physical-system IDs")
    if not 0 < probe_count < full_rung_count:
        raise ValueError("probe count must be a strict positive subset of the full rung")
    ranked = sorted(ids, key=lambda item: (_rank_digest(root_seed, item), item))
    return tuple(ranked[:probe_count])


def validate_direct_target_unit_weights(records: Iterable[Mapping[str, Any]]) -> int:
    """Validate q=p and unit weights without exposing density fields to the model."""

    count = 0
    for record in records:
        distribution = record["provenance"]["distribution"]
        proposal = float(distribution["proposal_log_probability"])
        evaluation = float(distribution["evaluation_prior_log_probability"])
        weight = float(distribution["importance_weight"])
        if proposal != evaluation or weight != 1.0:
            raise TrainingGateError("direct-target record violates exact q=p/unit weights")
        if distribution.get("weight_valid") is not True:
            raise TrainingGateError("direct-target record has an invalid weight")
        if distribution.get("clipping_applied") is not False:
            raise TrainingGateError("direct-target weights may not be clipped")
        count += 1
    return count


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def validate_scientific_training_gate(
    root: Path,
    *,
    authorization_path: Path,
    train_publication_root: Path,
    validation_publication_root: Path,
) -> Mapping[str, Any]:
    """Require a later explicit gate before any Stage A optimization is allowed."""

    load_training_stack_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != "authorized_probe_training_only":
        raise TrainingGateError("scientific probe-training authorization is absent")
    flags = authorization.get("authorization", {})
    if flags.get("scientific_probe_training_authorized") is not True:
        raise TrainingGateError("scientific probe training remains closed")
    for forbidden in (
        "model_tuning_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise TrainingGateError(f"probe gate must keep {forbidden} false")
    commitment_path = root / "results/phase4/final_evaluation_commitment.json"
    commitment = _load_json(commitment_path)
    if commitment.get("commitment_status") != "finalized_before_training":
        raise TrainingGateError("final-evaluation generation commitment is not finalized")
    if commitment.get("future_scientific_generator_commit") is None:
        raise TrainingGateError("final-evaluation commitment has no frozen generator commit")
    expected_commitment_hash = authorization.get("final_evaluation_commitment_sha256")
    actual_commitment_hash = hashlib.sha256(commitment_path.read_bytes()).hexdigest()
    if expected_commitment_hash != actual_commitment_hash:
        raise TrainingGateError("final-evaluation commitment hash mismatch")
    for name, path in (
        ("train", train_publication_root),
        ("validation", validation_publication_root),
    ):
        if "published" not in path.parts or not path.is_dir():
            raise TrainingGateError(f"{name} input is not an atomic published dataset")
        if not (path / "dataset_manifest.json").is_file():
            raise TrainingGateError(f"{name} publication manifest is absent")
    return authorization


def model_configuration_hash(model: Mapping[str, Any]) -> str:
    """Return the canonical candidate-model configuration identity."""

    return configuration_hash(model)
