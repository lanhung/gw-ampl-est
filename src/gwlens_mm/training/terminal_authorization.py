"""Construct one exact terminal-probe authorization after separate review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from .contracts import TrainingGateError
from .terminal131 import (
    SEEDS,
    TAIL_COUNT,
    TERMINAL_PREREGISTRATION_PATH,
    TERMINAL_RELEASE_REVIEW_ACCEPTANCE,
    TERMINAL_RELEASE_REVIEW_STATUS,
    TRAIN_131K_COUNT,
    TRAIN_INCREMENT_COUNT,
    validate_terminal_probe_release_binding,
)

TERMINAL_CONFIG_PATH = "configs/data/phase4_terminal_131k.yaml"
DELEGATED_REVIEW_STATUS = "delegated_terminal_probe_authorization_approved"
AUTHORIZATION_STATUS = "authorized_terminal_131k_probe_only"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
CLOSED_BOUNDARIES = (
    "model_tuning_authorized",
    "architecture_selection_authorized",
    "calibration_authorized",
    "sbc_authorized",
    "final_evaluation_authorized",
    "extension_above_131072_authorized",
    "real_noise_authorized",
    "gwosc_gwtc_access_authorized",
)
REVIEW_SCOPE_FIELDS = (
    "training_rung",
    "training_seeds",
    "publication_data_access_authorized",
    "probe_optimizer_execution_authorized",
    "terminal_learning_curve_decision_authorized",
    "retained_65k_output_root",
    "training_output_root",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def _validate_review(
    review: Mapping[str, Any], *, packet_sha256: str
) -> Mapping[str, Any]:
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if not isinstance(scope, Mapping) or not isinstance(closed, Mapping):
        raise TrainingGateError("terminal delegated review lacks explicit boundaries")
    if (
        set(scope) != set(REVIEW_SCOPE_FIELDS)
        or set(closed) != set(CLOSED_BOUNDARIES)
        or review.get("status") != DELEGATED_REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_sha256
        or not str(review.get("reviewed_by", ""))
        or int(scope.get("training_rung", -1)) != TRAIN_131K_COUNT
        or scope.get("training_seeds") != list(SEEDS)
        or scope.get("publication_data_access_authorized") is not True
        or scope.get("probe_optimizer_execution_authorized") is not True
        or scope.get("terminal_learning_curve_decision_authorized") is not True
    ):
        raise TrainingGateError("terminal delegated review does not approve exact scope")
    for key in CLOSED_BOUNDARIES:
        if closed.get(key) is not False:
            raise TrainingGateError(f"terminal delegated review must keep {key}=false")
    return review


def build_terminal_probe_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    authorization_output_path: Path,
    retained_65k_output_root: Path,
    training_output_root: Path,
) -> Mapping[str, Any]:
    """Create exact authorization content; never infer delegated approval."""

    packet_path = release_packet_path.resolve()
    review_path = delegated_review_path.resolve()
    output_path = authorization_output_path.resolve()
    packet_hash = _sha256(packet_path)
    packet = _load_json(packet_path)
    review = _validate_review(_load_json(review_path), packet_sha256=packet_hash)
    if (
        packet.get("status") != TERMINAL_RELEASE_REVIEW_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("optimizer_execution_authorized") is not False
        or packet.get("authorized_training_rungs_preview") != [TRAIN_131K_COUNT]
        or packet.get("authorized_training_seeds_preview") != list(SEEDS)
    ):
        raise TrainingGateError("terminal release packet is not review-ready")
    try:
        relative_output = output_path.relative_to(root.resolve())
    except ValueError as error:
        raise TrainingGateError("terminal authorization must remain inside repository") from error
    if relative_output.parts[:2] != ("configs", "execution") or output_path.suffix not in {
        ".yaml",
        ".yml",
    }:
        raise TrainingGateError("terminal authorization output path is invalid")
    retained_root = retained_65k_output_root.resolve()
    training_root = training_output_root.resolve()
    if (
        not retained_root.is_relative_to(PROJECT_ROOT)
        or not training_root.is_relative_to(PROJECT_ROOT)
        or retained_root == training_root
    ):
        raise TrainingGateError("terminal training output identities are invalid")
    reviewed_scope = review["authorization_scope"]
    if (
        Path(str(reviewed_scope["retained_65k_output_root"])).resolve()
        != retained_root
        or Path(str(reviewed_scope["training_output_root"])).resolve()
        != training_root
    ):
        raise TrainingGateError("terminal delegated review output identities changed")

    closeout_path = Path(str(packet.get("closeout_result_path", ""))).resolve()
    if (
        not closeout_path.is_file()
        or _sha256(closeout_path) != packet.get("closeout_result_sha256")
    ):
        raise TrainingGateError("terminal packet closeout evidence changed")
    closeout = _load_json(closeout_path)
    publication = packet.get("publication", {})
    immutable = packet.get("immutable_training", {})
    if not isinstance(publication, Mapping) or not isinstance(immutable, Mapping):
        raise TrainingGateError("terminal release packet identities are malformed")
    if (
        closeout.get("status") != "terminal_131k_independent_closeout_passed"
        or int(closeout.get("new_train_accepted_count", -1)) != TRAIN_INCREMENT_COUNT
        or int(closeout.get("development_tail_accepted_count", -1)) != TAIL_COUNT
        or int(closeout.get("logical_train_accepted_count", -1)) != TRAIN_131K_COUNT
        or closeout.get("combined_manifest_sha256")
        != publication.get("combined_manifest_sha256")
        or closeout.get("train_parent_manifest_sha256")
        != publication.get("train_parent_manifest_sha256")
        or closeout.get("development_tail_manifest_sha256")
        != publication.get("development_tail_manifest_sha256")
        or not isinstance(closeout.get("tree_evidence"), Mapping)
        or closeout["tree_evidence"].get("recomputed") is not True
    ):
        raise TrainingGateError("terminal independent closeout does not match packet")

    config = load_yaml(root / TERMINAL_CONFIG_PATH)
    reference = config["corrected_65k_reference"]
    paths = config["paths"]
    frozen = packet.get("terminal_preregistration", {})
    if not isinstance(frozen, Mapping):
        raise TrainingGateError("terminal packet preregistration identity is malformed")
    final_commitment_hash = str(packet.get("final_evaluation_commitment_sha256", ""))
    authorization_flags = {
        "corrected_65k_data_access_authorized": True,
        "terminal_train_increment_data_access_authorized": True,
        "development_tail_data_access_authorized": True,
        "scientific_131k_probe_training_authorized": True,
        "probe_optimizer_execution_authorized": True,
        "terminal_learning_curve_decision_authorized": True,
        **{str(key): False for key in review["closed_boundaries"]},
    }
    train_parent = Path(str(paths["train_publication_root"])) / str(
        closeout["parent_run_id"]
    )
    tail_parent = Path(str(paths["tail_publication_root"])) / str(
        closeout["development_tail_parent_id"]
    )
    combined_root = Path(str(paths["combined_publication_root"])) / str(
        closeout["combined_train_id"]
    )
    authorization = {
        "phase": "4-terminal-probe-131k",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "delegated_review_path": str(review_path),
            "delegated_review_sha256": _sha256(review_path),
            "release_packet_status": TERMINAL_RELEASE_REVIEW_STATUS,
            "independent_closeout_status": closeout["status"],
        },
        "terminal_probe_release_review": {
            "path": str(packet_path),
            "sha256": packet_hash,
            "delegated_review_status": TERMINAL_RELEASE_REVIEW_ACCEPTANCE,
        },
        "frozen_preregistration": {
            "version": "1.2.0-rc.1",
            "path": TERMINAL_PREREGISTRATION_PATH,
            "canonical_hash": frozen.get("canonical_hash"),
        },
        "authorization": authorization_flags,
        "authorized_training_rungs": [TRAIN_131K_COUNT],
        "authorized_training_seeds": list(SEEDS),
        "maximum_concurrent_fits": 3,
        "data_loader_worker_processes": 4,
        "corrected_65k_publication": {
            "base_generator_commit": reference["base_generator_commit"],
            "base_preregistration_hash": reference["base_preregistration_hash"],
            "correction_generator_commit": reference["correction_generator_commit"],
            "correction_preregistration_hash": reference[
                "correction_preregistration_hash"
            ],
            "correction_parent_manifest_sha256": reference[
                "correction_parent_manifest_sha256"
            ],
            "correction_publication_tree_sha256": reference[
                "correction_publication_tree_sha256"
            ],
            "combined_base_manifest_sha256": reference[
                "combined_base_manifest_sha256"
            ],
        },
        "terminal_publication": {
            "combined_manifest_sha256": publication["combined_manifest_sha256"],
            "train_parent_manifest_sha256": publication[
                "train_parent_manifest_sha256"
            ],
            "development_tail_manifest_sha256": publication[
                "development_tail_manifest_sha256"
            ],
        },
        "publication_roots": {
            **dict(reference["publication_roots"]),
            "terminal_train_increment": str(train_parent),
            "terminal_combined_131k": str(combined_root),
            "development_tail": str(tail_parent),
        },
        "immutable_training": dict(immutable),
        "final_evaluation_commitment_sha256": final_commitment_hash,
        "retained_65k_output_root": str(retained_root),
        "training_output_root": str(training_root),
        "learning_curve_output_path": str(
            training_root / "learning_curve_65k_to_131k_decision.json"
        ),
        "execution_contract": {
            "training_rung": TRAIN_131K_COUNT,
            "train_from_scratch": True,
            "seeds": list(SEEDS),
            "retained_65k_checkpoint_retraining_authorized": False,
            "same_core_validation_system_count": 6144,
            "development_tail_system_count": TAIL_COUNT,
            "calibration_during_scale_selection_forbidden": True,
            "final_evaluation_access_forbidden": True,
            "best_seed_selection_forbidden": True,
            "paired_bootstrap_replicates": 10000,
            "frozen_effective_batch_size": 256,
            "physical_microbatch_size": 64,
            "gradient_accumulation_steps": 4,
        },
        "resource_gates": {
            "observed_post_publication_free_bytes": int(
                closeout["observed_remaining_free_bytes"]
            ),
            "minimum_prelaunch_free_bytes": 100_000_000_000,
            "maximum_training_output_bytes": 50_000_000_000,
            "maximum_concurrent_fits": 3,
        },
        "post_freeze_allowed_paths": [
            str(relative_output),
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE4_TERMINAL_131K_PROBE_AUTHORIZATION_REPORT.md",
            "docs/reports/PHASE4_TERMINAL_131K_PROBE_RESULT_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase4/terminal_probe_delegated_review.json",
            "results/phase4/terminal_probe_closeout.json",
            "results/phase4/terminal_probe_release_packet.json",
            "results/phase4/terminal_probe_decision.json",
        ],
        "terminal_decision_states": [
            "lock_train_131k_saturated",
            "lock_train_131k_resource_capped_data_limited",
        ],
        "stop_after_terminal_learning_curve_decision": True,
    }
    validate_terminal_probe_release_binding(authorization)
    return authorization
