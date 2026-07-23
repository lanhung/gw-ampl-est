"""Build exact post-lock terminal architecture release and authorization evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from .architecture import (
    GRID_PATH,
    NEW_ARCHITECTURE_IDS,
    candidate_model_configuration,
    load_architecture_specs,
)
from .contracts import TrainingGateError, model_configuration_hash
from .rung65 import SEEDS, VALIDATION_COUNT
from .terminal131 import TRAIN_131K_COUNT

PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
RELEASE_STATUS = "ready_for_delegated_terminal_architecture_authorization_review"
REVIEW_STATUS = "delegated_terminal_architecture_authorization_approved"
AUTHORIZATION_STATUS = "authorized_terminal_131k_architecture_selection_only"
TERMINAL_DECISIONS = frozenset(
    {
        "lock_train_131k_saturated",
        "lock_train_131k_resource_capped_data_limited",
    }
)
CLOSED_BOUNDARIES = (
    "probe_architecture_retraining_authorized",
    "best_seed_selection_authorized",
    "model_tuning_authorized",
    "calibration_authorized",
    "sbc_authorized",
    "final_evaluation_authorized",
    "extension_above_131072_authorized",
    "real_noise_authorized",
    "gwosc_gwtc_access_authorized",
)
REVIEW_SCOPE_FIELDS = (
    "locked_training_rung",
    "authorized_new_architecture_ids",
    "authorized_training_seeds",
    "publication_data_access_authorized",
    "new_architecture_fit_execution_authorized",
    "architecture_selection_execution_authorized",
    "probe_output_root",
    "architecture_output_root",
    "architecture_selection_output_path",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected JSON mapping: {path}")
    return value


def _repo_relative(root: Path, path: Path, *, prefix: tuple[str, ...]) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise TrainingGateError("architecture evidence escaped repository") from error
    if relative.parts[: len(prefix)] != prefix:
        raise TrainingGateError("architecture evidence has an invalid repository path")
    return str(relative)


def _project_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise TrainingGateError("architecture output escaped the AutoDL project root")
    return resolved


def _validate_terminal_decision(path: Path) -> Mapping[str, Any]:
    decision = _load_json(path)
    if (
        decision.get("decision") not in TERMINAL_DECISIONS
        or decision.get("comparison")
        != "corrected_train_65k_to_train_131k_terminal"
        or int(decision.get("selected_training_count", -1)) != TRAIN_131K_COUNT
        or decision.get("architecture_selection_review_allowed") is not True
        or decision.get("extension_above_131072_authorized") is not False
    ):
        raise TrainingGateError("terminal decision does not lock the 131k rung")
    return decision


def _probe_artifacts(
    probe_output_root: Path,
    *,
    terminal_authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    summaries: dict[str, str] = {}
    checkpoints: dict[str, str] = {}
    identities = []
    expected_model_hash = terminal_authorization["immutable_training"][
        "model_configuration_hash"
    ]
    expected_train_hash = terminal_authorization["terminal_publication"][
        "combined_manifest_sha256"
    ]
    expected_validation_hash = terminal_authorization["corrected_65k_publication"].get(
        "validation_manifest_sha256"
    )
    # Older terminal authorization stores the validation hash only in retained
    # evidence. The scientific summaries remain the authoritative cross-check.
    for seed in SEEDS:
        directory = probe_output_root / "rung-131072" / f"seed-{seed}"
        summary_path = directory / "run_summary.json"
        checkpoint_path = directory / "best.ckpt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise TrainingGateError("terminal probe artifact is incomplete")
        summary = _load_json(summary_path)
        identity = summary.get("identity", {})
        development = summary.get("development", {})
        if (
            summary.get("status")
            != "completed_131k_probe_fit_and_development_validation"
            or int(identity.get("training_rung_count", -1)) != TRAIN_131K_COUNT
            or int(identity.get("seed", -1)) != seed
            or identity.get("model_configuration_hash") != expected_model_hash
            or identity.get("train_manifest_sha256") != expected_train_hash
            or development.get("status") != "completed_development_validation"
            or int(development.get("case_count", -1)) != VALIDATION_COUNT
            or development.get("posthoc_calibration_applied") is not False
            or development.get("final_evaluation_accessed") is not False
            or summary.get("architecture_selection_authorized") is not False
            or summary.get("final_evaluation_accessed") is not False
            or summary.get("extension_above_131072_authorized") is not False
        ):
            raise TrainingGateError("terminal probe summary violates the frozen contract")
        if expected_validation_hash is not None and (
            identity.get("validation_manifest_sha256") != expected_validation_hash
        ):
            raise TrainingGateError("terminal probe validation identity changed")
        summaries[str(seed)] = _sha256(summary_path)
        checkpoints[str(seed)] = _sha256(checkpoint_path)
        identities.append(identity)
    for key in (
        "training_code_commit",
        "training_environment_sha256",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "final_evaluation_commitment_sha256",
        "membership_sha256",
        "input_standardizer_sha256",
        "target_standardizer_sha256",
        "model_configuration_hash",
        "training_rung_count",
    ):
        if len({identity.get(key) for identity in identities}) != 1:
            raise TrainingGateError(f"terminal probe shared identity changed for {key}")
    preparation_path = probe_output_root / "rung-131072" / "rung_preparation.json"
    if not preparation_path.is_file():
        raise TrainingGateError("terminal rung preparation is absent")
    preparation = _load_json(preparation_path)
    if (
        preparation.get("status") != "ready_for_authorized_probe_fits"
        or int(preparation.get("rung_count", -1)) != TRAIN_131K_COUNT
        or int(preparation.get("member_count", -1)) != TRAIN_131K_COUNT
        or preparation.get("optimizer_started") is not False
    ):
        raise TrainingGateError("terminal rung preparation identity is invalid")
    return {
        "output_root": str(probe_output_root),
        "rung_preparation_sha256": _sha256(preparation_path),
        "run_summary_sha256": summaries,
        "best_checkpoint_sha256": checkpoints,
        "shared_identity": dict(identities[0]),
    }


def build_terminal_architecture_release_packet(
    root: Path,
    *,
    terminal_probe_authorization_path: Path,
    terminal_decision_path: Path,
    probe_output_root: Path,
    training_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    architecture_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Assemble non-authorizing evidence for delegated architecture review."""

    root = root.resolve()
    authorization_relative = _repo_relative(
        root,
        terminal_probe_authorization_path,
        prefix=("configs", "execution"),
    )
    output_relative = _repo_relative(
        root, output_path, prefix=("results", "phase5")
    )
    terminal_authorization = load_yaml(terminal_probe_authorization_path)
    if (
        terminal_authorization.get("authorization_status")
        != "authorized_terminal_131k_probe_only"
        or terminal_authorization.get("authorization", {}).get(
            "architecture_selection_authorized"
        )
        is not False
    ):
        raise TrainingGateError("terminal probe authorization is not a closed precursor")
    expected_probe_root = Path(
        str(terminal_authorization.get("training_output_root", ""))
    ).resolve()
    if probe_output_root.resolve() != expected_probe_root:
        raise TrainingGateError("terminal probe output root changed")
    decision = _validate_terminal_decision(terminal_decision_path)
    expected_decision = Path(
        str(terminal_authorization.get("learning_curve_output_path", ""))
    ).resolve()
    if terminal_decision_path.resolve() != expected_decision:
        raise TrainingGateError("terminal decision path changed")
    if len(training_commit) != 40:
        raise TrainingGateError("architecture training commit must be full length")
    synced_commit_path = root / "SYNCED_COMMIT"
    if (
        not synced_commit_path.is_file()
        or synced_commit_path.read_text(encoding="utf-8").strip() != training_commit
    ):
        raise TrainingGateError("architecture review checkout commit changed")
    wheel = wheel_path.resolve()
    environment = environment_lock_path.resolve()
    wheel_result_path = exact_wheel_test_result_path.resolve()
    if not wheel.is_file() or not environment.is_file() or not wheel_result_path.is_file():
        raise TrainingGateError("architecture immutable artifact is absent")
    wheel_hash = _sha256(wheel)
    environment_hash = _sha256(environment)
    wheel_result = _load_json(wheel_result_path)
    if (
        wheel_result.get("status") != "passed_exact_wheel_on_autodl"
        or wheel_result.get("wheel_sha256") != wheel_hash
        or Path(str(wheel_result.get("wheel_path", ""))).resolve() != wheel
        or wheel_result.get("focused_test_exit_code") != 0
        or wheel_result.get("full_test_exit_code") != 0
        or wheel_result.get("editable_install_used") is not False
        or wheel_result.get("installed_module_from_repository_source") is not False
        or wheel_result.get("torch_cuda_available") is not True
        or wheel_result.get("scientific_data_opened") is not False
        or wheel_result.get("optimizer_started") is not False
        or len(wheel_result.get("gpu_names", ())) < 3
    ):
        raise TrainingGateError("architecture exact-wheel evidence did not pass")
    model = load_yaml(root / "configs/models/phase4_probe_nsf.yaml")
    model_hash = model_configuration_hash(model)
    grid_path = root / GRID_PATH
    grid_hash = _sha256(grid_path)
    candidate_hashes = {
        specification.architecture_id: model_configuration_hash(
            candidate_model_configuration(root, specification)
        )
        for specification in load_architecture_specs(root)
        if not specification.reused_probe
    }
    if set(candidate_hashes) != set(NEW_ARCHITECTURE_IDS):
        raise TrainingGateError("terminal architecture grid changed")
    probe = _probe_artifacts(
        probe_output_root.resolve(), terminal_authorization=terminal_authorization
    )
    if probe["shared_identity"]["model_configuration_hash"] != model_hash:
        raise TrainingGateError("terminal probe model differs from architecture grid")
    output_root = _project_output(architecture_output_root)
    if output_root.exists():
        raise TrainingGateError("terminal architecture output identity already exists")
    packet = {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "architecture_fit_execution_authorized": False,
        "architecture_selection_execution_authorized": False,
        "release_review_checkout_commit": training_commit,
        "terminal_probe_authorization": {
            "path": authorization_relative,
            "sha256": _sha256(terminal_probe_authorization_path),
        },
        "terminal_decision": {
            "path": str(terminal_decision_path.resolve()),
            "sha256": _sha256(terminal_decision_path),
            "decision": decision["decision"],
            "selected_training_count": TRAIN_131K_COUNT,
        },
        "terminal_probe": probe,
        "frozen_preregistration": dict(
            terminal_authorization["frozen_preregistration"]
        ),
        "corrected_65k_publication": dict(
            terminal_authorization["corrected_65k_publication"]
        ),
        "terminal_publication": dict(
            terminal_authorization["terminal_publication"]
        ),
        "publication_roots": dict(terminal_authorization["publication_roots"]),
        "final_evaluation_commitment_sha256": terminal_authorization[
            "final_evaluation_commitment_sha256"
        ],
        "architecture_grid": {
            "path": GRID_PATH,
            "sha256": grid_hash,
            "reused_probe_model_configuration_hash": model_hash,
            "new_architecture_ids": list(NEW_ARCHITECTURE_IDS),
            "candidate_model_hashes": candidate_hashes,
            "seeds": list(SEEDS),
        },
        "immutable_training": {
            "git_commit": training_commit,
            "wheel_path": str(wheel),
            "wheel_filename": wheel.name,
            "wheel_sha256": wheel_hash,
            "environment_lock_path": str(environment),
            "environment_lock_sha256": environment_hash,
            "exact_wheel_test_result_path": str(wheel_result_path),
            "exact_wheel_test_result_sha256": _sha256(wheel_result_path),
            "model_configuration_path": "configs/models/phase4_probe_nsf.yaml",
            "model_configuration_hash": model_hash,
            "gpu_model": "NVIDIA RTX 5000 Ada Generation",
            "observed_gpu_names": list(wheel_result["gpu_names"]),
            "cuda_required": True,
            "editable_install_authorized": False,
        },
        "architecture_output_root": str(output_root),
        "architecture_selection_output_path": str(
            output_root / "architecture_selection.json"
        ),
        "resource_gates": {
            "maximum_concurrent_fits": 3,
            "maximum_new_fits": 9,
            "maximum_training_output_bytes": 50_000_000_000,
        },
        "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        "release_packet_repository_path": output_relative,
    }
    return packet


def _validate_review(
    review: Mapping[str, Any], *, packet_sha256: str, packet: Mapping[str, Any]
) -> None:
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if not isinstance(scope, Mapping) or not isinstance(closed, Mapping):
        raise TrainingGateError("terminal architecture review lacks explicit scope")
    if (
        review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_sha256
        or not str(review.get("reviewed_by", ""))
        or set(scope) != set(REVIEW_SCOPE_FIELDS)
        or set(closed) != set(CLOSED_BOUNDARIES)
        or int(scope.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or scope.get("authorized_new_architecture_ids")
        != list(NEW_ARCHITECTURE_IDS)
        or scope.get("authorized_training_seeds") != list(SEEDS)
        or scope.get("publication_data_access_authorized") is not True
        or scope.get("new_architecture_fit_execution_authorized") is not True
        or scope.get("architecture_selection_execution_authorized") is not True
        or Path(str(scope.get("probe_output_root", ""))).resolve()
        != Path(str(packet["terminal_probe"]["output_root"])).resolve()
        or Path(str(scope.get("architecture_output_root", ""))).resolve()
        != Path(str(packet["architecture_output_root"])).resolve()
        or Path(str(scope.get("architecture_selection_output_path", ""))).resolve()
        != Path(str(packet["architecture_selection_output_path"])).resolve()
    ):
        raise TrainingGateError("terminal architecture review does not approve exact scope")
    if any(closed.get(key) is not False for key in CLOSED_BOUNDARIES):
        raise TrainingGateError("terminal architecture review opened a downstream gate")


def build_terminal_architecture_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one reviewed packet into the exact runtime authorization."""

    root = root.resolve()
    packet_relative = _repo_relative(
        root, release_packet_path, prefix=("results", "phase5")
    )
    review_relative = _repo_relative(
        root, delegated_review_path, prefix=("results", "phase5")
    )
    output_relative = _repo_relative(
        root, output_path, prefix=("configs", "execution")
    )
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("architecture_fit_execution_authorized") is not False
        or packet.get("architecture_selection_execution_authorized") is not False
    ):
        raise TrainingGateError("terminal architecture release packet is not review-ready")
    review = _load_json(delegated_review_path)
    _validate_review(review, packet_sha256=packet_hash, packet=packet)
    grid = packet["architecture_grid"]
    probe = packet["terminal_probe"]
    decision = packet["terminal_decision"]
    authorization = {
        "phase": "5-terminal-architecture-131k",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": packet_relative,
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
            "delegated_review_status": REVIEW_STATUS,
        },
        "frozen_preregistration": dict(packet["frozen_preregistration"]),
        "authorization": {
            "corrected_65k_data_access_authorized": True,
            "terminal_train_increment_data_access_authorized": True,
            "new_architecture_fit_execution_authorized": True,
            "architecture_selection_execution_authorized": True,
            **{key: False for key in CLOSED_BOUNDARIES},
        },
        "locked_training_rung": TRAIN_131K_COUNT,
        "authorized_new_architecture_ids": list(NEW_ARCHITECTURE_IDS),
        "authorized_training_seeds": list(SEEDS),
        "maximum_concurrent_fits": 3,
        "data_loader_worker_processes": 4,
        "terminal_decision_path": decision["path"],
        "terminal_decision_sha256": decision["sha256"],
        "terminal_decision": decision["decision"],
        "corrected_65k_publication": dict(packet["corrected_65k_publication"]),
        "terminal_publication": dict(packet["terminal_publication"]),
        "publication_roots": dict(packet["publication_roots"]),
        "architecture_grid_sha256": grid["sha256"],
        "reused_probe_output_root": probe["output_root"],
        "reused_probe_run_summary_sha256": dict(probe["run_summary_sha256"]),
        "reused_probe_best_checkpoint_sha256": dict(
            probe["best_checkpoint_sha256"]
        ),
        "reused_probe_rung_preparation_sha256": probe[
            "rung_preparation_sha256"
        ],
        "reused_probe_model_configuration_hash": grid[
            "reused_probe_model_configuration_hash"
        ],
        "reused_probe_training_environment_sha256": probe["shared_identity"][
            "training_environment_sha256"
        ],
        "final_evaluation_commitment_sha256": packet[
            "final_evaluation_commitment_sha256"
        ],
        "candidate_model_hashes": dict(grid["candidate_model_hashes"]),
        "immutable_training": dict(packet["immutable_training"]),
        "architecture_output_root": packet["architecture_output_root"],
        "architecture_selection_output_path": packet[
            "architecture_selection_output_path"
        ],
        "resource_gates": dict(packet["resource_gates"]),
        "post_freeze_allowed_paths": [
            output_relative,
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE5_TERMINAL_ARCHITECTURE_SELECTION_REPORT.md",
            "results/experiment_registry.csv",
            packet_relative,
            review_relative,
            "results/phase5/terminal_architecture_selection.json",
        ],
        "stop_after_architecture_selection": True,
    }
    return authorization
