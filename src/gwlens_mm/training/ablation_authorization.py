"""Exact release control for the six preregistered input-ablation fits.

The release packet is deliberately non-authorizing.  It binds a completed
terminal size/architecture lock, the primary 131k preprocessing identity,
immutable training software and six fresh output identities.  Only a separate
delegated review can create the runtime authorization consumed by
``training.ablations``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..production.calibration_sbc import CORRECTED_COMBINED_TRAIN_MANIFEST_HASH
from .ablations import (
    ABLATION_VIEWS,
    FINAL_ANALYSIS_HASH,
    MAXIMUM_FITS,
    ablation_model_configuration,
    validate_ablation_stack_contract,
)
from .architecture import selected_model_configuration
from .contracts import TrainingGateError, model_configuration_hash
from .rung65 import SEEDS
from .terminal131 import TRAIN_131K_COUNT
from .terminal_architecture_authorization import (
    AUTHORIZATION_STATUS as ARCHITECTURE_AUTHORIZATION_STATUS,
)
from .terminal_downstream import validate_hashed_terminal_decisions

STACK_AUTHORIZATION = (
    "configs/execution/phase7_ablation_release_stack_authorization.yaml"
)
RELEASE_STATUS = "ready_for_delegated_terminal_ablation_training_review"
REVIEW_STATUS = "delegated_terminal_ablation_training_review_approved"
AUTHORIZATION_STATUS = "authorized_terminal_131k_ablation_training_only"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")

REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "ablation_views",
    "model_seeds",
    "fit_count",
    "maximum_concurrent_fits",
    "data_loader_worker_processes",
    "corrected_65k_data_access_authorized",
    "terminal_train_increment_data_access_authorized",
    "terminal_131k_combined_reference_access_authorized",
    "development_validation_authorized",
    "ablation_fit_execution_authorized",
)
CLOSED_BOUNDARIES = (
    "primary_model_retraining_authorized",
    "architecture_or_size_selection_authorized",
    "additional_architecture_tuning_authorized",
    "calibration_or_sbc_access_authorized",
    "final_evaluation_materialization_authorized",
    "final_evaluation_unsealing_authorized",
    "final_evaluation_inference_authorized",
    "extension_above_131072_authorized",
    "gwosc_gwtc_access_authorized",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected JSON mapping: {path}")
    return value


def _repo_relative(
    root: Path,
    path: Path,
    *,
    prefix: tuple[str, ...],
    name: str,
) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise TrainingGateError(f"{name} escaped the repository") from error
    if relative.parts[: len(prefix)] != prefix:
        raise TrainingGateError(f"{name} has an invalid repository path")
    return str(relative)


def _project_path(path: Path, *, name: str, require_file: bool = False) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise TrainingGateError(f"{name} escaped the AutoDL project root")
    if require_file and not resolved.is_file():
        raise TrainingGateError(f"{name} is absent")
    return resolved


def _full_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise TrainingGateError("ablation implementation commit is not full length")
    return commit


def _verify_checkout(root: Path, commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit:
            raise TrainingGateError("ablation synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("ablation checkout identity is absent")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit or dirty:
        raise TrainingGateError("ablation checkout is not exact and clean")


def load_ablation_release_stack_contract(root: Path) -> Mapping[str, Any]:
    """Validate that ablation release work remains synthetic-only."""

    validate_ablation_stack_contract(root)
    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise TrainingGateError("ablation release-stack gate is absent")
    frozen = authorization.get("frozen_contracts", {})
    if (
        frozen.get("final_analysis_hash") != FINAL_ANALYSIS_HASH
        or int(frozen.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or frozen.get("ablation_views") != list(ABLATION_VIEWS)
        or frozen.get("model_seeds") != list(SEEDS)
        or int(frozen.get("exact_fit_count", -1)) != MAXIMUM_FITS
        or int(frozen.get("maximum_concurrent_fits", -1)) != 3
        or int(frozen.get("data_loader_worker_processes", -1)) != 4
        or frozen.get("identical_optimizer_and_budget_required") is not True
        or frozen.get("final_evaluation_access_forbidden") is not True
    ):
        raise TrainingGateError("ablation release contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise TrainingGateError("ablation release implementation opened execution")
    return authorization


def _immutable_training(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="ablation wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="ablation exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path,
        name="ablation environment lock",
        require_file=True,
    )
    wheel_hash = _sha256(wheel)
    result = _load_json(result_path)
    if (
        result.get("status") != "passed_exact_wheel_on_autodl"
        or result.get("wheel_sha256") != wheel_hash
        or Path(str(result.get("wheel_path", ""))).resolve() != wheel
        or result.get("focused_test_exit_code") != 0
        or result.get("full_test_exit_code") != 0
        or result.get("editable_install_used") is not False
        or result.get("installed_module_from_repository_source") is not False
        or result.get("torch_cuda_available") is not True
        or result.get("scientific_data_opened") is not False
        or result.get("optimizer_started") is not False
        or len(result.get("gpu_names", ())) < 3
    ):
        raise TrainingGateError("ablation exact-wheel evidence did not pass")
    return {
        "git_commit": implementation_commit,
        "wheel_path": str(wheel),
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_hash,
        "exact_wheel_test_result_path": str(result_path),
        "exact_wheel_test_result_sha256": _sha256(result_path),
        "environment_lock_path": str(environment),
        "environment_lock_sha256": _sha256(environment),
        "gpu_model": "NVIDIA RTX 5000 Ada Generation",
        "observed_gpu_names": list(result["gpu_names"]),
        "cuda_required": True,
        "editable_install_authorized": False,
    }


def build_ablation_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    architecture_authorization_path: Path,
    architecture_decision_path: Path,
    terminal_decision_path: Path,
    primary_rung_preparation_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    ablation_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Assemble a non-authorizing packet for exactly six ablation fits."""

    root = root.resolve()
    load_ablation_release_stack_contract(root)
    implementation_commit = _full_commit(implementation_commit)
    _verify_checkout(root, implementation_commit)
    architecture_authorization = load_yaml(architecture_authorization_path)
    if architecture_authorization.get("authorization_status") != (
        ARCHITECTURE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("terminal architecture authorization is absent")
    if (
        architecture_decision_path.resolve()
        != Path(
            str(
                architecture_authorization.get(
                    "architecture_selection_output_path", ""
                )
            )
        ).resolve()
        or terminal_decision_path.resolve()
        != Path(
            str(architecture_authorization.get("terminal_decision_path", ""))
        ).resolve()
    ):
        raise TrainingGateError("ablation decision paths changed")
    decisions = validate_hashed_terminal_decisions(
        root,
        terminal_decision_path=terminal_decision_path,
        terminal_decision_sha256=_sha256(terminal_decision_path),
        architecture_decision_path=architecture_decision_path,
        architecture_decision_sha256=_sha256(architecture_decision_path),
    )
    if (
        _sha256(terminal_decision_path)
        != architecture_authorization.get("terminal_decision_sha256")
    ):
        raise TrainingGateError("ablation terminal decision differs from architecture")
    architecture = decisions["architecture"]
    architecture_id = str(architecture["architecture_id"])
    primary_model = selected_model_configuration(root, architecture_id)
    primary_hash = model_configuration_hash(primary_model)
    if primary_hash != architecture["model_configuration_hash"]:
        raise TrainingGateError("ablation selected primary model changed")
    ablation_hashes = {
        view: model_configuration_hash(
            ablation_model_configuration(
                root,
                architecture_id=architecture_id,
                view=view,
            )
        )
        for view in ABLATION_VIEWS
    }
    expected_preparation = (
        Path(str(architecture_authorization["reused_probe_output_root"]))
        / "rung-131072"
        / "rung_preparation.json"
    ).resolve()
    preparation_path = _project_path(
        primary_rung_preparation_path,
        name="primary 131k rung preparation",
        require_file=True,
    )
    preparation = _load_json(preparation_path)
    members = tuple(str(value) for value in preparation.get("member_ids", ()))
    if (
        preparation_path != expected_preparation
        or _sha256(preparation_path)
        != architecture_authorization.get("reused_probe_rung_preparation_sha256")
        or preparation.get("status") != "ready_for_authorized_probe_fits"
        or int(preparation.get("rung_count", -1)) != TRAIN_131K_COUNT
        or len(members) != TRAIN_131K_COUNT
        or len(set(members)) != TRAIN_131K_COUNT
        or preparation.get("final_evaluation_commitment_sha256")
        != architecture_authorization.get("final_evaluation_commitment_sha256")
    ):
        raise TrainingGateError("ablation primary rung preparation changed")
    output_root = _project_path(
        ablation_output_root,
        name="ablation output root",
    )
    if output_root.exists():
        raise TrainingGateError("ablation output identity already exists")
    immutable = _immutable_training(
        implementation_commit=implementation_commit,
        wheel_path=wheel_path,
        exact_wheel_test_result_path=exact_wheel_test_result_path,
        environment_lock_path=environment_lock_path,
    )
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("results", "phase7"),
        name="ablation release packet",
    )
    if output_relative != "results/phase7/ablation_release_packet.json":
        raise TrainingGateError("ablation release-packet path changed")
    review_scope = {
        "locked_training_rung": TRAIN_131K_COUNT,
        "selected_architecture_id": architecture_id,
        "ablation_views": list(ABLATION_VIEWS),
        "model_seeds": list(SEEDS),
        "fit_count": MAXIMUM_FITS,
        "maximum_concurrent_fits": 3,
        "data_loader_worker_processes": 4,
        "corrected_65k_data_access_authorized": True,
        "terminal_train_increment_data_access_authorized": True,
        "terminal_131k_combined_reference_access_authorized": True,
        "development_validation_authorized": True,
        "ablation_fit_execution_authorized": True,
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "ablation_fit_execution_authorized": False,
        "implementation_commit": implementation_commit,
        "architecture_authorization": {
            "path": _repo_relative(
                root,
                architecture_authorization_path,
                prefix=("configs", "execution"),
                name="architecture authorization",
            ),
            "sha256": _sha256(architecture_authorization_path),
        },
        "terminal_decision": {
            "path": str(terminal_decision_path.resolve()),
            "sha256": _sha256(terminal_decision_path),
            "decision": decisions["terminal"]["decision"],
        },
        "selected_architecture": {
            "decision_path": str(architecture_decision_path.resolve()),
            "decision_sha256": _sha256(architecture_decision_path),
            "architecture_id": architecture_id,
            "model_configuration_hash": primary_hash,
            "locked_training_rung": TRAIN_131K_COUNT,
        },
        "ablation_model_configuration_hashes": ablation_hashes,
        "primary_rung_preparation": {
            "path": str(preparation_path),
            "sha256": _sha256(preparation_path),
            "membership_sha256": preparation.get("membership_sha256"),
            "input_standardizer_sha256": preparation.get(
                "input_standardizer_sha256"
            ),
            "target_standardizer_sha256": preparation.get(
                "target_standardizer_sha256"
            ),
        },
        "corrected_65k_publication": dict(
            architecture_authorization["corrected_65k_publication"]
        ),
        "terminal_publication": dict(
            architecture_authorization["terminal_publication"]
        ),
        "publication_roots": dict(
            architecture_authorization["publication_roots"]
        ),
        "final_evaluation_commitment_sha256": architecture_authorization[
            "final_evaluation_commitment_sha256"
        ],
        "immutable_training": immutable,
        "ablation_output_root": str(output_root),
        "fit_output_identities": {
            view: {
                str(seed): str(output_root / view / f"seed-{seed}")
                for seed in SEEDS
            }
            for view in ABLATION_VIEWS
        },
        "review_scope": review_scope,
        "closed_boundaries": {
            key: False for key in CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/phase7_terminal_ablation_training_authorization.yaml"
        ),
        "future_review_path": "results/phase7/ablation_training_review.json",
        "release_packet_repository_path": output_relative,
    }


def build_ablation_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact six-fit packet after delegated review."""

    root = root.resolve()
    load_ablation_release_stack_contract(root)
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("ablation_fit_execution_authorized") is not False
        or review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by")
        != "codex_as_delegated_scientific_and_engineering_reviewer"
        or set(scope) != set(REVIEW_SCOPE)
        or any(scope.get(key) != packet["review_scope"][key] for key in REVIEW_SCOPE)
        or set(closed) != set(CLOSED_BOUNDARIES)
        or any(closed.get(key) is not False for key in CLOSED_BOUNDARIES)
    ):
        raise TrainingGateError("delegated ablation review is not exact")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="ablation release packet",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="ablation delegated review",
    )
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="ablation authorization",
    )
    if (
        release_relative != packet.get("release_packet_repository_path")
        or review_relative != packet.get("future_review_path")
        or output_relative != packet.get("future_authorization_path")
    ):
        raise TrainingGateError("ablation release evidence path changed")
    corrected = packet["corrected_65k_publication"]
    preparation = packet["primary_rung_preparation"]
    return {
        "phase": "7-terminal-131k-input-ablations",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": release_relative,
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "authorization": {
            "corrected_65k_data_access_authorized": True,
            "terminal_train_increment_data_access_authorized": True,
            "terminal_131k_combined_reference_access_authorized": True,
            "ablation_fit_execution_authorized": True,
            "development_validation_authorized": True,
            **{key: False for key in CLOSED_BOUNDARIES},
        },
        "locked_training_rung": TRAIN_131K_COUNT,
        "authorized_ablation_views": list(ABLATION_VIEWS),
        "authorized_training_seeds": list(SEEDS),
        "maximum_fit_count": MAXIMUM_FITS,
        "maximum_concurrent_fits": 3,
        "data_loader_worker_processes": 4,
        "terminal_size_decision_path": packet["terminal_decision"]["path"],
        "terminal_size_decision_sha256": packet["terminal_decision"]["sha256"],
        "selected_architecture_decision_path": packet["selected_architecture"][
            "decision_path"
        ],
        "selected_architecture_decision_sha256": packet[
            "selected_architecture"
        ]["decision_sha256"],
        "selected_primary_model_configuration_hash": packet[
            "selected_architecture"
        ]["model_configuration_hash"],
        "ablation_model_configuration_hashes": dict(
            packet["ablation_model_configuration_hashes"]
        ),
        "primary_rung_preparation_path": preparation["path"],
        "primary_rung_preparation_sha256": preparation["sha256"],
        "final_evaluation_commitment_sha256": packet[
            "final_evaluation_commitment_sha256"
        ],
        "base_generator_commit": corrected["base_generator_commit"],
        "base_preregistration_hash": corrected["base_preregistration_hash"],
        "combined_base_manifest_sha256": corrected[
            "combined_base_manifest_sha256"
        ],
        "corrected_combined_train_manifest_sha256": (
            CORRECTED_COMBINED_TRAIN_MANIFEST_HASH
        ),
        "correction_publication": {
            "generator_commit": corrected["correction_generator_commit"],
            "parent_manifest_sha256": corrected[
                "correction_parent_manifest_sha256"
            ],
            "publication_tree_sha256": corrected[
                "correction_publication_tree_sha256"
            ],
        },
        "terminal_publication": dict(packet["terminal_publication"]),
        "publication_roots": dict(packet["publication_roots"]),
        "immutable_training": dict(packet["immutable_training"]),
        "ablation_output_root": packet["ablation_output_root"],
        "fit_output_identities": dict(packet["fit_output_identities"]),
        "post_freeze_allowed_paths": [
            output_relative,
            release_relative,
            review_relative,
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE7_ABLATION_EXECUTION_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase7/ablation_summary.json",
        ],
        "stop_after_six_ablation_fits": True,
    }
