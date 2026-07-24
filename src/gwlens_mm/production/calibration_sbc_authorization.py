"""Exact release and authorization control for calibration/SBC materialization."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..provenance import configuration_hash
from ..training.contracts import TrainingGateError
from ..training.terminal_downstream import validate_hashed_terminal_decisions
from .calibration_sbc import (
    BASE_GENERATOR_COMMIT,
    CALIBRATION_SBC_HASH,
    COMBINED_BASE_MANIFEST_HASH,
    CONFIG_PATH,
    CORRECTED_COMBINED_TRAIN_MANIFEST_HASH,
    CORRECTION_GENERATOR_COMMIT,
    CORRECTION_PARENT_MANIFEST_HASH,
    CORRECTION_PUBLICATION_TREE_HASH,
    NUMERICAL_VALIDITY_COMMITMENT_HASH,
    RC4_HASH,
    STAGE_A_PARENT_MANIFEST_HASH,
    STAGE_B_PARENT_MANIFEST_HASH,
    TERMINAL_PREREGISTRATION_HASH,
    derive_calibration_sbc_identities,
    load_calibration_sbc_contract,
)
from .waveform_correction import CORRECTION_PREREGISTRATION_HASH

RELEASE_STATUS = "ready_for_delegated_calibration_sbc_materialization_review"
REVIEW_STATUS = "delegated_calibration_sbc_materialization_review_approved"
AUTHORIZATION_STATUS = "authorized_exact_calibration_sbc_materialization_only"
TERMINAL_PROBE_AUTHORIZATION_STATUS = "authorized_terminal_131k_probe_only"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
MATERIALIZATION_FLAGS = (
    "calibration_sbc_materialization_authorized",
    "accepted_pair_generator_authorized_within_stage_c_only",
    "calibration_fit_statistics_authorized",
    "sbc_statistics_authorized",
    "checkpoint_access_authorized",
    "final_evaluation_authorized",
    "model_retraining_or_tuning_authorized",
    "gwosc_gwtc_access_authorized",
)
CLOSED_FLAGS = MATERIALIZATION_FLAGS[2:]
REVIEW_SCOPE_FIELDS = (
    "training_reference_mode",
    "locked_training_rung",
    "selected_architecture_id",
    "three_model_seeds_retained",
    "materialization_execution_authorized",
    "calibration_fit_accepted_count",
    "sbc_diagnostic_accepted_count",
    "total_accepted_count",
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


def _repo_relative(root: Path, path: Path, prefix: tuple[str, ...]) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise TrainingGateError("release evidence escaped repository") from error
    if relative.parts[: len(prefix)] != prefix:
        raise TrainingGateError("release evidence repository path is invalid")
    return str(relative)


def _hex_digest(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a SHA-256 digest")
    return result


def _full_commit(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a full Git commit")
    return result


def _project_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise TrainingGateError(f"{name} is not an AutoDL project artifact")
    return resolved


def _verify_release_checkout(root: Path, implementation_commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != implementation_commit:
            raise TrainingGateError("calibration/SBC synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("calibration/SBC checkout identity is absent")
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
    if head != implementation_commit or dirty:
        raise TrainingGateError("calibration/SBC release checkout is not exact and clean")


def _terminal_references(
    terminal_authorization: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]]:
    if terminal_authorization.get("authorization_status") != (
        TERMINAL_PROBE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("terminal probe authorization identity changed")
    corrected = terminal_authorization.get("corrected_65k_publication", {})
    terminal = terminal_authorization.get("terminal_publication", {})
    roots = terminal_authorization.get("publication_roots", {})
    if not isinstance(corrected, Mapping) or not isinstance(terminal, Mapping):
        raise TrainingGateError("terminal publication evidence is malformed")
    if not isinstance(roots, Mapping):
        raise TrainingGateError("terminal publication roots are malformed")
    expected_corrected = {
        "base_generator_commit": BASE_GENERATOR_COMMIT,
        "base_preregistration_hash": RC4_HASH,
        "correction_generator_commit": CORRECTION_GENERATOR_COMMIT,
        "correction_preregistration_hash": CORRECTION_PREREGISTRATION_HASH,
        "correction_parent_manifest_sha256": CORRECTION_PARENT_MANIFEST_HASH,
        "correction_publication_tree_sha256": CORRECTION_PUBLICATION_TREE_HASH,
        "combined_base_manifest_sha256": COMBINED_BASE_MANIFEST_HASH,
    }
    if dict(corrected) != expected_corrected:
        raise TrainingGateError("corrected 65k reference changed")
    for name in (
        "combined_manifest_sha256",
        "train_parent_manifest_sha256",
        "development_tail_manifest_sha256",
    ):
        _hex_digest(terminal.get(name), name=f"terminal {name}")
    expected_roles = {
        "stage_a",
        "stage_b",
        "combined_base",
        "correction",
        "terminal_train_increment",
        "terminal_combined_131k",
        "development_tail",
    }
    if set(roots) != expected_roles:
        raise TrainingGateError("terminal publication root set changed")
    normalized = {}
    for role, value in roots.items():
        path = Path(str(value))
        if not path.is_absolute() or not path.is_relative_to(PROJECT_ROOT):
            raise TrainingGateError(f"{role} publication escaped AutoDL project root")
        normalized[str(role)] = str(path.resolve())
    return corrected, terminal, normalized


def _wheel_evidence(
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    *,
    implementation_commit: str,
) -> Mapping[str, Any]:
    wheel_path = _project_file(wheel_path, name="generator wheel")
    exact_wheel_test_result_path = _project_file(
        exact_wheel_test_result_path, name="exact-wheel result"
    )
    environment_lock_path = _project_file(
        environment_lock_path, name="environment lock"
    )
    if not (
        wheel_path.is_file()
        and exact_wheel_test_result_path.is_file()
        and environment_lock_path.is_file()
    ):
        raise TrainingGateError("calibration/SBC immutable artifact is absent")
    wheel_hash = _sha256(wheel_path)
    result = _load_json(exact_wheel_test_result_path)
    if (
        result.get("status") != "passed_exact_wheel_on_autodl"
        or result.get("wheel_sha256") != wheel_hash
        or Path(str(result.get("wheel_path", ""))).resolve()
        != wheel_path.resolve()
        or result.get("focused_test_exit_code") != 0
        or result.get("full_test_exit_code") != 0
        or result.get("editable_install_used") is not False
        or result.get("installed_module_from_repository_source") is not False
        or result.get("scientific_data_opened") is not False
        or result.get("optimizer_started") is not False
    ):
        raise TrainingGateError("calibration/SBC exact-wheel evidence did not pass")
    return {
        "git_commit": implementation_commit,
        "wheel_path": str(wheel_path.resolve()),
        "wheel_sha256": wheel_hash,
        "exact_wheel_test_result_path": str(
            exact_wheel_test_result_path.resolve()
        ),
        "exact_wheel_test_result_sha256": _sha256(
            exact_wheel_test_result_path
        ),
        "environment_lock_path": str(environment_lock_path.resolve()),
        "environment_lock_sha256": _sha256(environment_lock_path),
        "editable_install_authorized": False,
    }


def build_calibration_sbc_materialization_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    terminal_probe_authorization_path: Path,
    terminal_decision_path: Path,
    architecture_decision_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Build a non-authorizing packet after terminal size and architecture lock."""

    root = root.resolve()
    implementation_commit = _full_commit(
        implementation_commit, name="calibration/SBC implementation commit"
    )
    _verify_release_checkout(root, implementation_commit)
    config, _ = load_calibration_sbc_contract(root)
    config_hash = configuration_hash(config)
    terminal_authorization = load_yaml(terminal_probe_authorization_path)
    corrected, terminal, roots = _terminal_references(terminal_authorization)
    terminal_decision_path = _project_file(
        terminal_decision_path, name="terminal size decision"
    )
    architecture_decision_path = _project_file(
        architecture_decision_path, name="architecture decision"
    )
    terminal_hash = _sha256(terminal_decision_path)
    architecture_hash = _sha256(architecture_decision_path)
    decisions = validate_hashed_terminal_decisions(
        root,
        terminal_decision_path=terminal_decision_path,
        terminal_decision_sha256=terminal_hash,
        architecture_decision_path=architecture_decision_path,
        architecture_decision_sha256=architecture_hash,
    )
    validation_hash = _hex_digest(
        terminal_authorization["retained_65k_probe"]["shared_identity"][
            "validation_manifest_sha256"
        ],
        name="validation manifest",
    )
    immutable = _wheel_evidence(
        wheel_path.resolve(),
        exact_wheel_test_result_path.resolve(),
        environment_lock_path.resolve(),
        implementation_commit=implementation_commit,
    )
    identities = derive_calibration_sbc_identities(config, implementation_commit)
    output_relative = _repo_relative(root, output_path, ("results", "phase6"))
    authorization_relative = (
        "configs/execution/phase6_calibration_sbc_materialization_authorization.yaml"
    )
    review_relative = "results/phase6/calibration_sbc_materialization_review.json"
    architecture = decisions["architecture"]
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "materialization_execution_authorized": False,
        "implementation_commit": implementation_commit,
        "frozen_contract": {
            "configuration_path": CONFIG_PATH,
            "configuration_hash": config_hash,
            "calibration_sbc_preregistration_hash": CALIBRATION_SBC_HASH,
            "waveform_numerical_validity_preregistration_hash": (
                CORRECTION_PREREGISTRATION_HASH
            ),
            "waveform_numerical_validity_commitment_sha256": (
                NUMERICAL_VALIDITY_COMMITMENT_HASH
            ),
        },
        "terminal_probe_authorization": {
            "path": _repo_relative(
                root,
                terminal_probe_authorization_path,
                ("configs", "execution"),
            ),
            "sha256": _sha256(terminal_probe_authorization_path),
        },
        "training_size_decision": {
            "path": str(terminal_decision_path.resolve()),
            "sha256": terminal_hash,
            "decision": decisions["terminal"]["decision"],
            "selected_training_count": 131072,
        },
        "architecture_decision": {
            "path": str(architecture_decision_path.resolve()),
            "sha256": architecture_hash,
            "selected_architecture_id": architecture["architecture_id"],
            "model_configuration_hash": architecture[
                "model_configuration_hash"
            ],
            "locked_training_rung": 131072,
            "result_count": 12,
            "three_model_seeds_retained": True,
        },
        "corrected_training_reference": {
            **dict(corrected),
            "stage_a_parent_manifest_sha256": STAGE_A_PARENT_MANIFEST_HASH,
            "stage_b_parent_manifest_sha256": STAGE_B_PARENT_MANIFEST_HASH,
            "corrected_combined_train_manifest_sha256": (
                CORRECTED_COMBINED_TRAIN_MANIFEST_HASH
            ),
            "excluded_base_system_count": 5,
            "replacement_system_count": 5,
            "logical_train_system_count": 65536,
            "unchanged_validation_system_count": 6144,
        },
        "terminal_training_reference": {
            "terminal_preregistration_hash": TERMINAL_PREREGISTRATION_HASH,
            "terminal_combined_manifest_sha256": terminal[
                "combined_manifest_sha256"
            ],
            "terminal_train_increment_parent_manifest_sha256": terminal[
                "train_parent_manifest_sha256"
            ],
            "development_tail_manifest_sha256": terminal[
                "development_tail_manifest_sha256"
            ],
            "validation_manifest_sha256": validation_hash,
            "logical_train_system_count": 131072,
            "new_train_increment_system_count": 65536,
            "unchanged_validation_system_count": 6144,
            "development_tail_system_count": 512,
            "strict_corrected_65k_subset": True,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
            "development_tail_excluded_from_training": True,
            "extension_above_131072_authorized": False,
        },
        "publication_roots": roots,
        "immutable_generator": immutable,
        "prospective_official_identities": identities.as_dict(),
        "materialization_contract": {
            "calibration_fit_accepted_count": 4096,
            "sbc_diagnostic_accepted_count": 2048,
            "total_accepted_count": 6144,
            "total_shard_count": 48,
        },
        "closed_boundaries": {key: False for key in CLOSED_FLAGS},
        "future_authorization_path": authorization_relative,
        "future_review_path": review_relative,
        "release_packet_repository_path": output_relative,
    }


def _validate_review(
    review: Mapping[str, Any], *, packet: Mapping[str, Any], packet_hash: str
) -> None:
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    architecture = packet["architecture_decision"]
    if (
        review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or not str(review.get("reviewed_by", ""))
        or not isinstance(scope, Mapping)
        or set(scope) != set(REVIEW_SCOPE_FIELDS)
        or scope.get("training_reference_mode") != "terminal_131k"
        or int(scope.get("locked_training_rung", -1)) != 131072
        or scope.get("selected_architecture_id")
        != architecture["selected_architecture_id"]
        or scope.get("three_model_seeds_retained") is not True
        or scope.get("materialization_execution_authorized") is not True
        or (
            scope.get("calibration_fit_accepted_count"),
            scope.get("sbc_diagnostic_accepted_count"),
            scope.get("total_accepted_count"),
        )
        != (4096, 2048, 6144)
        or not isinstance(closed, Mapping)
        or set(closed) != set(CLOSED_FLAGS)
        or any(closed.get(key) is not False for key in CLOSED_FLAGS)
    ):
        raise TrainingGateError("delegated calibration/SBC review changed scope")


def _reference_entries(packet: Mapping[str, Any]) -> list[Mapping[str, str]]:
    roots = packet["publication_roots"]
    terminal = packet["terminal_training_reference"]
    hashes = {
        "stage_a": STAGE_A_PARENT_MANIFEST_HASH,
        "stage_b": STAGE_B_PARENT_MANIFEST_HASH,
        "combined_base": COMBINED_BASE_MANIFEST_HASH,
        "correction": CORRECTION_PARENT_MANIFEST_HASH,
        "terminal_train_increment": terminal[
            "terminal_train_increment_parent_manifest_sha256"
        ],
        "terminal_combined_131k": terminal[
            "terminal_combined_manifest_sha256"
        ],
        "development_tail": terminal["development_tail_manifest_sha256"],
    }
    roles = {
        "stage_a": "stage_a_train_and_validation",
        "stage_b": "stage_b_train_extension",
        "combined_base": "combined_65k_base_reference",
        "correction": "waveform_correction_overlay",
        "terminal_train_increment": "terminal_train_increment",
        "terminal_combined_131k": "terminal_131k_combined_reference",
        "development_tail": "terminal_development_tail",
    }
    return [
        {
            "role": roles[name],
            "root": str(roots[name]),
            "manifest_path": str(Path(str(roots[name])) / "dataset_manifest.json"),
            "manifest_sha256": str(hashes[name]),
        }
        for name in roles
    ]


def build_calibration_sbc_materialization_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one reviewed packet into a materialization-only authorization."""

    root = root.resolve()
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("materialization_execution_authorized") is not False
    ):
        raise TrainingGateError("calibration/SBC packet is not review-ready")
    review = _load_json(delegated_review_path)
    _validate_review(review, packet=packet, packet_hash=packet_hash)
    output_relative = _repo_relative(root, output_path, ("configs", "execution"))
    if output_relative != packet["future_authorization_path"]:
        raise TrainingGateError("calibration/SBC authorization path changed")
    release_relative = _repo_relative(
        root, release_packet_path, ("results", "phase6")
    )
    review_relative = _repo_relative(
        root, delegated_review_path, ("results", "phase6")
    )
    allowed_changes = [
        output_relative,
        release_relative,
        review_relative,
        "AGENTS.md",
        "docs/DECISIONS.md",
        "docs/FAILURES.md",
        "docs/PROJECT_STATE.md",
        "docs/reports/PHASE6_CALIBRATION_SBC_MATERIALIZATION_REPORT.md",
        "results/experiment_registry.csv",
        "results/phase6/calibration_sbc_materialization_result.json",
        "results/phase6/calibration_sbc_release_certificate.json",
    ]
    return {
        "phase": "6-calibration-sbc-materialization",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": release_relative,
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
            "delegated_review_status": REVIEW_STATUS,
        },
        "implementation_commit": packet["implementation_commit"],
        "allowed_release_changes": allowed_changes,
        "training_reference_mode": "terminal_131k",
        "frozen_contract": dict(packet["frozen_contract"]),
        "entry_gate": {
            "training_size_locked": True,
            "architecture_locked": True,
            "three_model_seeds_retained": True,
        },
        "training_size_decision": dict(packet["training_size_decision"]),
        "architecture_decision": {
            "path": packet["architecture_decision"]["path"],
            "sha256": packet["architecture_decision"]["sha256"],
            "decision": packet["architecture_decision"][
                "selected_architecture_id"
            ],
            "selected_architecture_id": packet["architecture_decision"][
                "selected_architecture_id"
            ],
            "model_configuration_hash": packet["architecture_decision"][
                "model_configuration_hash"
            ],
            "locked_training_rung": 131072,
            "result_count": 12,
        },
        "corrected_training_reference": dict(
            packet["corrected_training_reference"]
        ),
        "terminal_training_reference": dict(
            packet["terminal_training_reference"]
        ),
        "published_reference_datasets": _reference_entries(packet),
        "immutable_generator": dict(packet["immutable_generator"]),
        "materialization_contract": dict(packet["materialization_contract"]),
        "prospective_official_identities": dict(
            packet["prospective_official_identities"]
        ),
        "authorization": {
            "calibration_sbc_materialization_authorized": True,
            "accepted_pair_generator_authorized_within_stage_c_only": True,
            **{key: False for key in CLOSED_FLAGS},
        },
        "stop_after_atomic_materialization": True,
    }
