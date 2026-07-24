"""Release control for the read-only legacy SIS descriptive stress control."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from .contracts import TrainingGateError
from .legacy_sis_stress import (
    LEGACY_ROOT,
    NEW_PROJECT_ROOT,
    LegacySISStressContract,
    LegacySISStressGateError,
    legacy_sis_contract,
    validate_legacy_sis_stack_contract,
    verify_legacy_sis_stress_control,
    write_legacy_sis_evidence,
)
from .runner import _verify_training_checkout

RELEASE_STATUS = "ready_for_delegated_legacy_sis_read_only_review"
REVIEW_STATUS = "delegated_legacy_sis_read_only_review_approved"
AUTHORIZATION_STATUS = "authorized_read_only_reproduction"

REVIEW_FIELDS = (
    "legacy_asset_read_authorized",
    "baseline_execution_authorized",
    "legacy_checkpoint_deserialization_authorized",
    "legacy_asset_write_authorized",
    "scientific_data_access_authorized",
    "final_evaluation_access_authorized",
    "v2_final_data_application_authorized",
    "manuscript_claim_finalization_authorized",
    "gwosc_gwtc_access_authorized",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LegacySISStressGateError(f"expected JSON mapping: {path}")
    return value


def _full_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise LegacySISStressGateError(
            "legacy SIS implementation commit is not full length"
        )
    return commit


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
        raise LegacySISStressGateError(f"{name} escaped the repository") from error
    if relative.parts[: len(prefix)] != prefix:
        raise LegacySISStressGateError(f"{name} has an invalid repository path")
    return str(relative)


def _project_path(
    path: Path,
    *,
    name: str,
    require_file: bool = False,
) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(NEW_PROJECT_ROOT):
        raise LegacySISStressGateError(f"{name} escaped the new project root")
    if require_file and not resolved.is_file():
        raise LegacySISStressGateError(f"{name} is absent")
    return resolved


def _verify_checkout(root: Path, commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit:
            raise LegacySISStressGateError("legacy SIS synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise LegacySISStressGateError("legacy SIS checkout identity is absent")
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
        raise LegacySISStressGateError(
            "legacy SIS release checkout is not exact and clean"
        )


def _immutable_execution(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="legacy SIS wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="legacy SIS exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path,
        name="legacy SIS environment lock",
        require_file=True,
    )
    result = _load_json(result_path)
    wheel_hash = _sha256(wheel)
    if (
        result.get("status") != "passed_exact_wheel_on_autodl"
        or result.get("wheel_sha256") != wheel_hash
        or Path(str(result.get("wheel_path", ""))).resolve() != wheel
        or result.get("focused_test_exit_code") != 0
        or result.get("full_test_exit_code") != 0
        or result.get("editable_install_used") is not False
        or result.get("installed_module_from_repository_source") is not False
        or result.get("legacy_asset_read") is not False
        or result.get("scientific_data_opened") is not False
    ):
        raise LegacySISStressGateError("legacy SIS exact-wheel evidence did not pass")
    return {
        "git_commit": implementation_commit,
        "wheel_path": str(wheel),
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_hash,
        "exact_wheel_test_result_path": str(result_path),
        "exact_wheel_test_result_sha256": _sha256(result_path),
        "environment_lock_path": str(environment),
        "environment_lock_sha256": _sha256(environment),
        "editable_install_authorized": False,
    }


def build_legacy_sis_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    evidence_output_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Create a non-authorizing packet without touching a legacy asset."""

    root = root.resolve()
    authorization = validate_legacy_sis_stack_contract(root)
    implementation_commit = _full_commit(implementation_commit)
    _verify_checkout(root, implementation_commit)
    evidence = _project_path(evidence_output_path, name="legacy SIS evidence output")
    if evidence.exists():
        raise LegacySISStressGateError("legacy SIS evidence identity already exists")
    release_relative = _repo_relative(
        root,
        output_path,
        prefix=("results", "phase7"),
        name="legacy SIS release packet",
    )
    if release_relative != "results/phase7/legacy_sis_release_packet.json":
        raise LegacySISStressGateError("legacy SIS release-packet path changed")
    immutable = _immutable_execution(
        implementation_commit=implementation_commit,
        wheel_path=wheel_path,
        exact_wheel_test_result_path=exact_wheel_test_result_path,
        environment_lock_path=environment_lock_path,
    )
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "legacy_asset_read_authorized": False,
        "implementation_commit": implementation_commit,
        "frozen_legacy_identity": dict(authorization["frozen_legacy_identity"]),
        "descriptive_metric_contract": dict(
            authorization["descriptive_metric_contract"]
        ),
        "claim_boundary": dict(authorization["claim_boundary"]),
        "immutable_execution": immutable,
        "evidence_output_path": str(evidence),
        "review_scope": {
            "legacy_asset_read_authorized": True,
            "baseline_execution_authorized": True,
            "legacy_checkpoint_deserialization_authorized": False,
            "legacy_asset_write_authorized": False,
            "scientific_data_access_authorized": False,
            "final_evaluation_access_authorized": False,
            "v2_final_data_application_authorized": False,
            "manuscript_claim_finalization_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "read_only_evidence_contract": {
            "checkpoint_deserialization_forbidden": True,
            "before_after_inode_size_mtime_identity_required": True,
            "legacy_write_forbidden": True,
            "output_must_be_new_project_only": True,
        },
        "future_authorization_path": (
            "configs/execution/phase7_legacy_sis_read_only_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase7/legacy_sis_delegated_review.json"
        ),
        "release_packet_repository_path": release_relative,
    }


def build_legacy_sis_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact packet after a separate delegated read-only review."""

    root = root.resolve()
    validate_legacy_sis_stack_contract(root)
    packet = _load_json(release_packet_path)
    review = _load_json(delegated_review_path)
    packet_hash = _sha256(release_packet_path)
    scope = review.get("authorization_scope", {})
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("legacy_asset_read_authorized") is not False
        or review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by")
        != "codex_as_delegated_scientific_and_engineering_reviewer"
        or set(scope) != set(REVIEW_FIELDS)
        or any(scope.get(key) != packet["review_scope"][key] for key in REVIEW_FIELDS)
    ):
        raise LegacySISStressGateError("delegated legacy SIS review is not exact")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="legacy SIS release packet",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="legacy SIS delegated review",
    )
    authorization_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="legacy SIS authorization",
    )
    if (
        release_relative != packet["release_packet_repository_path"]
        or review_relative != packet["future_review_path"]
        or authorization_relative != packet["future_authorization_path"]
    ):
        raise LegacySISStressGateError("legacy SIS release evidence path changed")
    return {
        "phase": "7-legacy-sis-read-only-reproduction",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": release_relative,
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "frozen_legacy_identity": dict(packet["frozen_legacy_identity"]),
        "descriptive_metric_contract": dict(packet["descriptive_metric_contract"]),
        "claim_boundary": dict(packet["claim_boundary"]),
        "authorization": dict(packet["review_scope"]),
        "immutable_execution": dict(packet["immutable_execution"]),
        "evidence_output_path": packet["evidence_output_path"],
        "post_freeze_allowed_paths": [
            authorization_relative,
            release_relative,
            review_relative,
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE7_LEGACY_SIS_READ_ONLY_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase7/legacy_sis_summary.json",
        ],
        "stop_after_read_only_reproduction": True,
    }


def validate_legacy_sis_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    checkpoint_path: Path,
    predictions_path: Path,
    evidence_output_path: Path,
    execution_commit: str,
) -> LegacySISStressContract:
    """Validate exact immutable inputs before any legacy byte is read."""

    validate_legacy_sis_stack_contract(root)
    authorization = load_yaml(authorization_path)
    flags = authorization.get("authorization", {})
    if (
        authorization.get("authorization_status") != AUTHORIZATION_STATUS
        or set(flags) != set(REVIEW_FIELDS)
        or flags.get("legacy_asset_read_authorized") is not True
        or flags.get("baseline_execution_authorized") is not True
        or any(
            flags.get(key) is not False
            for key in REVIEW_FIELDS
            if key
            not in {
                "legacy_asset_read_authorized",
                "baseline_execution_authorized",
            }
        )
        or authorization.get("stop_after_read_only_reproduction") is not True
    ):
        raise LegacySISStressGateError("legacy SIS read-only execution gate is absent")
    identity = authorization.get("frozen_legacy_identity", {})
    configured_checkpoint = Path(str(identity.get("checkpoint_path", ""))).resolve()
    configured_predictions = Path(
        str(identity.get("validation_predictions_path", ""))
    ).resolve()
    configured_output = Path(
        str(authorization.get("evidence_output_path", ""))
    ).resolve()
    if (
        checkpoint_path.resolve() != configured_checkpoint
        or predictions_path.resolve() != configured_predictions
        or evidence_output_path.resolve() != configured_output
        or not configured_checkpoint.is_relative_to(LEGACY_ROOT)
        or not configured_predictions.is_relative_to(LEGACY_ROOT)
        or not configured_output.is_relative_to(NEW_PROJECT_ROOT)
        or configured_output.exists()
    ):
        raise LegacySISStressGateError("legacy SIS execution identity changed")
    immutable = authorization.get("immutable_execution", {})
    wheel = Path(str(immutable.get("wheel_path", ""))).resolve()
    environment = Path(str(immutable.get("environment_lock_path", ""))).resolve()
    if (
        immutable.get("git_commit") != execution_commit
        or not wheel.is_file()
        or _sha256(wheel) != immutable.get("wheel_sha256")
        or not environment.is_file()
        or _sha256(environment) != immutable.get("environment_lock_sha256")
        or immutable.get("editable_install_authorized") is not False
    ):
        raise LegacySISStressGateError("legacy SIS immutable execution changed")
    try:
        _verify_training_checkout(root, execution_commit, authorization_path)
    except (TrainingGateError, ValueError) as error:
        raise LegacySISStressGateError("legacy SIS checkout changed") from error
    return legacy_sis_contract(authorization)


def run_authorized_legacy_sis_reproduction(
    root: Path,
    *,
    authorization_path: Path,
    checkpoint_path: Path,
    predictions_path: Path,
    evidence_output_path: Path,
    execution_commit: str,
) -> Mapping[str, Any]:
    """Run the exact read-only reproduction and write only new-project evidence."""

    contract = validate_legacy_sis_execution_gate(
        root,
        authorization_path=authorization_path,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        evidence_output_path=evidence_output_path,
        execution_commit=execution_commit,
    )
    result = verify_legacy_sis_stress_control(
        checkpoint_path,
        predictions_path,
        contract,
    )
    completed = {
        **result,
        "execution_commit": execution_commit,
        "authorization_sha256": _sha256(authorization_path),
    }
    write_legacy_sis_evidence(evidence_output_path, completed)
    return completed
