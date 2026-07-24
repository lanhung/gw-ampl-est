"""Two-stage exact release control for ablation calibration and IID analysis.

The builders in this module are intentionally non-executable.  The first
release binds the six completed ablation checkpoints to the already-published
calibration-fit cases and allocates six independent calibration maps.  The
second release cannot be built until those maps and the primary same-seed IID
scores exist; it then binds six ablation IID score artifacts and six paired
comparisons.  No builder opens a scientific array or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from .ablation_authorization import (
    AUTHORIZATION_STATUS as ABLATION_TRAINING_AUTHORIZATION_STATUS,
)
from .ablation_evaluation import (
    ABLATION_EVALUATION_CONFIG_HASH,
    ABLATION_VIEWS,
    BOOTSTRAP_REPLICATES,
    CALIBRATION_COUNT,
    IID_COUNT,
    MODEL_SEEDS,
    POSTERIOR_DRAW_COUNT,
    load_ablation_evaluation_contract,
)
from .calibration_execution_authorization import (
    SCORE_AUTHORIZATION_STATUS,
    STATISTICS_AUTHORIZATION_STATUS,
)
from .contracts import TrainingGateError
from .final_inference_authorization import (
    AUTHORIZATION_STATUS as FINAL_INFERENCE_AUTHORIZATION_STATUS,
)
from .terminal131 import TRAIN_131K_COUNT

STACK_AUTHORIZATION = (
    "configs/execution/"
    "phase7_ablation_evaluation_release_stack_authorization.yaml"
)
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")

CALIBRATION_RELEASE_STATUS = (
    "ready_for_delegated_ablation_calibration_execution_review"
)
CALIBRATION_REVIEW_STATUS = (
    "delegated_ablation_calibration_execution_review_approved"
)
CALIBRATION_AUTHORIZATION_STATUS = (
    "authorized_ablation_calibration_score_and_map_execution_only"
)
IID_RELEASE_STATUS = "ready_for_delegated_ablation_iid_execution_review"
IID_REVIEW_STATUS = "delegated_ablation_iid_execution_review_approved"
IID_AUTHORIZATION_STATUS = (
    "authorized_ablation_iid_inference_and_paired_comparison_only"
)

CALIBRATION_REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "ablation_views",
    "model_seeds",
    "ablation_checkpoint_count",
    "calibration_case_count",
    "calibration_score_artifact_count",
    "calibration_map_count",
    "scientific_checkpoint_access_authorized",
    "calibration_fit_data_access_authorized",
    "ablation_calibration_score_execution_authorized",
    "ablation_calibration_map_fit_authorized",
)
CALIBRATION_CLOSED_BOUNDARIES = (
    "iid_unsealing_authorized",
    "iid_inference_or_comparison_authorized",
    "primary_calibration_refit_authorized",
    "sbc_authorized",
    "model_training_or_tuning_authorized",
    "architecture_or_size_selection_authorized",
    "ood_or_mismatch_ablation_authorized",
    "gwosc_gwtc_access_authorized",
)
IID_REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "ablation_views",
    "model_seeds",
    "iid_case_count",
    "ablation_iid_score_artifact_count",
    "paired_comparison_count",
    "final_iid_unsealing_authorized",
    "ablation_checkpoint_access_authorized",
    "matching_ablation_calibration_map_access_authorized",
    "primary_same_seed_iid_score_access_authorized",
    "ablation_iid_inference_authorized",
    "paired_comparison_execution_authorized",
)
IID_CLOSED_BOUNDARIES = (
    "calibration_refit_authorized",
    "sbc_authorized",
    "model_training_or_tuning_authorized",
    "architecture_or_size_selection_authorized",
    "non_iid_ablation_inference_authorized",
    "result_driven_retraining_authorized",
    "manuscript_claim_finalization_authorized",
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


def _full_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise TrainingGateError("ablation evaluation commit is not full length")
    return commit


def _repo_relative(
    root: Path, path: Path, *, prefix: tuple[str, ...], name: str
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


def _verify_checkout(root: Path, commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit:
            raise TrainingGateError("ablation evaluation synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("ablation evaluation checkout identity is absent")
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
        raise TrainingGateError(
            "ablation evaluation checkout is not exact and clean"
        )


def load_ablation_evaluation_release_stack_contract(
    root: Path,
) -> Mapping[str, Any]:
    """Validate the synthetic-only release-stack boundary."""

    load_ablation_evaluation_contract(root)
    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise TrainingGateError("ablation evaluation release gate is absent")
    frozen = authorization.get("frozen_contracts", {})
    if (
        frozen.get("ablation_evaluation_hash")
        != ABLATION_EVALUATION_CONFIG_HASH
        or int(frozen.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or frozen.get("ablation_views") != list(ABLATION_VIEWS)
        or frozen.get("model_seeds") != list(MODEL_SEEDS)
        or int(frozen.get("ablation_checkpoint_count", -1)) != 6
        or int(frozen.get("calibration_case_count", -1))
        != CALIBRATION_COUNT
        or int(frozen.get("ablation_calibration_map_count", -1)) != 6
        or int(frozen.get("iid_case_count", -1)) != IID_COUNT
        or int(frozen.get("ablation_iid_score_artifact_count", -1)) != 6
        or int(frozen.get("paired_comparison_count", -1)) != 6
        or int(frozen.get("posterior_draws_per_case", -1))
        != POSTERIOR_DRAW_COUNT
        or int(frozen.get("bootstrap_replicates", -1))
        != BOOTSTRAP_REPLICATES
        or frozen.get("calibration_release_precedes_iid_release") is not True
        or frozen.get("same_cases_as_primary_required") is not True
        or frozen.get("same_seed_primary_comparator_required") is not True
    ):
        raise TrainingGateError("ablation evaluation release contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "nonauthorizing_calibration_release_packet_implementation_authorized",
        "nonauthorizing_iid_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise TrainingGateError(
            "ablation evaluation release stack opened scientific execution"
        )
    return authorization


def _immutable_inference(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="ablation inference wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="ablation inference exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path,
        name="ablation inference environment",
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
        or result.get("scientific_data_opened") is not False
        or result.get("optimizer_started") is not False
    ):
        raise TrainingGateError("ablation inference exact-wheel evidence failed")
    return {
        "git_commit": implementation_commit,
        "wheel_path": str(wheel),
        "wheel_sha256": wheel_hash,
        "exact_wheel_test_result_path": str(result_path),
        "exact_wheel_test_result_sha256": _sha256(result_path),
        "environment_lock_path": str(environment),
        "environment_lock_sha256": _sha256(environment),
        "editable_install_authorized": False,
    }


def _completed_ablation_checkpoints(
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    if authorization.get("authorization_status") != (
        ABLATION_TRAINING_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("six-fit ablation authorization is absent")
    if (
        int(authorization.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or authorization.get("authorized_ablation_views")
        != list(ABLATION_VIEWS)
        or authorization.get("authorized_training_seeds") != list(MODEL_SEEDS)
        or int(authorization.get("maximum_fit_count", -1)) != 6
    ):
        raise TrainingGateError("ablation training identity changed")
    outputs = authorization.get("fit_output_identities", {})
    artifacts: dict[str, Any] = {}
    observed_paths: set[str] = set()
    for view in ABLATION_VIEWS:
        artifacts[view] = {}
        for seed in MODEL_SEEDS:
            run_root = _project_path(
                Path(str(outputs.get(view, {}).get(str(seed), ""))),
                name=f"{view} seed-{seed} ablation result",
            )
            summary_path = run_root / "run_summary.json"
            checkpoint_path = run_root / "best.ckpt"
            if not summary_path.is_file() or not checkpoint_path.is_file():
                raise TrainingGateError("completed ablation fit artifact is absent")
            summary = _load_json(summary_path)
            identity = summary.get("identity", {})
            if (
                summary.get("status")
                != "completed_ablation_fit_and_development_validation"
                or summary.get("ablation_view") != view
                or int(summary.get("seed", -1)) != seed
                or int(identity.get("training_rung_count", -1))
                != TRAIN_131K_COUNT
                or summary.get("calibration_or_sbc_accessed") is not False
                or summary.get("final_evaluation_accessed") is not False
            ):
                raise TrainingGateError("completed ablation fit violates its contract")
            if str(run_root) in observed_paths:
                raise TrainingGateError("ablation fit output identity is duplicated")
            observed_paths.add(str(run_root))
            artifacts[view][str(seed)] = {
                "run_root": str(run_root),
                "run_summary_path": str(summary_path.resolve()),
                "run_summary_sha256": _sha256(summary_path),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "model_configuration_hash": identity.get(
                    "model_configuration_hash"
                ),
            }
    return artifacts


def _primary_statistics_completed(
    authorization: Mapping[str, Any],
    *,
    selected_architecture_id: str,
) -> Mapping[str, Any]:
    if authorization.get("authorization_status") != (
        STATISTICS_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("primary calibration/SBC statistics are incomplete")
    selected = authorization.get("selected_architecture", {})
    if (
        selected.get("architecture_id") != selected_architecture_id
        or int(selected.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or authorization.get("authorized_model_seeds") != list(MODEL_SEEDS)
    ):
        raise TrainingGateError("primary statistics use a different selected model")
    roots = authorization.get("statistics_output_roots", {})
    score_artifacts = authorization.get("score_artifacts_by_seed", {})
    calibration_artifacts: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        root = _project_path(
            Path(str(roots.get(str(seed), ""))),
            name=f"primary seed-{seed} statistics",
        )
        summary_path = root / "run_summary.json"
        if not summary_path.is_file():
            raise TrainingGateError("primary calibration/SBC result is absent")
        summary = _load_json(summary_path)
        if (
            summary.get("status")
            != "completed_calibration_fit_and_independent_sbc"
        ):
            raise TrainingGateError("primary calibration/SBC result did not complete")
        calibration = score_artifacts.get(str(seed), {}).get(
            "calibration_fit", {}
        )
        score_path = _project_path(
            Path(str(calibration.get("path", ""))),
            name=f"primary seed-{seed} calibration score",
            require_file=True,
        )
        if _sha256(score_path) != calibration.get("sha256"):
            raise TrainingGateError("primary calibration score identity changed")
        calibration_artifacts[str(seed)] = {
            "path": str(score_path),
            "sha256": _sha256(score_path),
        }
    return calibration_artifacts


def build_ablation_calibration_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    ablation_training_authorization_path: Path,
    primary_score_authorization_path: Path,
    primary_statistics_authorization_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    calibration_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Bind six completed fits to the shared calibration-fit publication."""

    root = root.resolve()
    load_ablation_evaluation_release_stack_contract(root)
    implementation_commit = _full_commit(implementation_commit)
    _verify_checkout(root, implementation_commit)
    training = load_yaml(ablation_training_authorization_path)
    checkpoints = _completed_ablation_checkpoints(training)
    primary_model_hash = str(
        training.get("selected_primary_model_configuration_hash", "")
    )
    decision_path = _project_path(
        Path(str(training.get("selected_architecture_decision_path", ""))),
        name="selected architecture decision",
        require_file=True,
    )
    decision = _load_json(decision_path)
    architecture_id = str(decision.get("selected_architecture_id", ""))
    if (
        not primary_model_hash
        or not architecture_id
        or _sha256(decision_path)
        != training.get("selected_architecture_decision_sha256")
    ):
        raise TrainingGateError("selected primary model hash is absent")
    score_authorization = load_yaml(primary_score_authorization_path)
    if score_authorization.get("authorization_status") != SCORE_AUTHORIZATION_STATUS:
        raise TrainingGateError("primary calibration score authorization is absent")
    score_selected = score_authorization.get("selected_architecture", {})
    if (
        score_selected.get("architecture_id") != architecture_id
        or score_selected.get("model_configuration_hash") != primary_model_hash
        or int(score_selected.get("locked_training_rung", -1))
        != TRAIN_131K_COUNT
    ):
        raise TrainingGateError("primary score and ablation model identities differ")
    statistics = load_yaml(primary_statistics_authorization_path)
    primary_calibration_scores = _primary_statistics_completed(
        statistics, selected_architecture_id=architecture_id
    )
    publication = score_authorization.get("development_publication", {})
    publication_root = _project_path(
        Path(str(publication.get("parent_root", ""))),
        name="calibration/SBC publication",
    )
    manifest_path = publication_root / "dataset_manifest.json"
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != publication.get("parent_manifest_sha256")
        or int(publication.get("calibration_fit_accepted_count", -1))
        != CALIBRATION_COUNT
    ):
        raise TrainingGateError("shared calibration-fit publication changed")
    output_root = _project_path(
        calibration_output_root, name="ablation calibration output root"
    )
    if output_root.exists():
        raise TrainingGateError("ablation calibration output identity exists")
    score_outputs = {
        view: {
            str(seed): str(
                output_root / view / f"seed-{seed}" / "calibration_scores.npz"
            )
            for seed in MODEL_SEEDS
        }
        for view in ABLATION_VIEWS
    }
    map_outputs = {
        view: {
            str(seed): str(
                output_root
                / view
                / f"seed-{seed}"
                / "calibration_region_maps.json"
            )
            for seed in MODEL_SEEDS
        }
        for view in ABLATION_VIEWS
    }
    scope = {
        "locked_training_rung": TRAIN_131K_COUNT,
        "selected_architecture_id": architecture_id,
        "ablation_views": list(ABLATION_VIEWS),
        "model_seeds": list(MODEL_SEEDS),
        "ablation_checkpoint_count": 6,
        "calibration_case_count": CALIBRATION_COUNT,
        "calibration_score_artifact_count": 6,
        "calibration_map_count": 6,
        "scientific_checkpoint_access_authorized": True,
        "calibration_fit_data_access_authorized": True,
        "ablation_calibration_score_execution_authorized": True,
        "ablation_calibration_map_fit_authorized": True,
    }
    return {
        "status": CALIBRATION_RELEASE_STATUS,
        "authorization_created": False,
        "ablation_calibration_execution_authorized": False,
        "implementation_commit": implementation_commit,
        "selected_architecture": {
            "architecture_id": architecture_id,
            "model_configuration_hash": primary_model_hash,
            "locked_training_rung": TRAIN_131K_COUNT,
        },
        "ablation_checkpoints": checkpoints,
        "primary_statistics_authorization": {
            "path": _repo_relative(
                root,
                primary_statistics_authorization_path,
                prefix=("configs", "execution"),
                name="primary statistics authorization",
            ),
            "sha256": _sha256(primary_statistics_authorization_path),
        },
        "primary_same_seed_calibration_scores": primary_calibration_scores,
        "calibration_publication": {
            **dict(publication),
            "parent_manifest_sha256": _sha256(manifest_path),
        },
        "immutable_inference": _immutable_inference(
            implementation_commit=implementation_commit,
            wheel_path=wheel_path,
            exact_wheel_test_result_path=exact_wheel_test_result_path,
            environment_lock_path=environment_lock_path,
        ),
        "calibration_score_outputs": score_outputs,
        "calibration_map_outputs": map_outputs,
        "review_scope": scope,
        "closed_boundaries": {
            name: False for name in CALIBRATION_CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/"
            "phase7_ablation_calibration_execution_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase7/ablation_calibration_execution_review.json"
        ),
        "release_packet_repository_path": _repo_relative(
            root,
            output_path,
            prefix=("results", "phase7"),
            name="ablation calibration release packet",
        ),
    }


def _reviewed_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
    release_status: str,
    review_status: str,
    packet_execution_flag: str,
    review_scope_names: tuple[str, ...],
    closed_names: tuple[str, ...],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str, str]:
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if (
        packet.get("status") != release_status
        or packet.get("authorization_created") is not False
        or packet.get(packet_execution_flag) is not False
        or review.get("status") != review_status
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by")
        != "codex_as_delegated_scientific_and_engineering_reviewer"
        or set(scope) != set(review_scope_names)
        or any(scope.get(name) != packet["review_scope"][name] for name in review_scope_names)
        or set(closed) != set(closed_names)
        or any(closed.get(name) is not False for name in closed_names)
    ):
        raise TrainingGateError("delegated ablation evaluation review is not exact")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="ablation evaluation release packet",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="ablation evaluation review",
    )
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="ablation evaluation authorization",
    )
    if (
        release_relative != packet.get("release_packet_repository_path")
        or review_relative != packet.get("future_review_path")
        or output_relative != packet.get("future_authorization_path")
    ):
        raise TrainingGateError("ablation evaluation evidence path changed")
    return packet, review, packet_hash, review_relative, output_relative


def build_ablation_calibration_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact six-map calibration packet after review."""

    packet, review, packet_hash, review_relative, output_relative = (
        _reviewed_authorization(
            root.resolve(),
            release_packet_path=release_packet_path,
            delegated_review_path=delegated_review_path,
            output_path=output_path,
            release_status=CALIBRATION_RELEASE_STATUS,
            review_status=CALIBRATION_REVIEW_STATUS,
            packet_execution_flag="ablation_calibration_execution_authorized",
            review_scope_names=CALIBRATION_REVIEW_SCOPE,
            closed_names=CALIBRATION_CLOSED_BOUNDARIES,
        )
    )
    return {
        "phase": "7-ablation-calibration",
        "authorization_status": CALIBRATION_AUTHORIZATION_STATUS,
        "authorized_by": review["reviewed_by"],
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": packet["release_packet_repository_path"],
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "authorization": {
            "scientific_checkpoint_access_authorized": True,
            "calibration_fit_data_access_authorized": True,
            "ablation_calibration_score_execution_authorized": True,
            "ablation_calibration_map_fit_authorized": True,
            **{name: False for name in CALIBRATION_CLOSED_BOUNDARIES},
        },
        "selected_architecture": dict(packet["selected_architecture"]),
        "ablation_checkpoints": dict(packet["ablation_checkpoints"]),
        "calibration_publication": dict(packet["calibration_publication"]),
        "primary_same_seed_calibration_scores": dict(
            packet["primary_same_seed_calibration_scores"]
        ),
        "immutable_inference": dict(packet["immutable_inference"]),
        "calibration_score_outputs": dict(packet["calibration_score_outputs"]),
        "calibration_map_outputs": dict(packet["calibration_map_outputs"]),
        "post_freeze_allowed_paths": [
            output_relative,
            packet["release_packet_repository_path"],
            packet["future_review_path"],
            "results/phase7/ablation_calibration_summary.json",
            "docs/reports/PHASE7_ABLATION_EVALUATION_REPORT.md",
        ],
        "stop_after_six_calibration_maps": True,
    }


def _completed_ablation_maps(
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    if authorization.get("authorization_status") != (
        CALIBRATION_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("ablation calibration authorization is absent")
    maps = authorization.get("calibration_map_outputs", {})
    scores = authorization.get("calibration_score_outputs", {})
    checkpoints = authorization.get("ablation_checkpoints", {})
    artifacts: dict[str, Any] = {}
    for view in ABLATION_VIEWS:
        artifacts[view] = {}
        for seed in MODEL_SEEDS:
            map_path = _project_path(
                Path(str(maps.get(view, {}).get(str(seed), ""))),
                name=f"{view} seed-{seed} calibration map",
                require_file=True,
            )
            score_path = _project_path(
                Path(str(scores.get(view, {}).get(str(seed), ""))),
                name=f"{view} seed-{seed} calibration score",
                require_file=True,
            )
            summary_path = map_path.with_name("run_summary.json")
            if not summary_path.is_file():
                raise TrainingGateError("ablation calibration summary is absent")
            summary = _load_json(summary_path)
            checkpoint_hash = checkpoints[view][str(seed)]["checkpoint_sha256"]
            if (
                summary.get("status")
                != "completed_ablation_calibration_score_and_map"
                or summary.get("view") != view
                or int(summary.get("model_seed", -1)) != seed
                or int(summary.get("calibration_case_count", -1))
                != CALIBRATION_COUNT
                or summary.get("checkpoint_sha256") != checkpoint_hash
                or summary.get("calibration_map_sha256") != _sha256(map_path)
                or summary.get("calibration_score_sha256") != _sha256(score_path)
                or summary.get("iid_accessed") is not False
            ):
                raise TrainingGateError("ablation calibration result is incomplete")
            artifacts[view][str(seed)] = {
                "calibration_map_path": str(map_path),
                "calibration_map_sha256": _sha256(map_path),
                "calibration_score_path": str(score_path),
                "calibration_score_sha256": _sha256(score_path),
                "run_summary_path": str(summary_path.resolve()),
                "run_summary_sha256": _sha256(summary_path),
                "checkpoint_sha256": checkpoint_hash,
            }
    return artifacts


def _primary_iid_scores(
    authorization: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    if authorization.get("authorization_status") != (
        FINAL_INFERENCE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("primary final inference authorization is absent")
    outputs = authorization.get("score_outputs", {})
    artifacts: dict[str, Any] = {}
    namespace_id: str | None = None
    for seed in MODEL_SEEDS:
        seed_outputs = outputs.get(str(seed), {})
        matches: list[tuple[str, Path, Mapping[str, Any]]] = []
        for candidate_namespace, value in seed_outputs.items():
            path = _project_path(
                Path(str(value)),
                name=f"primary seed-{seed} final score",
                require_file=True,
            )
            summary_path = path.with_suffix(".summary.json")
            if not summary_path.is_file():
                raise TrainingGateError("primary final score summary is absent")
            summary = _load_json(summary_path)
            if summary.get("split") == "iid_test":
                matches.append((str(candidate_namespace), path, summary))
        if len(matches) != 1:
            raise TrainingGateError("primary inference must contain one IID namespace")
        observed_namespace, path, summary = matches[0]
        if namespace_id is None:
            namespace_id = observed_namespace
        if (
            observed_namespace != namespace_id
            or int(summary.get("model_seed", -1)) != seed
            or int(summary.get("case_count", -1)) != IID_COUNT
            or summary.get("score_artifact_sha256") != _sha256(path)
            or summary.get("calibration_refit") is not False
            or summary.get("model_retrained_or_tuned") is not False
        ):
            raise TrainingGateError("primary IID score identity changed")
        artifacts[str(seed)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "summary_path": str(path.with_suffix(".summary.json").resolve()),
            "summary_sha256": _sha256(path.with_suffix(".summary.json")),
        }
    if namespace_id is None:
        raise TrainingGateError("primary IID namespace is absent")
    return namespace_id, artifacts


def build_ablation_iid_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    ablation_calibration_authorization_path: Path,
    primary_final_inference_authorization_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    iid_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Bind six maps and same-seed primary IID scores before IID unsealing."""

    root = root.resolve()
    load_ablation_evaluation_release_stack_contract(root)
    implementation_commit = _full_commit(implementation_commit)
    _verify_checkout(root, implementation_commit)
    calibration_authorization = load_yaml(
        ablation_calibration_authorization_path
    )
    maps = _completed_ablation_maps(calibration_authorization)
    final_authorization = load_yaml(primary_final_inference_authorization_path)
    namespace_id, primary_scores = _primary_iid_scores(final_authorization)
    selected = calibration_authorization.get("selected_architecture", {})
    final_selected = final_authorization.get("selected_architecture", {})
    if (
        selected.get("architecture_id") != final_selected.get("architecture_id")
        or selected.get("model_configuration_hash")
        != final_selected.get("model_configuration_hash")
        or int(selected.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
    ):
        raise TrainingGateError("ablation and primary IID selected models differ")
    output_root = _project_path(iid_output_root, name="ablation IID output root")
    if output_root.exists():
        raise TrainingGateError("ablation IID output identity exists")
    score_outputs = {
        view: {
            str(seed): str(output_root / view / f"seed-{seed}" / "iid_scores.npz")
            for seed in MODEL_SEEDS
        }
        for view in ABLATION_VIEWS
    }
    comparison_outputs = {
        view: {
            str(seed): str(
                output_root / view / f"seed-{seed}" / "paired_comparison.json"
            )
            for seed in MODEL_SEEDS
        }
        for view in ABLATION_VIEWS
    }
    scope = {
        "locked_training_rung": TRAIN_131K_COUNT,
        "selected_architecture_id": selected["architecture_id"],
        "ablation_views": list(ABLATION_VIEWS),
        "model_seeds": list(MODEL_SEEDS),
        "iid_case_count": IID_COUNT,
        "ablation_iid_score_artifact_count": 6,
        "paired_comparison_count": 6,
        "final_iid_unsealing_authorized": True,
        "ablation_checkpoint_access_authorized": True,
        "matching_ablation_calibration_map_access_authorized": True,
        "primary_same_seed_iid_score_access_authorized": True,
        "ablation_iid_inference_authorized": True,
        "paired_comparison_execution_authorized": True,
    }
    return {
        "status": IID_RELEASE_STATUS,
        "authorization_created": False,
        "ablation_iid_execution_authorized": False,
        "implementation_commit": implementation_commit,
        "selected_architecture": dict(selected),
        "iid_namespace_id": namespace_id,
        "iid_case_count": IID_COUNT,
        "ablation_calibration_authorization": {
            "path": _repo_relative(
                root,
                ablation_calibration_authorization_path,
                prefix=("configs", "execution"),
                name="ablation calibration authorization",
            ),
            "sha256": _sha256(ablation_calibration_authorization_path),
        },
        "ablation_checkpoints": dict(
            calibration_authorization["ablation_checkpoints"]
        ),
        "ablation_calibration_maps": maps,
        "primary_final_inference_authorization": {
            "path": _repo_relative(
                root,
                primary_final_inference_authorization_path,
                prefix=("configs", "execution"),
                name="primary final inference authorization",
            ),
            "sha256": _sha256(primary_final_inference_authorization_path),
        },
        "sealed_publication": dict(final_authorization["sealed_publication"]),
        "primary_same_seed_iid_scores": primary_scores,
        "immutable_inference": _immutable_inference(
            implementation_commit=implementation_commit,
            wheel_path=wheel_path,
            exact_wheel_test_result_path=exact_wheel_test_result_path,
            environment_lock_path=environment_lock_path,
        ),
        "ablation_iid_score_outputs": score_outputs,
        "paired_comparison_outputs": comparison_outputs,
        "review_scope": scope,
        "closed_boundaries": {
            name: False for name in IID_CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/phase7_ablation_iid_execution_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase7/ablation_iid_execution_review.json"
        ),
        "release_packet_repository_path": _repo_relative(
            root,
            output_path,
            prefix=("results", "phase7"),
            name="ablation IID release packet",
        ),
    }


def build_ablation_iid_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact six-score/six-comparison IID packet after review."""

    packet, review, packet_hash, review_relative, output_relative = (
        _reviewed_authorization(
            root.resolve(),
            release_packet_path=release_packet_path,
            delegated_review_path=delegated_review_path,
            output_path=output_path,
            release_status=IID_RELEASE_STATUS,
            review_status=IID_REVIEW_STATUS,
            packet_execution_flag="ablation_iid_execution_authorized",
            review_scope_names=IID_REVIEW_SCOPE,
            closed_names=IID_CLOSED_BOUNDARIES,
        )
    )
    return {
        "phase": "7-ablation-iid",
        "authorization_status": IID_AUTHORIZATION_STATUS,
        "authorized_by": review["reviewed_by"],
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": packet["release_packet_repository_path"],
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "authorization": {
            "final_iid_unsealing_authorized": True,
            "ablation_checkpoint_access_authorized": True,
            "matching_ablation_calibration_map_access_authorized": True,
            "primary_same_seed_iid_score_access_authorized": True,
            "ablation_iid_inference_authorized": True,
            "paired_comparison_execution_authorized": True,
            **{name: False for name in IID_CLOSED_BOUNDARIES},
        },
        "selected_architecture": dict(packet["selected_architecture"]),
        "iid_namespace_id": packet["iid_namespace_id"],
        "iid_case_count": IID_COUNT,
        "ablation_checkpoints": dict(packet["ablation_checkpoints"]),
        "ablation_calibration_maps": dict(
            packet["ablation_calibration_maps"]
        ),
        "sealed_publication": dict(packet["sealed_publication"]),
        "primary_same_seed_iid_scores": dict(
            packet["primary_same_seed_iid_scores"]
        ),
        "immutable_inference": dict(packet["immutable_inference"]),
        "ablation_iid_score_outputs": dict(
            packet["ablation_iid_score_outputs"]
        ),
        "paired_comparison_outputs": dict(
            packet["paired_comparison_outputs"]
        ),
        "post_freeze_allowed_paths": [
            output_relative,
            packet["release_packet_repository_path"],
            packet["future_review_path"],
            "results/phase7/ablation_iid_summary.json",
            "docs/reports/PHASE7_ABLATION_EVALUATION_REPORT.md",
        ],
        "stop_after_six_iid_scores_and_paired_comparisons": True,
    }
