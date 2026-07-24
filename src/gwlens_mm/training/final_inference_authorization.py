"""Exact release control for sealed final-evaluation inference.

This module never opens a final record or a checkpoint.  It hash-binds the
completed terminal architecture, selected three-seed checkpoints, sealed final
publication and same-seed calibration/SBC outputs into a non-authorizing
packet.  A separate delegated review is required before the runtime
authorization consumed by :mod:`gwlens_mm.training.final_inference` can exist.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..production.final_evaluation_authorization import (
    AUTHORIZATION_STATUS as MATERIALIZATION_AUTHORIZATION_STATUS,
)
from .calibration_execution_authorization import (
    STATISTICS_AUTHORIZATION_STATUS,
    _selected_checkpoint_artifacts,
)
from .contracts import TrainingGateError
from .final_inference import (
    FINAL_ANALYSIS_HASH,
    FINAL_CASE_COUNT,
    FINAL_COMMITMENT_SHA256,
    FINAL_NAMESPACE_COUNT,
    MAXIMUM_DRAW_MICROBATCH,
    MODEL_SEEDS,
    POSTERIOR_DRAW_COUNT,
    _expected_namespaces,
    _validate_calibration_statistics,
    resolve_sealed_final_publication,
)
from .reference_baseline import REFERENCE_CONFIG_HASH
from .terminal131 import TRAIN_131K_COUNT
from .terminal_architecture_authorization import (
    AUTHORIZATION_STATUS as ARCHITECTURE_AUTHORIZATION_STATUS,
)

STACK_AUTHORIZATION = "configs/execution/phase7_final_inference_release_stack_authorization.yaml"
RELEASE_STATUS = "ready_for_delegated_final_inference_review"
REVIEW_STATUS = "delegated_final_inference_review_approved"
AUTHORIZATION_STATUS = "authorized_final_evaluation_inference_only"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
SCORE_ARTIFACT_COUNT = MODEL_SEEDS.__len__() * FINAL_NAMESPACE_COUNT

REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "model_seeds",
    "namespace_count",
    "accepted_case_count",
    "score_artifact_count",
    "final_evaluation_unsealing_authorized",
    "final_evaluation_data_access_authorized",
    "selected_checkpoint_inference_authorized",
    "same_seed_calibration_map_application_authorized",
    "immutable_score_artifact_creation_authorized",
)
REVIEW_CLOSED_BOUNDARIES = (
    "model_training_or_tuning_authorized",
    "calibration_refit_authorized",
    "architecture_or_size_selection_authorized",
    "final_result_threshold_change_authorized",
    "reference_baseline_execution_authorized",
    "ablation_training_authorized",
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


def _full_commit(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a full Git commit")
    return result


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


def _verify_checkout(root: Path, commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit:
            raise TrainingGateError("final-inference synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("final-inference checkout identity is absent")
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
        raise TrainingGateError("final-inference checkout is not exact and clean")


def load_final_inference_release_stack_contract(root: Path) -> Mapping[str, Any]:
    """Validate that the release stack remains implementation-only."""

    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise TrainingGateError("final-inference release-stack gate is absent")
    frozen = authorization.get("frozen_contracts", {})
    if (
        int(frozen.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or frozen.get("model_seeds") != list(MODEL_SEEDS)
        or frozen.get("final_analysis_hash") != FINAL_ANALYSIS_HASH
        or frozen.get("reference_baseline_hash") != REFERENCE_CONFIG_HASH
        or frozen.get("final_generation_commitment_sha256") != FINAL_COMMITMENT_SHA256
        or int(frozen.get("namespace_count", -1)) != FINAL_NAMESPACE_COUNT
        or int(frozen.get("accepted_case_count", -1)) != FINAL_CASE_COUNT
        or int(frozen.get("model_seed_namespace_score_artifact_count", -1)) != SCORE_ARTIFACT_COUNT
        or int(frozen.get("posterior_draws_per_case", -1)) != POSTERIOR_DRAW_COUNT
    ):
        raise TrainingGateError("final-inference release parent contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise TrainingGateError("final-inference release implementation opened execution")
    return authorization


def _immutable_inference(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="final-inference wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="final-inference exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path,
        name="final-inference environment lock",
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
        raise TrainingGateError("final-inference exact-wheel evidence did not pass")
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


def _statistics_artifacts(
    statistics_authorization: Mapping[str, Any],
    *,
    selected_architecture_id: str,
) -> Mapping[str, Any]:
    if statistics_authorization.get("authorization_status") != (STATISTICS_AUTHORIZATION_STATUS):
        raise TrainingGateError("calibration/SBC statistics were not authorized")
    selected = statistics_authorization.get("selected_architecture", {})
    if (
        int(selected.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or selected.get("architecture_id") != selected_architecture_id
        or statistics_authorization.get("authorized_model_seeds") != list(MODEL_SEEDS)
    ):
        raise TrainingGateError("calibration/SBC statistics mix model identity")
    roots = statistics_authorization.get("statistics_output_roots", {})
    artifacts: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        root = _project_path(
            Path(str(roots.get(str(seed), ""))),
            name=f"seed-{seed} calibration/SBC statistics root",
        )
        paths = {
            "run_summary": root / "run_summary.json",
            "calibration_map": root / "calibration_region_maps.json",
            "sbc_summary": root / "sbc_rank_summary.json",
            "independent_coverage": root / "independent_calibrated_coverage.json",
        }
        if any(not path.is_file() for path in paths.values()):
            raise TrainingGateError("same-seed calibration/SBC result is incomplete")
        artifacts[str(seed)] = {
            **{f"{name}_path": str(path.resolve()) for name, path in paths.items()},
            **{f"{name}_sha256": _sha256(path) for name, path in paths.items()},
        }
    _validate_calibration_statistics(artifacts)
    return artifacts


def build_final_inference_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    architecture_authorization_path: Path,
    architecture_decision_path: Path,
    final_materialization_authorization_path: Path,
    final_materialization_result_path: Path,
    publication_root: Path,
    statistics_authorization_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    score_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Build a non-authorizing packet for all 45 final score artifacts."""

    root = root.resolve()
    load_final_inference_release_stack_contract(root)
    implementation_commit = _full_commit(
        implementation_commit,
        name="final-inference implementation commit",
    )
    _verify_checkout(root, implementation_commit)
    architecture_authorization = load_yaml(architecture_authorization_path)
    if architecture_authorization.get("authorization_status") != (
        ARCHITECTURE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("terminal architecture execution was not authorized")
    architecture_decision_path = _project_path(
        architecture_decision_path,
        name="terminal architecture decision",
        require_file=True,
    )
    if (
        architecture_decision_path
        != Path(
            str(architecture_authorization.get("architecture_selection_output_path", ""))
        ).resolve()
    ):
        raise TrainingGateError("terminal architecture decision path changed")
    architecture_decision = _load_json(architecture_decision_path)
    checkpoints = _selected_checkpoint_artifacts(
        root,
        architecture_authorization=architecture_authorization,
        architecture_decision=architecture_decision,
    )
    materialization_authorization = load_yaml(final_materialization_authorization_path)
    if materialization_authorization.get("authorization_status") != (
        MATERIALIZATION_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("sealed final materialization was not authorized")
    publication_root = _project_path(
        publication_root,
        name="sealed final publication",
    )
    result_path = _project_path(
        final_materialization_result_path,
        name="final materialization result",
        require_file=True,
    )
    result = _load_json(result_path)
    publication = resolve_sealed_final_publication(root, publication_root)
    if (
        result.get("status") != "passed_sealed"
        or Path(str(result.get("publication_path", ""))).resolve() != publication_root
        or result.get("accepted_pair_count") != FINAL_CASE_COUNT
        or result.get("complete_shard_count") != 160
        or result.get("namespace_count") != FINAL_NAMESPACE_COUNT
        or result.get("unsealing_authorized") is not False
        or result.get("gwosc_gwtc_accessed") is not False
        or publication.generator_commit
        != materialization_authorization.get("implementation_commit")
        or materialization_authorization.get("architecture_decision", {}).get("sha256")
        != _sha256(architecture_decision_path)
    ):
        raise TrainingGateError("sealed final publication is not an exact closed result")
    statistics_authorization = load_yaml(statistics_authorization_path)
    statistics = _statistics_artifacts(
        statistics_authorization,
        selected_architecture_id=str(checkpoints["architecture_id"]),
    )
    output_root = _project_path(
        score_output_root,
        name="final score output root",
    )
    if output_root.exists():
        raise TrainingGateError("final score output identity already exists")
    namespace_ids = tuple(item.namespace_id for item in _expected_namespaces(root))
    if len(namespace_ids) != FINAL_NAMESPACE_COUNT or len(set(namespace_ids)) != (
        FINAL_NAMESPACE_COUNT
    ):
        raise TrainingGateError("final inference namespace contract changed")
    score_outputs = {
        str(seed): {
            namespace: str(output_root / f"seed-{seed}" / f"{namespace}_scores.npz")
            for namespace in namespace_ids
        }
        for seed in MODEL_SEEDS
    }
    immutable = _immutable_inference(
        implementation_commit=implementation_commit,
        wheel_path=wheel_path,
        exact_wheel_test_result_path=exact_wheel_test_result_path,
        environment_lock_path=environment_lock_path,
    )
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("results", "phase7"),
        name="final-inference release packet",
    )
    if output_relative != "results/phase7/final_inference_release_packet.json":
        raise TrainingGateError("final-inference release-packet path changed")
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "final_inference_authorized": False,
        "implementation_commit": implementation_commit,
        "selected_architecture": {
            "decision_path": str(architecture_decision_path),
            "decision_sha256": _sha256(architecture_decision_path),
            "architecture_id": checkpoints["architecture_id"],
            "model_configuration_hash": checkpoints["model_configuration_hash"],
            "locked_training_rung": TRAIN_131K_COUNT,
        },
        "selected_seed_checkpoints": checkpoints["selected_seed_checkpoints"],
        "selected_checkpoint_shared_identity": checkpoints["shared_identity"],
        "sealed_publication": {
            "parent_root": str(publication_root),
            "manifest_sha256": publication.manifest_sha256,
            "generator_commit": publication.generator_commit,
            "accepted_case_count": FINAL_CASE_COUNT,
            "namespace_count": FINAL_NAMESPACE_COUNT,
            "materialization_result_path": str(result_path),
            "materialization_result_sha256": _sha256(result_path),
            "publication_tree_sha256": str(result.get("publication_tree_sha256", "")),
        },
        "same_seed_calibration_sbc_statistics": statistics,
        "statistics_authorization": {
            "path": _repo_relative(
                root,
                statistics_authorization_path,
                prefix=("configs", "execution"),
                name="calibration/SBC statistics authorization",
            ),
            "sha256": _sha256(statistics_authorization_path),
        },
        "immutable_inference": immutable,
        "score_outputs": score_outputs,
        "score_output_root": str(output_root),
        "inference_contract": {
            "posterior_draws_per_case": POSTERIOR_DRAW_COUNT,
            "maximum_draw_microbatch": MAXIMUM_DRAW_MICROBATCH,
            "physical_batch_size": 16,
            "draw_microbatch": 256,
            "model_seeds": list(MODEL_SEEDS),
            "namespace_count": FINAL_NAMESPACE_COUNT,
            "score_artifact_count": SCORE_ARTIFACT_COUNT,
            "posterior_draws_persisted": False,
        },
        "frozen_contracts": {
            "final_analysis_hash": FINAL_ANALYSIS_HASH,
            "reference_baseline_hash": REFERENCE_CONFIG_HASH,
            "final_generation_commitment_sha256": FINAL_COMMITMENT_SHA256,
        },
        "review_scope": {
            "locked_training_rung": TRAIN_131K_COUNT,
            "selected_architecture_id": checkpoints["architecture_id"],
            "model_seeds": list(MODEL_SEEDS),
            "namespace_count": FINAL_NAMESPACE_COUNT,
            "accepted_case_count": FINAL_CASE_COUNT,
            "score_artifact_count": SCORE_ARTIFACT_COUNT,
            "final_evaluation_unsealing_authorized": True,
            "final_evaluation_data_access_authorized": True,
            "selected_checkpoint_inference_authorized": True,
            "same_seed_calibration_map_application_authorized": True,
            "immutable_score_artifact_creation_authorized": True,
        },
        "closed_boundaries": {key: False for key in REVIEW_CLOSED_BOUNDARIES},
        "future_authorization_path": (
            "configs/execution/phase7_final_inference_authorization.yaml"
        ),
        "future_review_path": "results/phase7/final_inference_review.json",
        "release_packet_repository_path": output_relative,
    }


def build_final_inference_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact packet after a separate delegated review."""

    root = root.resolve()
    load_final_inference_release_stack_contract(root)
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("final_inference_authorized") is not False
        or review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by") != "codex_as_delegated_scientific_and_engineering_reviewer"
        or set(scope) != set(REVIEW_SCOPE)
        or any(scope.get(key) != packet["review_scope"][key] for key in REVIEW_SCOPE)
        or set(closed) != set(REVIEW_CLOSED_BOUNDARIES)
        or any(closed.get(key) is not False for key in REVIEW_CLOSED_BOUNDARIES)
    ):
        raise TrainingGateError("delegated final-inference review is not exact")
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="final-inference authorization",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="final-inference delegated review",
    )
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="final-inference release packet",
    )
    if (
        output_relative != packet.get("future_authorization_path")
        or review_relative != packet.get("future_review_path")
        or release_relative != packet.get("release_packet_repository_path")
    ):
        raise TrainingGateError("final-inference evidence path changed")
    return {
        "phase": "7-final-evaluation-inference",
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
            "final_evaluation_unsealing_authorized": True,
            "final_evaluation_data_access_authorized": True,
            "selected_checkpoint_inference_authorized": True,
            "same_seed_calibration_map_application_authorized": True,
            "immutable_score_artifact_creation_authorized": True,
            **{key: False for key in REVIEW_CLOSED_BOUNDARIES},
        },
        "frozen_contracts": dict(packet["frozen_contracts"]),
        "selected_architecture": dict(packet["selected_architecture"]),
        "selected_seed_checkpoints": dict(packet["selected_seed_checkpoints"]),
        "same_seed_calibration_sbc_statistics": dict(
            packet["same_seed_calibration_sbc_statistics"]
        ),
        "sealed_publication": dict(packet["sealed_publication"]),
        "immutable_inference": dict(packet["immutable_inference"]),
        "score_outputs": dict(packet["score_outputs"]),
        "inference_contract": dict(packet["inference_contract"]),
        "post_freeze_allowed_paths": [
            output_relative,
            release_relative,
            review_relative,
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE7_FINAL_EVALUATION_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase7/final_inference_summary.json",
        ],
        "stop_after_immutable_score_artifacts": True,
    }
