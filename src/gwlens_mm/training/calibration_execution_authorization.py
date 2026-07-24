"""Exact post-lock release control for calibration/SBC score and statistic execution.

The builders in this module are intentionally separated from the executable
inference/statistics runners.  A release packet is non-authorizing; only a
second, hash-bound delegated review may create an execution authorization.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import load_yaml
from ..provenance import canonical_json, configuration_hash
from .architecture import PROBE_ARCHITECTURE_ID, selected_model_configuration
from .calibration import SBC_STATISTICS
from .contracts import TrainingGateError, model_configuration_hash
from .terminal131 import TRAIN_131K_COUNT
from .terminal_architecture_authorization import (
    AUTHORIZATION_STATUS as ARCHITECTURE_AUTHORIZATION_STATUS,
)

PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
SCORE_CONFIG_PATH = "configs/inference/phase6_calibration_sbc_scores.yaml"
SCORE_CONFIG_HASH = "47df45922b8db62970e5b0a7c8315c14d95b5fc1ac7e97b030a975fe31d4f2d8"
MATERIALIZATION_AUTHORIZATION_STATUS = (
    "authorized_exact_calibration_sbc_materialization_only"
)
SCORE_RELEASE_STATUS = "ready_for_delegated_calibration_sbc_score_inference_review"
SCORE_REVIEW_STATUS = "delegated_calibration_sbc_score_inference_review_approved"
SCORE_AUTHORIZATION_STATUS = "authorized_calibration_sbc_checkpoint_inference_only"
STATISTICS_RELEASE_STATUS = "ready_for_delegated_calibration_sbc_statistics_review"
STATISTICS_REVIEW_STATUS = "delegated_calibration_sbc_statistics_review_approved"
STATISTICS_AUTHORIZATION_STATUS = "authorized_calibration_sbc_statistics_only"
SEEDS = (0, 1, 2)
SPLITS = ("calibration_fit", "sbc_diagnostic")

SCORE_REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "model_seeds",
    "score_artifact_count",
    "calibration_fit_data_access_authorized",
    "sbc_diagnostic_data_access_authorized",
    "selected_checkpoint_inference_authorized",
    "score_artifact_creation_authorized",
)
SCORE_CLOSED_BOUNDARIES = (
    "calibration_map_fitting_authorized",
    "sbc_statistical_test_authorized",
    "model_retraining_or_tuning_authorized",
    "final_evaluation_authorized",
    "gwosc_gwtc_access_authorized",
)
STATISTICS_REVIEW_SCOPE = (
    "locked_training_rung",
    "selected_architecture_id",
    "model_seeds",
    "calibration_map_count",
    "independent_sbc_result_count",
    "calibration_fit_authorized",
    "sbc_execution_authorized",
)
STATISTICS_CLOSED_BOUNDARIES = (
    "checkpoint_access_authorized",
    "model_retraining_or_tuning_authorized",
    "final_evaluation_authorized",
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


def _hex_digest(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a SHA-256 digest")
    return result


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
    if head != commit or dirty:
        raise TrainingGateError("calibration/SBC checkout is not exact and clean")


def load_score_contract(root: Path) -> Mapping[str, Any]:
    contract = load_yaml(root / SCORE_CONFIG_PATH)
    if (
        configuration_hash(contract) != SCORE_CONFIG_HASH
        or int(contract.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or contract.get("model_seeds") != list(SEEDS)
        or contract.get("best_seed_selection_authorized") is not False
        or int(contract.get("calibration_posterior_draws_per_case", -1)) != 4096
        or int(contract.get("sbc_posterior_draws_per_replicate", -1)) != 1024
        or int(contract.get("sbc_subset_seed", -1)) != 2026071601
        or int(contract.get("posterior_draw_chunk_size", -1)) != 256
        or int(contract.get("physical_batch_size", -1)) != 32
        or any(
            value is not False
            for value in contract.get("closed_boundaries", {}).values()
        )
    ):
        raise TrainingGateError("calibration/SBC score-inference contract changed")
    seed_maps = contract.get("root_seed_by_split_and_model_seed", {})
    if set(seed_maps) != set(SPLITS):
        raise TrainingGateError("score-inference split seed domains changed")
    flattened: list[int] = []
    for split in SPLITS:
        values = seed_maps.get(split, {})
        if set(values) != {str(seed) for seed in SEEDS}:
            raise TrainingGateError("score-inference model seed domains changed")
        flattened.extend(int(values[str(seed)]) for seed in SEEDS)
    if len(set(flattened)) != 6 or any(value < 0 for value in flattened):
        raise TrainingGateError("score-inference seed domains collide")
    return contract


def _immutable_inference(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="inference wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="inference exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path, name="inference environment lock", require_file=True
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
        or result.get("scientific_data_opened") is not False
        or result.get("optimizer_started") is not False
    ):
        raise TrainingGateError("calibration/SBC exact-wheel evidence did not pass")
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


def _selected_checkpoint_artifacts(
    root: Path,
    *,
    architecture_authorization: Mapping[str, Any],
    architecture_decision: Mapping[str, Any],
) -> Mapping[str, Any]:
    if architecture_authorization.get("authorization_status") != (
        ARCHITECTURE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("terminal architecture authorization changed")
    architecture_id = str(architecture_decision.get("selected_architecture_id", ""))
    model = selected_model_configuration(root, architecture_id)
    expected_model_hash = model_configuration_hash(model)
    if int(architecture_authorization.get("locked_training_rung", -1)) != (
        TRAIN_131K_COUNT
    ):
        raise TrainingGateError("selected architecture is not locked at 131k")
    if architecture_id == PROBE_ARCHITECTURE_ID:
        base = Path(str(architecture_authorization["reused_probe_output_root"]))
        directories = {
            seed: base / "rung-131072" / f"seed-{seed}" for seed in SEEDS
        }
        statuses = {"completed_131k_probe_fit_and_development_validation"}
    else:
        base = Path(str(architecture_authorization["architecture_output_root"]))
        directories = {
            seed: base / architecture_id / f"seed-{seed}" for seed in SEEDS
        }
        statuses = {
            "completed_terminal_architecture_fit_and_development_validation"
        }
    artifacts: dict[str, Any] = {}
    shared: list[Mapping[str, Any]] = []
    for seed, directory in directories.items():
        directory = _project_path(directory, name="selected run directory")
        summary_path = directory / "run_summary.json"
        checkpoint_path = directory / "best.ckpt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise TrainingGateError("selected seed artifact is incomplete")
        summary = _load_json(summary_path)
        identity = summary.get("identity", {})
        if (
            summary.get("status") not in statuses
            or int(identity.get("seed", -1)) != seed
            or int(identity.get("training_rung_count", -1)) != TRAIN_131K_COUNT
            or identity.get("model_configuration_hash") != expected_model_hash
            or summary.get("calibration_accessed") is not False
            or summary.get("final_evaluation_accessed") is not False
        ):
            raise TrainingGateError("selected seed summary violates the lock")
        if architecture_id != PROBE_ARCHITECTURE_ID and (
            summary.get("architecture_id") != architecture_id
        ):
            raise TrainingGateError("selected seed architecture changed")
        artifacts[str(seed)] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "run_summary_path": str(summary_path.resolve()),
            "run_summary_sha256": _sha256(summary_path),
        }
        shared.append(identity)
    for name in (
        "training_rung_count",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "membership_sha256",
        "input_standardizer_sha256",
        "target_standardizer_sha256",
        "model_configuration_hash",
        "final_evaluation_commitment_sha256",
    ):
        if len({identity.get(name) for identity in shared}) != 1:
            raise TrainingGateError(f"selected checkpoints mix {name}")
    return {
        "architecture_id": architecture_id,
        "model_configuration_hash": expected_model_hash,
        "selected_seed_checkpoints": artifacts,
        "shared_identity": dict(shared[0]),
    }


def build_score_inference_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    architecture_authorization_path: Path,
    architecture_decision_path: Path,
    materialization_authorization_path: Path,
    materialization_result_path: Path,
    publication_root: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    score_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Build a non-authorizing packet for six immutable score artifacts."""

    root = root.resolve()
    implementation_commit = _full_commit(
        implementation_commit, name="score-inference implementation commit"
    )
    _verify_checkout(root, implementation_commit)
    contract = load_score_contract(root)
    architecture_authorization = load_yaml(architecture_authorization_path)
    architecture_decision_path = _project_path(
        architecture_decision_path,
        name="architecture decision",
        require_file=True,
    )
    if architecture_decision_path != Path(
        str(architecture_authorization.get("architecture_selection_output_path", ""))
    ).resolve():
        raise TrainingGateError("architecture decision path changed")
    architecture_decision = _load_json(architecture_decision_path)
    materialization_authorization = load_yaml(materialization_authorization_path)
    if materialization_authorization.get("authorization_status") != (
        MATERIALIZATION_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("calibration/SBC materialization was not authorized")
    selected = materialization_authorization.get("architecture_decision", {})
    if (
        selected.get("locked_training_rung") != TRAIN_131K_COUNT
        or selected.get("selected_architecture_id")
        != architecture_decision.get("selected_architecture_id")
        or selected.get("sha256") != _sha256(architecture_decision_path)
    ):
        raise TrainingGateError("materialization and architecture decisions differ")
    checkpoints = _selected_checkpoint_artifacts(
        root,
        architecture_authorization=architecture_authorization,
        architecture_decision=architecture_decision,
    )
    materialization_result_path = _project_path(
        materialization_result_path,
        name="materialization result",
        require_file=True,
    )
    result = _load_json(materialization_result_path)
    publication_root = _project_path(
        publication_root, name="calibration/SBC publication"
    )
    manifest_path = publication_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise TrainingGateError("calibration/SBC parent manifest is absent")
    manifest = _load_json(manifest_path)
    prospective = materialization_authorization.get(
        "prospective_official_identities", {}
    )
    if (
        result.get("status") != "passed"
        or Path(str(result.get("publication_path", ""))).resolve()
        != publication_root
        or int(result.get("accepted_pair_count", -1)) != 6144
        or int(result.get("complete_shard_count", -1)) != 48
        or result.get("calibration_fit_statistics_authorized") is not False
        or result.get("sbc_statistics_authorized") is not False
        or result.get("final_evaluation_authorized") is not False
        or result.get("gwosc_gwtc_accessed") is not False
        or manifest.get("status") != "passed"
        or int(manifest.get("calibration_fit_accepted_count", -1)) != 4096
        or int(manifest.get("sbc_diagnostic_accepted_count", -1)) != 2048
        or int(manifest.get("accepted_pair_count", -1)) != 6144
        or int(manifest.get("complete_shard_count", -1)) != 48
        or manifest.get("group_disjoint_from_train_validation_and_each_other")
        is not True
        or manifest.get("calibration_fit_statistics_authorized") is not False
        or manifest.get("sbc_statistics_authorized") is not False
        or manifest.get("checkpoint_access_authorized") is not False
        or manifest.get("final_evaluation_authorized") is not False
    ):
        raise TrainingGateError("calibration/SBC publication is not a closed exact result")
    if (
        result.get("parent_run_id") != prospective.get("parent_run_id")
        or result.get("calibration_dataset_id")
        != prospective.get("calibration_dataset_id")
        or result.get("sbc_dataset_id") != prospective.get("sbc_dataset_id")
    ):
        raise TrainingGateError("calibration/SBC publication identity changed")
    output_root = _project_path(score_output_root, name="score output root")
    if output_root.exists():
        raise TrainingGateError("score output identity already exists")
    score_outputs = {
        str(seed): {
            split: str(output_root / f"seed-{seed}" / f"{split}_scores.npz")
            for split in SPLITS
        }
        for seed in SEEDS
    }
    immutable = _immutable_inference(
        implementation_commit=implementation_commit,
        wheel_path=wheel_path,
        exact_wheel_test_result_path=exact_wheel_test_result_path,
        environment_lock_path=environment_lock_path,
    )
    return {
        "status": SCORE_RELEASE_STATUS,
        "authorization_created": False,
        "score_inference_authorized": False,
        "implementation_commit": implementation_commit,
        "score_contract": {
            "path": SCORE_CONFIG_PATH,
            "canonical_hash": configuration_hash(contract),
            **dict(contract),
        },
        "architecture_authorization": {
            "path": _repo_relative(
                root,
                architecture_authorization_path,
                prefix=("configs", "execution"),
                name="architecture authorization",
            ),
            "sha256": _sha256(architecture_authorization_path),
        },
        "selected_architecture": {
            "decision_path": str(architecture_decision_path),
            "decision_sha256": _sha256(architecture_decision_path),
            "architecture_id": checkpoints["architecture_id"],
            "model_configuration_hash": checkpoints["model_configuration_hash"],
            "locked_training_rung": TRAIN_131K_COUNT,
        },
        "selected_seed_checkpoints": checkpoints["selected_seed_checkpoints"],
        "selected_checkpoint_shared_identity": checkpoints["shared_identity"],
        "materialization_authorization": {
            "path": _repo_relative(
                root,
                materialization_authorization_path,
                prefix=("configs", "execution"),
                name="materialization authorization",
            ),
            "sha256": _sha256(materialization_authorization_path),
        },
        "materialization_result": {
            "path": str(materialization_result_path),
            "sha256": _sha256(materialization_result_path),
        },
        "development_publication": {
            "parent_root": str(publication_root),
            "parent_manifest_sha256": _sha256(manifest_path),
            "publication_tree_sha256": _hex_digest(
                result.get("publication_tree_sha256"),
                name="calibration/SBC publication tree",
            ),
            "calibration_dataset_id": result["calibration_dataset_id"],
            "sbc_dataset_id": result["sbc_dataset_id"],
            "calibration_fit_accepted_count": 4096,
            "sbc_diagnostic_accepted_count": 2048,
        },
        "immutable_inference": immutable,
        "score_outputs": score_outputs,
        "score_output_root": str(output_root),
        "closed_boundaries": {
            key: False for key in SCORE_CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/phase6_calibration_sbc_score_inference_authorization.yaml"
        ),
        "future_review_path": (
            "results/phase6/calibration_sbc_score_inference_review.json"
        ),
        "release_packet_repository_path": _repo_relative(
            root,
            output_path,
            prefix=("results", "phase6"),
            name="score release packet",
        ),
    }


def build_score_inference_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one hash-reviewed score packet into a six-artifact authorization."""

    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    selected = packet.get("selected_architecture", {})
    if (
        packet.get("status") != SCORE_RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("score_inference_authorized") is not False
        or review.get("status") != SCORE_REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or not str(review.get("reviewed_by", ""))
        or set(scope) != set(SCORE_REVIEW_SCOPE)
        or int(scope.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or scope.get("selected_architecture_id") != selected.get("architecture_id")
        or scope.get("model_seeds") != list(SEEDS)
        or int(scope.get("score_artifact_count", -1)) != 6
        or any(scope.get(key) is not True for key in SCORE_REVIEW_SCOPE[4:])
        or set(closed) != set(SCORE_CLOSED_BOUNDARIES)
        or any(closed.get(key) is not False for key in SCORE_CLOSED_BOUNDARIES)
    ):
        raise TrainingGateError("delegated score-inference review is not exact")
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="score authorization",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase6"),
        name="score delegated review",
    )
    if (
        output_relative != packet.get("future_authorization_path")
        or review_relative != packet.get("future_review_path")
    ):
        raise TrainingGateError("score authorization or review path changed")
    contract = dict(packet["score_contract"])
    contract.pop("path")
    contract.pop("canonical_hash")
    return {
        "phase": "6-calibration-sbc-score-inference",
        "authorization_status": SCORE_AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": _repo_relative(
                root,
                release_packet_path,
                prefix=("results", "phase6"),
                name="score release packet",
            ),
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "authorization": {
            "selected_checkpoint_inference_authorized": True,
            "calibration_fit_data_access_authorized": True,
            "sbc_diagnostic_data_access_authorized": True,
            "score_artifact_creation_authorized": True,
            **{key: False for key in SCORE_CLOSED_BOUNDARIES},
        },
        "selected_architecture": dict(selected),
        "selected_seed_checkpoints": dict(packet["selected_seed_checkpoints"]),
        "development_publication": dict(packet["development_publication"]),
        "score_outputs": dict(packet["score_outputs"]),
        "inference_contract": contract,
        "immutable_inference": dict(packet["immutable_inference"]),
        "post_freeze_allowed_paths": [
            output_relative,
            packet["release_packet_repository_path"],
            packet["future_review_path"],
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE6_CALIBRATION_SBC_EXECUTION_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase6/calibration_sbc_score_inference_summary.json",
        ],
        "stop_after_score_artifacts": True,
    }


def _score_scalar(archive: Mapping[str, np.ndarray], name: str) -> str:
    value = archive.get(name)
    if value is None or value.shape != ():
        raise TrainingGateError(f"score artifact {name} is not scalar")
    return str(value.item())


def _load_score(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def build_statistics_release_packet(
    root: Path,
    *,
    score_authorization_path: Path,
    statistics_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Bind all six completed scores before authorizing three seedwise analyses."""

    authorization = load_yaml(score_authorization_path)
    if authorization.get("authorization_status") != SCORE_AUTHORIZATION_STATUS:
        raise TrainingGateError("score-inference authorization changed")
    outputs = authorization.get("score_outputs", {})
    selected = authorization.get("selected_architecture", {})
    checkpoints = authorization.get("selected_seed_checkpoints", {})
    score_artifacts: dict[str, Any] = {}
    score_identities: dict[str, Any] = {}
    shared_ids: dict[str, str] = {}
    for seed in SEEDS:
        seed_key = str(seed)
        seed_outputs = outputs.get(seed_key, {})
        seed_artifacts: dict[str, Any] = {}
        seed_ids: dict[str, set[str]] = {}
        seed_identity: dict[str, str] | None = None
        for split in SPLITS:
            path = _project_path(
                Path(str(seed_outputs.get(split, ""))),
                name=f"{split} seed-{seed} score",
                require_file=True,
            )
            summary_path = path.with_suffix(".summary.json")
            if not summary_path.is_file():
                raise TrainingGateError("score summary is absent")
            summary = _load_json(summary_path)
            score = _load_score(path)
            expected_count = 4096 if split == "calibration_fit" else 1024
            expected_draws = 4096 if split == "calibration_fit" else 1024
            identifiers = tuple(str(value) for value in score["physical_system_ids"])
            if (
                summary.get("status") != "completed_score_extraction_only"
                or summary.get("split") != split
                or int(summary.get("model_seed", -1)) != seed
                or summary.get("architecture_id") != selected.get("architecture_id")
                or int(summary.get("case_count", -1)) != expected_count
                or int(summary.get("posterior_draw_count", -1)) != expected_draws
                or summary.get("score_artifact_sha256") != _sha256(path)
                or summary.get("calibration_map_fitted") is not False
                or summary.get("sbc_statistical_test_executed") is not False
                or summary.get("model_retrained_or_tuned") is not False
                or summary.get("final_evaluation_accessed") is not False
                or score.get("marginal_scores", np.empty((0, 0))).shape
                != (expected_count, 2)
                or score.get("joint_scores", np.empty(0)).shape != (expected_count,)
                or len(identifiers) != expected_count
                or len(set(identifiers)) != expected_count
                or _score_scalar(score, "split") != split
                or int(_score_scalar(score, "model_seed")) != seed
                or _score_scalar(score, "architecture_id")
                != selected.get("architecture_id")
                or _score_scalar(score, "checkpoint_sha256")
                != checkpoints[seed_key]["sha256"]
            ):
                raise TrainingGateError("score artifact violates its exact contract")
            if split == "sbc_diagnostic" and any(
                score.get(f"rank_{statistic}", np.empty(0)).shape
                != (expected_count,)
                for statistic in SBC_STATISTICS
            ):
                raise TrainingGateError("SBC score artifact lacks frozen rank statistics")
            seed_ids[split] = set(identifiers)
            identifier_hash = hashlib.sha256(
                canonical_json(sorted(identifiers)).encode()
            ).hexdigest()
            if summary.get("physical_system_ids_sha256") != identifier_hash:
                raise TrainingGateError("score physical-system identity hash changed")
            if split in shared_ids and shared_ids[split] != identifier_hash:
                raise TrainingGateError("model seeds scored different development cases")
            shared_ids[split] = identifier_hash
            seed_artifacts[split] = {
                "path": str(path),
                "sha256": _sha256(path),
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
            }
            observed_identity = {
                "model_seed": _score_scalar(score, "model_seed"),
                "architecture_id": _score_scalar(score, "architecture_id"),
                "checkpoint_sha256": _score_scalar(score, "checkpoint_sha256"),
                "publication_manifest_sha256": _score_scalar(
                    score, "publication_manifest_sha256"
                ),
                "inference_commit": _score_scalar(score, "inference_commit"),
            }
            if seed_identity is not None and seed_identity != observed_identity:
                raise TrainingGateError("calibration/SBC scores mix model identity")
            seed_identity = observed_identity
        if seed_ids["calibration_fit"] & seed_ids["sbc_diagnostic"]:
            raise TrainingGateError("calibration-fit and SBC score IDs overlap")
        if seed_identity is None:
            raise TrainingGateError("score identity is absent")
        score_artifacts[seed_key] = seed_artifacts
        score_identities[seed_key] = seed_identity
    output_root = _project_path(
        statistics_output_root, name="calibration/SBC statistics output root"
    )
    if output_root.exists():
        raise TrainingGateError("statistics output identity already exists")
    return {
        "status": STATISTICS_RELEASE_STATUS,
        "authorization_created": False,
        "statistics_execution_authorized": False,
        "score_authorization": {
            "path": _repo_relative(
                root,
                score_authorization_path,
                prefix=("configs", "execution"),
                name="score authorization",
            ),
            "sha256": _sha256(score_authorization_path),
        },
        "selected_architecture": dict(selected),
        "score_artifacts_by_seed": score_artifacts,
        "score_identities_by_seed": score_identities,
        "shared_physical_system_ids_sha256": shared_ids,
        "statistics_output_roots": {
            str(seed): str(output_root / f"seed-{seed}") for seed in SEEDS
        },
        "statistics_output_root": str(output_root),
        "closed_boundaries": {
            key: False for key in STATISTICS_CLOSED_BOUNDARIES
        },
        "future_authorization_path": (
            "configs/execution/phase6_calibration_sbc_statistics_authorization.yaml"
        ),
        "future_review_path": "results/phase6/calibration_sbc_statistics_review.json",
        "release_packet_repository_path": _repo_relative(
            root,
            output_path,
            prefix=("results", "phase6"),
            name="statistics release packet",
        ),
    }


def build_statistics_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote all three seedwise statistic jobs after one exact delegated review."""

    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    selected = packet.get("selected_architecture", {})
    if (
        packet.get("status") != STATISTICS_RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("statistics_execution_authorized") is not False
        or review.get("status") != STATISTICS_REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or not str(review.get("reviewed_by", ""))
        or set(scope) != set(STATISTICS_REVIEW_SCOPE)
        or int(scope.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or scope.get("selected_architecture_id") != selected.get("architecture_id")
        or scope.get("model_seeds") != list(SEEDS)
        or int(scope.get("calibration_map_count", -1)) != 3
        or int(scope.get("independent_sbc_result_count", -1)) != 3
        or scope.get("calibration_fit_authorized") is not True
        or scope.get("sbc_execution_authorized") is not True
        or set(closed) != set(STATISTICS_CLOSED_BOUNDARIES)
        or any(
            closed.get(key) is not False for key in STATISTICS_CLOSED_BOUNDARIES
        )
    ):
        raise TrainingGateError("delegated calibration/SBC statistics review is not exact")
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="statistics authorization",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase6"),
        name="statistics delegated review",
    )
    if (
        output_relative != packet.get("future_authorization_path")
        or review_relative != packet.get("future_review_path")
    ):
        raise TrainingGateError("statistics authorization or review path changed")
    return {
        "phase": "6-calibration-sbc-statistics",
        "authorization_status": STATISTICS_AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": _repo_relative(
                root,
                release_packet_path,
                prefix=("results", "phase6"),
                name="statistics release packet",
            ),
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "authorization": {
            "calibration_fit_authorized": True,
            "sbc_execution_authorized": True,
            **{key: False for key in STATISTICS_CLOSED_BOUNDARIES},
        },
        "selected_architecture": dict(selected),
        "authorized_model_seeds": list(SEEDS),
        "score_artifacts_by_seed": dict(packet["score_artifacts_by_seed"]),
        "score_identities_by_seed": dict(packet["score_identities_by_seed"]),
        "statistics_output_roots": dict(packet["statistics_output_roots"]),
        "post_freeze_allowed_paths": [
            output_relative,
            packet["release_packet_repository_path"],
            packet["future_review_path"],
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE6_CALIBRATION_SBC_EXECUTION_REPORT.md",
            "results/experiment_registry.csv",
            "results/phase6/calibration_sbc_seed_aggregate.json",
        ],
        "stop_after_seedwise_calibration_and_sbc": True,
    }
