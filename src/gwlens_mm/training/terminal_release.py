"""Fail-closed review packet for a future terminal-131k probe authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import load_yaml
from ..provenance import configuration_hash
from .contracts import TrainingGateError, model_configuration_hash
from .terminal131 import TAIL_COUNT, TRAIN_131K_COUNT, TRAIN_INCREMENT_COUNT

TERMINAL_PREREGISTRATION_PATH = (
    "configs/statistics/terminal_131k_preregistration.yaml"
)
TERMINAL_PREREGISTRATION_HASH = (
    "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
)
MODEL_CONFIGURATION_PATH = "configs/models/phase4_probe_nsf.yaml"
MODEL_CONFIGURATION_HASH = (
    "8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087"
)
ENVIRONMENT_LOCK_HASH = (
    "2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95"
)
FINAL_COMMITMENT_PATH = "results/phase4/final_evaluation_commitment.json"
FINAL_COMMITMENT_HASH = (
    "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
)
EXPECTED_GPU_MODEL = "NVIDIA RTX 5000 Ada Generation"
MINIMUM_GPU_COUNT = 3
RETAINED_65K_COUNT = 65536
RETAINED_65K_STATUS = "completed_65k_probe_fit_and_development_validation"
RELEASE_POST_FREEZE_ALLOWED_PATHS = frozenset(
    {
        "AGENTS.md",
        "docs/DECISIONS.md",
        "docs/FAILURES.md",
        "docs/PROJECT_STATE.md",
        "docs/reports/PHASE4_TERMINAL_131K_CLOSEOUT_REPORT.md",
        "results/experiment_registry.csv",
        "results/phase4/terminal_131k_execution_result.json",
        "results/phase4/terminal_probe_closeout.json",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingGateError(f"expected a JSON mapping: {path}")
    return value


def _retained_65k_probe_binding(
    retained_output_root: Path,
    *,
    expected_model_hash: str,
    expected_train_manifest_hash: str,
    expected_environment_hash: str,
    expected_commitment_hash: str,
) -> Mapping[str, Any]:
    """Hash and structurally validate all three retained 65k probe artifacts."""

    output_root = retained_output_root.resolve()
    if not output_root.is_absolute():
        raise TrainingGateError("retained 65k probe root must be absolute")
    artifacts: dict[str, Mapping[str, Any]] = {}
    shared_identity: dict[str, str] | None = None
    for seed in (0, 1, 2):
        run_root = output_root / "rung-65536" / f"seed-{seed}"
        summary_path = run_root / "run_summary.json"
        checkpoint_path = run_root / "best.ckpt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise TrainingGateError("retained 65k probe artifact is absent")
        summary = _load_json(summary_path)
        identity = summary.get("identity")
        development = summary.get("development")
        if not isinstance(identity, Mapping) or not isinstance(development, Mapping):
            raise TrainingGateError("retained 65k probe summary is malformed")
        if (
            summary.get("status") != RETAINED_65K_STATUS
            or int(identity.get("training_rung_count", -1)) != RETAINED_65K_COUNT
            or int(identity.get("seed", -1)) != seed
            or identity.get("model_configuration_hash") != expected_model_hash
            or identity.get("train_manifest_sha256")
            != expected_train_manifest_hash
            or identity.get("training_environment_sha256")
            != expected_environment_hash
            or identity.get("final_evaluation_commitment_sha256")
            != expected_commitment_hash
            or development.get("status") != "completed_development_validation"
            or int(development.get("case_count", -1)) != 6144
            or summary.get("architecture_selection_authorized") is not False
            or summary.get("calibration_accessed") is not False
            or summary.get("final_evaluation_accessed") is not False
        ):
            raise TrainingGateError("retained 65k probe summary violates its contract")
        observed_shared = {
            key: str(identity.get(key, ""))
            for key in (
                "model_configuration_hash",
                "training_code_commit",
                "training_environment_sha256",
                "train_manifest_sha256",
                "validation_manifest_sha256",
                "final_evaluation_commitment_sha256",
                "membership_sha256",
                "input_standardizer_sha256",
                "target_standardizer_sha256",
            )
        }
        if len(observed_shared["training_code_commit"]) != 40 or any(
            len(value) != 64
            for key, value in observed_shared.items()
            if key != "training_code_commit"
        ):
            raise TrainingGateError("retained 65k probe identity hash is invalid")
        if shared_identity is None:
            shared_identity = observed_shared
        elif shared_identity != observed_shared:
            raise TrainingGateError("retained 65k probe fits do not share one identity")
        artifacts[str(seed)] = {
            "run_summary_sha256": _sha256(summary_path),
            "best_checkpoint_sha256": _sha256(checkpoint_path),
        }
    assert shared_identity is not None
    return {
        "output_root": str(output_root),
        "training_rung_count": RETAINED_65K_COUNT,
        "shared_identity": shared_identity,
        "artifacts": artifacts,
    }


def _validate_closeout(closeout: Mapping[str, Any]) -> Mapping[str, Any]:
    tree = closeout.get("tree_evidence")
    if not isinstance(tree, Mapping) or tree.get("recomputed") is not True:
        raise TrainingGateError("terminal probe review lacks independent tree replay")
    if (
        closeout.get("status"),
        int(closeout.get("new_train_accepted_count", -1)),
        int(closeout.get("new_train_shard_count", -1)),
        int(closeout.get("development_tail_accepted_count", -1)),
        int(closeout.get("development_tail_namespace_count", -1)),
        int(closeout.get("logical_train_accepted_count", -1)),
        closeout.get("proposal_equals_evaluation"),
        closeout.get("all_importance_weights_one"),
        closeout.get("training_authorized"),
        closeout.get("architecture_selection_authorized"),
        closeout.get("calibration_authorized"),
        closeout.get("sbc_authorized"),
        closeout.get("final_evaluation_authorized"),
        closeout.get("extension_above_131072_authorized"),
        closeout.get("gwosc_gwtc_accessed"),
    ) != (
        "terminal_131k_independent_closeout_passed",
        TRAIN_INCREMENT_COUNT,
        512,
        TAIL_COUNT,
        4,
        TRAIN_131K_COUNT,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ):
        raise TrainingGateError("terminal probe review closeout contract failed")
    for field in (
        "combined_manifest_sha256",
        "train_parent_manifest_sha256",
        "development_tail_manifest_sha256",
    ):
        value = str(closeout.get(field, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise TrainingGateError(f"terminal probe review has invalid {field}")
    return closeout


def validate_terminal_release_checkout_paths(changed_paths: Sequence[str]) -> None:
    """Allow post-freeze publication evidence but reject every software drift."""

    normalized = tuple(str(Path(value)) for value in changed_paths)
    unexpected = sorted(set(normalized) - RELEASE_POST_FREEZE_ALLOWED_PATHS)
    if unexpected:
        raise TrainingGateError(
            "terminal release checkout changed frozen software: "
            + ", ".join(unexpected)
        )


def prepare_terminal_probe_review_packet(
    root: Path,
    *,
    closeout_result_path: Path,
    training_commit: str,
    review_checkout_commit: str,
    wheel_path: Path,
    environment_lock_path: Path,
    wheel_test_result_path: Path,
    gpu_names: Sequence[str],
    retained_65k_output_root: Path,
) -> Mapping[str, Any]:
    """Bind every exact pre-authorization identity without opening data."""

    root_resolved = root.resolve()
    closeout_path = closeout_result_path.resolve()
    try:
        closeout_relative = closeout_path.relative_to(root_resolved)
    except ValueError as error:
        raise TrainingGateError(
            "terminal closeout evidence must remain inside repository"
        ) from error
    if closeout_relative.parts[:2] != ("results", "phase4"):
        raise TrainingGateError(
            "terminal closeout evidence must remain under results/phase4"
        )
    closeout = _validate_closeout(_load_json(closeout_path))
    if any(
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
        for commit in (training_commit, review_checkout_commit)
    ):
        raise TrainingGateError("terminal probe review training commit is invalid")
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise TrainingGateError("terminal probe review wheel is absent")
    wheel_hash = _sha256(wheel_path)
    if not environment_lock_path.is_file() or _sha256(
        environment_lock_path
    ) != ENVIRONMENT_LOCK_HASH:
        raise TrainingGateError("terminal probe review environment lock changed")

    wheel_test = _load_json(wheel_test_result_path)
    if not (
        wheel_test.get("status") == "passed_exact_wheel_on_autodl"
        and wheel_test.get("wheel_sha256") == wheel_hash
        and int(wheel_test.get("focused_test_exit_code", -1)) == 0
        and int(wheel_test.get("full_test_exit_code", -1)) == 0
        and wheel_test.get("torch_cuda_available") is True
        and wheel_test.get("editable_install_used") is False
        and wheel_test.get("wheel_import_verified") is True
        and wheel_test.get("installed_distribution_name") == "gwlens-mm"
        and wheel_test.get("installed_module_from_repository_source") is False
        and wheel_test.get("repository_root_pythonpath_used") is True
        and wheel_test.get("repository_src_pythonpath_used") is False
    ):
        raise TrainingGateError("terminal probe review exact-wheel test failed")
    observed_gpu_names = tuple(str(name) for name in gpu_names)
    tested_gpu_names = tuple(str(name) for name in wheel_test.get("gpu_names", ()))
    if (
        len(observed_gpu_names) < MINIMUM_GPU_COUNT
        or observed_gpu_names != tested_gpu_names
        or any(name != EXPECTED_GPU_MODEL for name in observed_gpu_names)
    ):
        raise TrainingGateError("terminal probe review GPU identity changed")

    preregistration = load_yaml(root / TERMINAL_PREREGISTRATION_PATH)
    model = load_yaml(root / MODEL_CONFIGURATION_PATH)
    commitment_path = root / FINAL_COMMITMENT_PATH
    commitment = _load_json(commitment_path)
    if (
        configuration_hash(preregistration) != TERMINAL_PREREGISTRATION_HASH
        or model_configuration_hash(model) != MODEL_CONFIGURATION_HASH
        or _sha256(commitment_path) != FINAL_COMMITMENT_HASH
        or commitment.get("commitment_status") != "finalized_before_training"
    ):
        raise TrainingGateError("terminal probe review frozen scientific input changed")
    retained_65k_probe = _retained_65k_probe_binding(
        retained_65k_output_root,
        expected_model_hash=MODEL_CONFIGURATION_HASH,
        expected_train_manifest_hash=str(
            preregistration["terminal_training_ladder"]["corrected_train_65k"][
                "manifest_sha256"
            ]
        ),
        expected_environment_hash=ENVIRONMENT_LOCK_HASH,
        expected_commitment_hash=FINAL_COMMITMENT_HASH,
    )
    return {
        "status": "ready_for_delegated_terminal_probe_authorization_review",
        "authorization_created": False,
        "optimizer_execution_authorized": False,
        "release_review_checkout_commit": review_checkout_commit,
        "closeout_result_path": str(closeout_relative),
        "closeout_result_sha256": _sha256(closeout_path),
        "terminal_preregistration": {
            "path": TERMINAL_PREREGISTRATION_PATH,
            "canonical_hash": TERMINAL_PREREGISTRATION_HASH,
        },
        "publication": {
            "combined_manifest_sha256": closeout["combined_manifest_sha256"],
            "train_parent_manifest_sha256": closeout[
                "train_parent_manifest_sha256"
            ],
            "development_tail_manifest_sha256": closeout[
                "development_tail_manifest_sha256"
            ],
            "logical_train_accepted_count": TRAIN_131K_COUNT,
            "development_tail_accepted_count": TAIL_COUNT,
        },
        "immutable_training": {
            "git_commit": training_commit,
            "wheel_path": str(wheel_path),
            "wheel_filename": wheel_path.name,
            "wheel_sha256": wheel_hash,
            "model_configuration_path": MODEL_CONFIGURATION_PATH,
            "model_configuration_hash": MODEL_CONFIGURATION_HASH,
            "environment_lock_path": str(environment_lock_path),
            "environment_lock_sha256": ENVIRONMENT_LOCK_HASH,
            "editable_install_authorized": False,
            "cuda_required": True,
            "gpu_model": EXPECTED_GPU_MODEL,
            "minimum_gpu_count": MINIMUM_GPU_COUNT,
            "observed_gpu_names": list(observed_gpu_names),
            "exact_wheel_test_result_path": str(wheel_test_result_path),
            "exact_wheel_test_result_sha256": _sha256(wheel_test_result_path),
        },
        "retained_65k_probe": retained_65k_probe,
        "final_evaluation_commitment_sha256": FINAL_COMMITMENT_HASH,
        "authorized_training_rungs_preview": [TRAIN_131K_COUNT],
        "authorized_training_seeds_preview": [0, 1, 2],
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_access_authorized": False,
    }
