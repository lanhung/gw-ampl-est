"""Exact release control for one frozen RC.7 reference query.

The packet binds one query role to atomically published dataset children,
the terminal training/architecture locks, primary preprocessing and immutable
reference software.  It is deliberately non-authorizing until a separate
delegated review creates the runtime YAML consumed by ``reference_execution``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..production.calibration_sbc import CORRECTED_COMBINED_TRAIN_MANIFEST_HASH
from ..production.final_evaluation import (
    BoundPublishedReferenceDataset,
    resolve_bound_published_reference_dataset,
)
from .architecture import selected_model_configuration
from .contracts import TrainingGateError, model_configuration_hash
from .reference_baseline import (
    NEIGHBOR_COUNT,
    POSTERIOR_DRAW_COUNT,
    REFERENCE_CONFIG_HASH,
)
from .reference_execution import (
    QUERY_COMPONENT_COUNTS,
    QUERY_SPECS,
    validate_reference_execution_stack_contract,
)
from .terminal131 import TRAIN_131K_COUNT
from .terminal_architecture_authorization import (
    AUTHORIZATION_STATUS as ARCHITECTURE_AUTHORIZATION_STATUS,
)
from .terminal_downstream import validate_hashed_terminal_decisions

STACK_AUTHORIZATION = (
    "configs/execution/phase7_reference_execution_stack_authorization.yaml"
)
RELEASE_STATUS = "ready_for_delegated_reference_query_review"
REVIEW_STATUS = "delegated_reference_query_review_approved"
AUTHORIZATION_STATUS = "authorized_reference_query_execution_only"
CATALOG_STATUS = "validated_atomic_reference_query_catalog"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")

REVIEW_FIELDS = (
    "locked_training_rung",
    "selected_architecture_id",
    "query_role",
    "query_count",
    "query_dataset_count",
    "scientific_reference_bank_access_authorized",
    "reference_query_execution_authorized",
    "validation_reference_execution_authorized",
    "final_evaluation_unsealing_authorized",
    "final_reference_execution_authorized",
)
CLOSED_BOUNDARIES = (
    "final_evaluation_materialization_authorized",
    "checkpoint_access_authorized",
    "calibration_refit_authorized",
    "model_retraining_or_tuning_authorized",
    "architecture_or_size_selection_authorized",
    "extension_above_131072_authorized",
    "likelihood_gold_claim_authorized",
    "importance_sampling_efficiency_claim_authorized",
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
        raise TrainingGateError("reference implementation commit is not full length")
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
            raise TrainingGateError("reference synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("reference checkout identity is absent")
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
        raise TrainingGateError("reference checkout is not exact and clean")


def _immutable_execution(
    *,
    implementation_commit: str,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
) -> Mapping[str, Any]:
    wheel = _project_path(wheel_path, name="reference wheel", require_file=True)
    result_path = _project_path(
        exact_wheel_test_result_path,
        name="reference exact-wheel result",
        require_file=True,
    )
    environment = _project_path(
        environment_lock_path,
        name="reference environment lock",
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
        or result.get("scientific_data_opened") is not False
        or result.get("optimizer_started") is not False
    ):
        raise TrainingGateError("reference exact-wheel evidence did not pass")
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


def validate_reference_query_catalog(
    catalog: Mapping[str, Any],
    *,
    approved_root: Path = PROJECT_ROOT,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Validate one exact query role and its atomic child/parent references."""

    role = str(catalog.get("query_role", ""))
    if (
        catalog.get("status") != CATALOG_STATUS
        or role not in QUERY_SPECS
        or catalog.get("scientific_query_opened") is not False
        or catalog.get("reference_executed") is not False
    ):
        raise TrainingGateError("reference query catalog status is invalid")
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list):
        raise TrainingGateError("reference query catalog datasets are absent")
    expected_counts = tuple(sorted(QUERY_COMPONENT_COUNTS[role]))
    observed_counts = tuple(
        sorted(int(item.get("accepted_count", -1)) for item in datasets)
    )
    if (
        len(datasets) != len(expected_counts)
        or observed_counts != expected_counts
        or sum(observed_counts) != QUERY_SPECS[role][1]
    ):
        raise TrainingGateError("reference query catalog arithmetic changed")
    resolved: list[BoundPublishedReferenceDataset] = []
    normalized: list[Mapping[str, Any]] = []
    for specification in datasets:
        if not isinstance(specification, Mapping):
            raise TrainingGateError("reference query dataset is malformed")
        try:
            reference = resolve_bound_published_reference_dataset(
                specification,
                approved_root=approved_root,
            )
        except ValueError as error:
            raise TrainingGateError(
                "reference query child is not atomically bound"
            ) from error
        resolved.append(reference)
        normalized.append(
            {
                "dataset_id": reference.dataset_id,
                "dataset_root": str(reference.dataset_root),
                "parent_root": str(reference.parent_root),
                "parent_manifest_sha256": reference.parent_manifest_sha256,
                "accepted_count": int(specification["accepted_count"]),
            }
        )
    roots = {reference.dataset_root for reference in resolved}
    if len(roots) != len(resolved):
        raise TrainingGateError("reference query dataset root is duplicated")
    return role, tuple(normalized)


def build_reference_query_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    architecture_authorization_path: Path,
    architecture_decision_path: Path,
    terminal_decision_path: Path,
    primary_rung_preparation_path: Path,
    query_catalog_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    reference_output_root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Assemble one non-authorizing validation or final reference packet."""

    root = root.resolve()
    validate_reference_execution_stack_contract(root)
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
        raise TrainingGateError("reference decision paths changed")
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
        raise TrainingGateError("reference terminal decision differs from architecture")
    architecture = decisions["architecture"]
    architecture_id = str(architecture["architecture_id"])
    primary_model = selected_model_configuration(root, architecture_id)
    primary_hash = model_configuration_hash(primary_model)
    if primary_hash != architecture["model_configuration_hash"]:
        raise TrainingGateError("reference selected primary model changed")
    expected_preparation = (
        Path(str(architecture_authorization["reused_probe_output_root"]))
        / "rung-131072"
        / "rung_preparation.json"
    ).resolve()
    preparation_path = _project_path(
        primary_rung_preparation_path,
        name="reference primary preparation",
        require_file=True,
    )
    preparation = _load_json(preparation_path)
    member_count = len(tuple(preparation.get("member_ids", ())))
    if (
        preparation_path != expected_preparation
        or _sha256(preparation_path)
        != architecture_authorization.get("reused_probe_rung_preparation_sha256")
        or preparation.get("status") != "ready_for_authorized_probe_fits"
        or int(preparation.get("rung_count", -1)) != TRAIN_131K_COUNT
        or member_count != TRAIN_131K_COUNT
    ):
        raise TrainingGateError("reference primary preparation changed")
    catalog_path = _project_path(
        query_catalog_path,
        name="reference query catalog",
        require_file=True,
    )
    role, datasets = validate_reference_query_catalog(
        _load_json(catalog_path),
        approved_root=PROJECT_ROOT,
    )
    final_role = role != "validation"
    output_root = _project_path(reference_output_root, name="reference output root")
    if output_root.exists():
        raise TrainingGateError("reference output identity already exists")
    immutable = _immutable_execution(
        implementation_commit=implementation_commit,
        wheel_path=wheel_path,
        exact_wheel_test_result_path=exact_wheel_test_result_path,
        environment_lock_path=environment_lock_path,
    )
    packet_relative = _repo_relative(
        root,
        output_path,
        prefix=("results", "phase7"),
        name="reference release packet",
    )
    expected_packet = f"results/phase7/reference_{role}_release_packet.json"
    if packet_relative != expected_packet:
        raise TrainingGateError("reference release-packet path changed")
    review_scope = {
        "locked_training_rung": TRAIN_131K_COUNT,
        "selected_architecture_id": architecture_id,
        "query_role": role,
        "query_count": QUERY_SPECS[role][1],
        "query_dataset_count": len(datasets),
        "scientific_reference_bank_access_authorized": True,
        "reference_query_execution_authorized": True,
        "validation_reference_execution_authorized": not final_role,
        "final_evaluation_unsealing_authorized": final_role,
        "final_reference_execution_authorized": final_role,
    }
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "reference_query_execution_authorized": False,
        "implementation_commit": implementation_commit,
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
        "primary_rung_preparation": {
            "path": str(preparation_path),
            "sha256": _sha256(preparation_path),
        },
        "query_catalog": {
            "path": str(catalog_path),
            "sha256": _sha256(catalog_path),
            "query_role": role,
            "datasets": list(datasets),
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
        "immutable_execution": immutable,
        "reference_output_root": str(output_root),
        "review_scope": review_scope,
        "closed_boundaries": {key: False for key in CLOSED_BOUNDARIES},
        "future_authorization_path": (
            f"configs/execution/phase7_reference_{role}_authorization.yaml"
        ),
        "future_review_path": (
            f"results/phase7/reference_{role}_delegated_review.json"
        ),
        "release_packet_repository_path": packet_relative,
    }


def build_reference_query_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one exact reference packet after delegated review."""

    root = root.resolve()
    validate_reference_execution_stack_contract(root)
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    review = _load_json(delegated_review_path)
    scope = review.get("authorization_scope", {})
    closed = review.get("closed_boundaries", {})
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("reference_query_execution_authorized") is not False
        or review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by")
        != "codex_as_delegated_scientific_and_engineering_reviewer"
        or set(scope) != set(REVIEW_FIELDS)
        or any(scope.get(key) != packet["review_scope"][key] for key in REVIEW_FIELDS)
        or set(closed) != set(CLOSED_BOUNDARIES)
        or any(closed.get(key) is not False for key in CLOSED_BOUNDARIES)
    ):
        raise TrainingGateError("delegated reference review is not exact")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="reference release packet",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="reference delegated review",
    )
    output_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="reference authorization",
    )
    if (
        release_relative != packet["release_packet_repository_path"]
        or review_relative != packet["future_review_path"]
        or output_relative != packet["future_authorization_path"]
    ):
        raise TrainingGateError("reference release evidence path changed")
    corrected = packet["corrected_65k_publication"]
    role = str(packet["query_catalog"]["query_role"])
    scope = packet["review_scope"]
    preparation = packet["primary_rung_preparation"]
    return {
        "phase": f"7-reference-{role}",
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
            "scientific_reference_bank_access_authorized": True,
            "reference_query_execution_authorized": True,
            "validation_reference_execution_authorized": scope[
                "validation_reference_execution_authorized"
            ],
            "final_evaluation_unsealing_authorized": scope[
                "final_evaluation_unsealing_authorized"
            ],
            "final_reference_execution_authorized": scope[
                "final_reference_execution_authorized"
            ],
            **{key: False for key in CLOSED_BOUNDARIES},
        },
        "locked_training_rung": TRAIN_131K_COUNT,
        "query_role": role,
        "query_count": QUERY_SPECS[role][1],
        "reference_id": "selected_prior_em_timing_knn_kde_v1",
        "neighbor_count": NEIGHBOR_COUNT,
        "posterior_draws_per_case": POSTERIOR_DRAW_COUNT,
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
        "primary_rung_preparation_path": preparation["path"],
        "primary_rung_preparation_sha256": preparation["sha256"],
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
        "query_datasets": list(packet["query_catalog"]["datasets"]),
        "immutable_execution": dict(packet["immutable_execution"]),
        "reference_output_root": packet["reference_output_root"],
        "post_freeze_allowed_paths": [
            output_relative,
            release_relative,
            review_relative,
            "AGENTS.md",
            "docs/DECISIONS.md",
            "docs/FAILURES.md",
            "docs/PROJECT_STATE.md",
            "docs/reports/PHASE7_REFERENCE_EXECUTION_REPORT.md",
            "results/experiment_registry.csv",
            f"results/phase7/reference_{role}_summary.json",
        ],
        "stop_after_reference_query": True,
        "reference_is_exact_likelihood_or_gold": False,
        "importance_sampling_efficiency_claim_authorized": False,
        "reference_config_sha256": REFERENCE_CONFIG_HASH,
    }
