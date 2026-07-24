"""Exact release control for sealed final-evaluation materialization.

This module is deliberately non-executing.  It turns already published,
hash-bound scientific parents into a structured child-dataset catalog, then
requires a second delegated review before producing the authorization consumed
by the existing final-evaluation preflight.  No official final identity is
derived here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..provenance import configuration_hash
from ..training.contracts import TrainingGateError
from ..training.terminal131 import resolve_terminal_131k_training_publication
from ..training.terminal_downstream import validate_hashed_terminal_decisions
from .calibration_sbc import (
    CORRECTED_COMBINED_TRAIN_MANIFEST_HASH,
    CORRECTION_PARENT_MANIFEST_HASH,
    CORRECTION_PUBLICATION_TREE_HASH,
)
from .final_evaluation import (
    FINAL_EVALUATION_COMMITMENT_HASH,
    FINAL_EVALUATION_CONFIG,
    NUMERICAL_VALIDITY_ADDENDUM_HASH,
    ORIGINAL_COMMITTED_GENERATOR,
    TERMINAL_PREREGISTRATION_HASH,
    load_final_evaluation_contract,
    resolve_bound_published_reference_dataset,
)
from .waveform_correction import CORRECTION_PREREGISTRATION_HASH

STACK_AUTHORIZATION = (
    "configs/execution/"
    "phase7_final_materialization_release_stack_authorization.yaml"
)
RELEASE_STATUS = "ready_for_delegated_final_materialization_review"
REVIEW_STATUS = "delegated_final_materialization_review_approved"
AUTHORIZATION_STATUS = "authorized_sealed_final_evaluation_materialization_only"
REFERENCE_CATALOG_STATUS = "validated_atomic_final_reference_catalog"
PROJECT_ROOT = Path("/root/autodl-tmp/lensing-4")
REFERENCE_COUNTS = {
    "train": (131072, 5),
    "validation": (6144, 1),
    "calibration_fit": (4096, 1),
    "sbc_diagnostic": (2048, 1),
}
TERMINAL_PROBE_AUTHORIZATION_STATUS = "authorized_terminal_131k_probe_only"
REVIEW_FIELDS = (
    "training_reference_mode",
    "locked_training_rung",
    "selected_architecture_id",
    "accepted_pair_count",
    "shard_count",
    "namespace_count",
    "sealed_materialization_authorized",
    "unsealing_authorized",
    "scientific_analysis_authorized",
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


def _digest(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a SHA-256 digest")
    return result


def _commit(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise TrainingGateError(f"{name} is not a full Git commit")
    return result


def _repo_relative(root: Path, path: Path, prefix: tuple[str, ...]) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise TrainingGateError("final release evidence escaped repository") from error
    if relative.parts[: len(prefix)] != prefix:
        raise TrainingGateError("final release evidence path is invalid")
    return str(relative)


def _project_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise TrainingGateError(f"{name} is not an AutoDL project artifact")
    return resolved


def _verify_release_checkout(root: Path, implementation_commit: str) -> None:
    marker = root / "SYNCED_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != implementation_commit:
            raise TrainingGateError("final materialization synced commit changed")
        return
    if not (root / ".git").is_dir():
        raise TrainingGateError("final materialization checkout identity is absent")
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
        raise TrainingGateError("final materialization checkout is not exact and clean")


def load_final_materialization_release_stack_contract(
    root: Path,
) -> Mapping[str, Any]:
    """Validate the implementation-only boundary."""

    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise TrainingGateError("final materialization release-stack gate is absent")
    config, _ = load_final_evaluation_contract(root)
    frozen = authorization.get("frozen_contracts", {})
    if (
        frozen.get("final_generator_configuration_hash")
        != configuration_hash(config)
        or frozen.get("final_generation_commitment_sha256")
        != FINAL_EVALUATION_COMMITMENT_HASH
        or frozen.get("numerical_validity_addendum_sha256")
        != NUMERICAL_VALIDITY_ADDENDUM_HASH
        or frozen.get("terminal_preregistration_hash")
        != TERMINAL_PREREGISTRATION_HASH
    ):
        raise TrainingGateError("final materialization parent contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "atomic_parent_child_reference_resolver_implementation_authorized",
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise TrainingGateError("final materialization implementation boundary opened")
    return authorization


def _dataset_entry(
    dataset_root: Path,
    parent_root: Path,
    parent_manifest_sha256: str,
) -> Mapping[str, str]:
    value = {
        "dataset_id": dataset_root.name,
        "dataset_root": str(dataset_root.resolve()),
        "parent_root": str(parent_root.resolve()),
        "parent_manifest_sha256": _digest(
            parent_manifest_sha256,
            name="reference parent manifest",
        ),
    }
    resolve_bound_published_reference_dataset(value, approved_root=PROJECT_ROOT)
    return value


def validate_final_reference_catalog(
    catalog: Mapping[str, Any],
    *,
    approved_root: Path = PROJECT_ROOT,
) -> Mapping[str, Any]:
    """Validate exact child roots and their atomic parent manifests."""

    if (
        catalog.get("status") != REFERENCE_CATALOG_STATUS
        or catalog.get("training_reference_mode") != "terminal_131k"
        or catalog.get("scientific_data_opened") is not False
        or catalog.get("final_evaluation_materialized") is not False
    ):
        raise TrainingGateError("final reference catalog status is invalid")
    roles = catalog.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(REFERENCE_COUNTS):
        raise TrainingGateError("final reference catalog role set changed")
    observed_roots: set[Path] = set()
    for role, (accepted_count, dataset_count) in REFERENCE_COUNTS.items():
        specification = roles[role]
        if not isinstance(specification, Mapping):
            raise TrainingGateError(f"final reference role {role} is malformed")
        datasets = specification.get("datasets")
        exclusions = specification.get("excluded_physical_system_ids", [])
        if (
            int(specification.get("accepted_system_count", -1)) != accepted_count
            or not isinstance(datasets, list)
            or len(datasets) != dataset_count
            or not isinstance(exclusions, list)
            or (role == "train" and len(exclusions) != 5)
            or (role != "train" and exclusions)
        ):
            raise TrainingGateError(f"final reference role {role} arithmetic changed")
        resolved = [
            resolve_bound_published_reference_dataset(
                item,
                approved_root=approved_root,
            )
            for item in datasets
        ]
        for item in resolved:
            if item.dataset_root in observed_roots:
                raise TrainingGateError("final reference dataset root is reused")
            observed_roots.add(item.dataset_root)
    return catalog


def build_final_reference_catalog(
    root: Path,
    *,
    terminal_probe_authorization_path: Path,
    calibration_sbc_result_path: Path,
) -> Mapping[str, Any]:
    """Resolve the 131k train, validation, calibration and SBC child roots."""

    load_final_materialization_release_stack_contract(root)
    terminal_authorization = load_yaml(terminal_probe_authorization_path)
    if terminal_authorization.get("authorization_status") != (
        TERMINAL_PROBE_AUTHORIZATION_STATUS
    ):
        raise TrainingGateError("terminal probe authorization identity changed")
    roots = terminal_authorization.get("publication_roots")
    if not isinstance(roots, Mapping):
        raise TrainingGateError("terminal publication roots are absent")
    required_roots = {
        "stage_a",
        "stage_b",
        "combined_base",
        "correction",
        "terminal_train_increment",
        "terminal_combined_131k",
        "development_tail",
    }
    if set(roots) != required_roots:
        raise TrainingGateError("terminal publication root set changed")
    terminal = resolve_terminal_131k_training_publication(
        terminal_authorization,
        stage_a_publication_root=Path(str(roots["stage_a"])),
        stage_b_publication_root=Path(str(roots["stage_b"])),
        combined_base_publication_root=Path(str(roots["combined_base"])),
        correction_publication_root=Path(str(roots["correction"])),
        train_parent_root=Path(str(roots["terminal_train_increment"])),
        combined_131k_publication_root=Path(str(roots["terminal_combined_131k"])),
        development_tail_parent_root=Path(str(roots["development_tail"])),
    )
    corrected = terminal.corrected_65k
    excluded = list(
        corrected.stage_a_excluded_ids + corrected.stage_b_excluded_ids
    )
    if (
        len(excluded) != 5
        or corrected.corrected_combined_train_manifest_sha256
        != CORRECTED_COMBINED_TRAIN_MANIFEST_HASH
    ):
        raise TrainingGateError("corrected train overlay identity changed")
    calibration_result_path = _project_file(
        calibration_sbc_result_path,
        name="calibration/SBC materialization result",
    )
    calibration_result = _load_json(calibration_result_path)
    calibration_parent = Path(
        str(calibration_result.get("publication_path", ""))
    ).resolve()
    calibration_manifest = calibration_parent / "dataset_manifest.json"
    if (
        calibration_result.get("status") != "passed"
        or int(calibration_result.get("accepted_pair_count", -1)) != 6144
        or int(calibration_result.get("complete_shard_count", -1)) != 48
        or calibration_result.get("calibration_fit_statistics_authorized") is not False
        or calibration_result.get("sbc_statistics_authorized") is not False
        or calibration_result.get("final_evaluation_authorized") is not False
        or not calibration_parent.is_relative_to(PROJECT_ROOT)
        or "staging" in calibration_parent.parts
        or not calibration_manifest.is_file()
    ):
        raise TrainingGateError("calibration/SBC publication result is invalid")
    calibration_manifest_hash = _sha256(calibration_manifest)
    calibration_id = str(calibration_result.get("calibration_dataset_id", ""))
    sbc_id = str(calibration_result.get("sbc_dataset_id", ""))
    stage_a = corrected.stage_a
    stage_b = corrected.combined_base
    train_datasets = [
        _dataset_entry(
            stage_a.train_root,
            stage_a.parent_root,
            stage_a.manifest_sha256,
        ),
        _dataset_entry(
            stage_b.stage_b_train_root,
            stage_b.stage_b_parent_root,
            stage_b.stage_b_parent_manifest_sha256,
        ),
        _dataset_entry(
            corrected.stage_a_replacement_root,
            corrected.correction_root,
            corrected.correction_manifest_sha256,
        ),
        _dataset_entry(
            corrected.stage_b_replacement_root,
            corrected.correction_root,
            corrected.correction_manifest_sha256,
        ),
        _dataset_entry(
            terminal.train_increment_root,
            terminal.train_parent_root,
            terminal.train_parent_manifest_sha256,
        ),
    ]
    roles = {
        "train": {
            "accepted_system_count": 131072,
            "datasets": train_datasets,
            "excluded_physical_system_ids": excluded,
            "logical_manifest_sha256": terminal.combined_manifest_sha256,
        },
        "validation": {
            "accepted_system_count": 6144,
            "datasets": [
                _dataset_entry(
                    stage_a.validation_root,
                    stage_a.parent_root,
                    stage_a.manifest_sha256,
                )
            ],
            "excluded_physical_system_ids": [],
            "logical_manifest_sha256": terminal_authorization[
                "retained_65k_probe"
            ]["shared_identity"]["validation_manifest_sha256"],
        },
        "calibration_fit": {
            "accepted_system_count": 4096,
            "datasets": [
                _dataset_entry(
                    calibration_parent / calibration_id,
                    calibration_parent,
                    calibration_manifest_hash,
                )
            ],
            "excluded_physical_system_ids": [],
            "logical_manifest_sha256": calibration_manifest_hash,
        },
        "sbc_diagnostic": {
            "accepted_system_count": 2048,
            "datasets": [
                _dataset_entry(
                    calibration_parent / sbc_id,
                    calibration_parent,
                    calibration_manifest_hash,
                )
            ],
            "excluded_physical_system_ids": [],
            "logical_manifest_sha256": calibration_manifest_hash,
        },
    }
    catalog = {
        "status": REFERENCE_CATALOG_STATUS,
        "training_reference_mode": "terminal_131k",
        "roles": roles,
        "terminal_combined_train_manifest_sha256": (
            terminal.combined_manifest_sha256
        ),
        "terminal_train_increment_parent_manifest_sha256": (
            terminal.train_parent_manifest_sha256
        ),
        "development_tail_manifest_sha256": (
            terminal.development_tail_manifest_sha256
        ),
        "corrected_combined_train_manifest_sha256": (
            corrected.corrected_combined_train_manifest_sha256
        ),
        "correction_parent_manifest_sha256": (
            corrected.correction_manifest_sha256
        ),
        "correction_publication_tree_sha256": (
            corrected.correction_tree_sha256
        ),
        "calibration_sbc_result_path": str(calibration_result_path),
        "calibration_sbc_result_sha256": _sha256(calibration_result_path),
        "calibration_sbc_parent_manifest_sha256": calibration_manifest_hash,
        "scientific_data_opened": False,
        "final_evaluation_materialized": False,
    }
    return validate_final_reference_catalog(catalog)


def _wheel_evidence(
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    *,
    implementation_commit: str,
) -> Mapping[str, Any]:
    wheel = _project_file(wheel_path, name="final generator wheel")
    test_result_path = _project_file(
        exact_wheel_test_result_path,
        name="final exact-wheel test result",
    )
    environment = _project_file(
        environment_lock_path,
        name="final environment lock",
    )
    wheel_hash = _sha256(wheel)
    result = _load_json(test_result_path)
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
        raise TrainingGateError("final exact-wheel evidence did not pass")
    return {
        "git_commit": implementation_commit,
        "wheel_path": str(wheel),
        "wheel_sha256": wheel_hash,
        "exact_wheel_test_result_path": str(test_result_path),
        "exact_wheel_test_result_sha256": _sha256(test_result_path),
        "environment_lock_path": str(environment),
        "environment_lock_sha256": _sha256(environment),
        "editable_install_authorized": False,
    }


def build_final_materialization_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    terminal_decision_path: Path,
    architecture_decision_path: Path,
    reference_catalog_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Build a non-authorizing packet after every upstream publication exists."""

    root = root.resolve()
    load_final_materialization_release_stack_contract(root)
    implementation_commit = _commit(
        implementation_commit,
        name="final materialization implementation commit",
    )
    _verify_release_checkout(root, implementation_commit)
    config, _ = load_final_evaluation_contract(root)
    terminal_decision_path = _project_file(
        terminal_decision_path,
        name="terminal size decision",
    )
    architecture_decision_path = _project_file(
        architecture_decision_path,
        name="terminal architecture decision",
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
    reference_catalog_path = _project_file(
        reference_catalog_path,
        name="final reference catalog",
    )
    catalog = validate_final_reference_catalog(_load_json(reference_catalog_path))
    immutable = _wheel_evidence(
        wheel_path,
        exact_wheel_test_result_path,
        environment_lock_path,
        implementation_commit=implementation_commit,
    )
    output_relative = _repo_relative(root, output_path, ("results", "phase7"))
    if output_relative != (
        "results/phase7/final_materialization_release_packet.json"
    ):
        raise TrainingGateError("final materialization release-packet path changed")
    architecture = decisions["architecture"]
    terminal = decisions["terminal"]
    references = catalog["roles"]
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "materialization_execution_authorized": False,
        "official_identities": None,
        "implementation_commit": implementation_commit,
        "frozen_contract": {
            "configuration_path": FINAL_EVALUATION_CONFIG,
            "configuration_hash": configuration_hash(config),
            "commitment_sha256": FINAL_EVALUATION_COMMITMENT_HASH,
            "numerical_validity_addendum_sha256": (
                NUMERICAL_VALIDITY_ADDENDUM_HASH
            ),
            "waveform_numerical_validity_preregistration_hash": (
                CORRECTION_PREREGISTRATION_HASH
            ),
        },
        "prospective_generator_revision": {
            "original_committed_generator": ORIGINAL_COMMITTED_GENERATOR,
            "scope": "waveform_numerical_validity_implementation_only",
            "counts_seeds_distributions_changed": False,
            "original_commitment_mutated": False,
        },
        "training_size_decision": {
            "path": str(terminal_decision_path),
            "sha256": terminal_hash,
            "decision": terminal["decision"],
            "selected_training_count": 131072,
        },
        "architecture_decision": {
            "path": str(architecture_decision_path),
            "sha256": architecture_hash,
            "selected_architecture_id": architecture["architecture_id"],
            "model_configuration_hash": architecture[
                "model_configuration_hash"
            ],
            "locked_training_rung": 131072,
            "result_count": 12,
            "three_model_seeds_retained": True,
        },
        "reference_catalog": {
            "path": str(reference_catalog_path),
            "sha256": _sha256(reference_catalog_path),
            "status": REFERENCE_CATALOG_STATUS,
        },
        "published_reference_contract": {
            "training_reference_mode": "terminal_131k",
            "corrected_combined_train_manifest_sha256": catalog[
                "corrected_combined_train_manifest_sha256"
            ],
            "correction_parent_manifest_sha256": (
                CORRECTION_PARENT_MANIFEST_HASH
            ),
            "correction_publication_tree_sha256": (
                CORRECTION_PUBLICATION_TREE_HASH
            ),
            "terminal_preregistration_hash": TERMINAL_PREREGISTRATION_HASH,
            "terminal_combined_train_manifest_sha256": catalog[
                "terminal_combined_train_manifest_sha256"
            ],
            "terminal_train_increment_parent_manifest_sha256": catalog[
                "terminal_train_increment_parent_manifest_sha256"
            ],
            "development_tail_manifest_sha256": catalog[
                "development_tail_manifest_sha256"
            ],
            "validation_manifest_sha256": references["validation"][
                "logical_manifest_sha256"
            ],
            "strict_corrected_65k_subset": True,
            "development_tail_excluded_from_final_reference": True,
            "extension_above_131072_authorized": False,
            "terminal_size_decision": terminal["decision"],
            "terminal_size_decision_sha256": terminal_hash,
            "selected_architecture_locked_rung": 131072,
            "selected_architecture_decision_sha256": architecture_hash,
            "logical_system_counts": {
                role: count for role, (count, _) in REFERENCE_COUNTS.items()
            },
        },
        "published_reference_datasets": references,
        "immutable_generator": immutable,
        "materialization_contract": {
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "training_size_and_architecture_locked": True,
        },
        "future_authorization_path": (
            "configs/execution/"
            "phase7_final_evaluation_materialization_authorization.yaml"
        ),
        "future_delegated_review_path": (
            "results/phase7/final_materialization_review.json"
        ),
        "release_packet_path": output_relative,
        "review_scope": {
            "training_reference_mode": "terminal_131k",
            "locked_training_rung": 131072,
            "selected_architecture_id": architecture["architecture_id"],
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "sealed_materialization_authorized": True,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
        },
    }


def _validate_review(
    review: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    packet_hash: str,
) -> None:
    if (
        review.get("status") != REVIEW_STATUS
        or review.get("release_packet_sha256") != packet_hash
        or review.get("reviewed_by") != (
            "codex_as_delegated_scientific_and_engineering_reviewer"
        )
    ):
        raise TrainingGateError("final materialization delegated review is invalid")
    expected = packet["review_scope"]
    if any(review.get(name) != expected[name] for name in REVIEW_FIELDS):
        raise TrainingGateError("final materialization review scope changed")


def build_final_materialization_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Promote one reviewed packet into sealed-materialization authorization."""

    root = root.resolve()
    load_final_materialization_release_stack_contract(root)
    packet = _load_json(release_packet_path)
    packet_hash = _sha256(release_packet_path)
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("materialization_execution_authorized") is not False
        or packet.get("official_identities") is not None
    ):
        raise TrainingGateError("final materialization packet is not review-ready")
    review = _load_json(delegated_review_path)
    _validate_review(review, packet=packet, packet_hash=packet_hash)
    output_relative = _repo_relative(root, output_path, ("configs", "execution"))
    if output_relative != packet["future_authorization_path"]:
        raise TrainingGateError("final materialization authorization path changed")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        ("results", "phase7"),
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        ("results", "phase7"),
    )
    if (
        release_relative != packet["release_packet_path"]
        or review_relative != packet["future_delegated_review_path"]
    ):
        raise TrainingGateError("final materialization review evidence path changed")
    return {
        "phase": "7-final-evaluation-materialization",
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
        "immutable_generator": dict(packet["immutable_generator"]),
        "frozen_contract": dict(packet["frozen_contract"]),
        "prospective_generator_revision": dict(
            packet["prospective_generator_revision"]
        ),
        "materialization_contract": dict(packet["materialization_contract"]),
        "training_size_decision": dict(packet["training_size_decision"]),
        "architecture_decision": dict(packet["architecture_decision"]),
        "published_reference_contract": dict(
            packet["published_reference_contract"]
        ),
        "published_reference_datasets": dict(
            packet["published_reference_datasets"]
        ),
        "authorizing_commit": None,
        "authorization": {
            "sealed_materialization_authorized": True,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
            "model_training_authorized": False,
            "calibration_fit_authorized": False,
            "learning_curve_use_authorized": False,
            "architecture_selection_use_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "stop_after_atomic_sealed_materialization": True,
    }
