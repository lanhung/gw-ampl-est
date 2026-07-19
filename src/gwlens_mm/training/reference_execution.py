"""Fail-closed execution surface for the frozen RC.7 simulation reference."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..config import load_yaml
from ..schema import SplitName
from .architecture import selected_model_configuration
from .calibration import wilson_interval
from .contracts import WAVEFORM_CORRECTION_HASH, model_configuration_hash
from .data import (
    CorrectedTrainingPublication,
    PublishedStageADataset,
    corrected_65k_training_dataset,
    resolve_corrected_training_publication,
)
from .engine import standardizer_hash
from .features import InputStandardizer, load_input_policy
from .reference_baseline import (
    POSTERIOR_DRAW_COUNT,
    REFERENCE_CONFIG_HASH,
    REFERENCE_LEVELS,
    ReferenceBankIndex,
    ReferenceBaselineGateError,
    ReferenceCaseScore,
    build_standardized_reference_bank,
    load_reference_baseline_contract,
)
from .rung65 import TRAIN_65K_COUNT, VALIDATION_COUNT, _load_standardizers
from .runner import _verified_curves, _verify_training_checkout

IMPLEMENTATION_AUTHORIZATION = (
    "configs/execution/phase7_reference_execution_stack_authorization.yaml"
)
QUERY_SPECS: Mapping[str, Tuple[SplitName, int]] = {
    "validation": (SplitName.VALIDATION, VALIDATION_COUNT),
    "iid_test": (SplitName.IID_TEST, 8192),
    "balanced_tail_diagnostic": (SplitName.BALANCED_TAIL_DIAGNOSTIC, 4096),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReferenceBaselineGateError(f"expected JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def validate_reference_execution_stack_contract(root: Path) -> Mapping[str, Any]:
    """Require RC.7 and keep the implementation checkpoint non-executable."""

    config, _ = load_reference_baseline_contract(root)
    authorization = load_yaml(root / IMPLEMENTATION_AUTHORIZATION)
    if authorization.get("authorization_status") != "authorized_implementation_only":
        raise ReferenceBaselineGateError("reference runner implementation gate is absent")
    if authorization.get("frozen_addendum", {}).get("canonical_hash") != (
        REFERENCE_CONFIG_HASH
    ):
        raise ReferenceBaselineGateError("reference runner gate changed RC.7")
    contract = authorization.get("implementation_contract", {})
    if not (
        contract.get("query_roles") == list(QUERY_SPECS)
        and contract.get("query_counts")
        == {role: count for role, (_, count) in QUERY_SPECS.items()}
        and int(contract.get("exact_neighbors", -1)) == 256
        and int(contract.get("posterior_draws_per_case", -1))
        == POSTERIOR_DRAW_COUNT
        and contract.get("posterior_draws_persisted") is False
        and contract.get("raw_coverage_counts_and_wilson_intervals_required") is True
        and contract.get("final_query_roles_require_separate_unsealing_gate") is True
    ):
        raise ReferenceBaselineGateError("reference runner implementation contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "fail_closed_reference_runner_implementation_authorized",
        "bounded_score_writer_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed):
        raise ReferenceBaselineGateError("reference runner implementation is incomplete")
    if any(value is not False for name, value in flags.items() if name not in allowed):
        raise ReferenceBaselineGateError("reference implementation opened execution")
    return {"config": config, "authorization": authorization}


def _validate_terminal_and_architecture(
    root: Path,
    authorization: Mapping[str, Any],
    *,
    terminal_decision_path: Path,
    selected_architecture_decision_path: Path,
) -> Tuple[str, Mapping[str, Any]]:
    configured_terminal = Path(
        str(authorization.get("terminal_size_decision_path", ""))
    ).resolve()
    terminal = _load_mapping(terminal_decision_path)
    if not (
        terminal_decision_path.resolve() == configured_terminal
        and _sha256(terminal_decision_path)
        == authorization.get("terminal_size_decision_sha256")
        and terminal.get("decision") == "lock_train_65k"
        and terminal.get("comparison") == "train_32k_to_train_65k"
        and terminal.get("extension_above_65536_authorized") is False
    ):
        raise ReferenceBaselineGateError("reference gate lacks the exact terminal lock")

    configured_selection = Path(
        str(authorization.get("selected_architecture_decision_path", ""))
    ).resolve()
    selected = _load_mapping(selected_architecture_decision_path)
    architecture_id = str(selected.get("selected_architecture_id", ""))
    if not (
        selected_architecture_decision_path.resolve() == configured_selection
        and _sha256(selected_architecture_decision_path)
        == authorization.get("selected_architecture_decision_sha256")
        and selected.get("status") == "architecture_locked_on_development_validation"
        and int(selected.get("total_result_count", -1)) == 12
        and selected.get("best_seed_selected") is False
        and selected.get("calibration_accessed") is False
        and selected.get("sbc_accessed") is False
        and selected.get("final_evaluation_accessed") is False
    ):
        raise ReferenceBaselineGateError(
            "reference gate lacks the exact development-only architecture lock"
        )
    model = selected_model_configuration(root, architecture_id)
    if model_configuration_hash(model) != authorization.get(
        "selected_primary_model_configuration_hash"
    ):
        raise ReferenceBaselineGateError("selected primary model changed")
    return architecture_id, model


def validate_reference_query_execution_gate(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Path,
    terminal_decision_path: Path,
    selected_architecture_decision_path: Path,
    primary_rung_preparation_path: Path,
    query_dataset_root: Path,
    query_parent_manifest_path: Path,
) -> Mapping[str, Any]:
    """Resolve exact metadata-only bank/query identities before record access."""

    validate_reference_execution_stack_contract(root)
    authorization = load_yaml(authorization_path)
    if authorization.get("authorization_status") != (
        "authorized_reference_query_execution_only"
    ):
        raise ReferenceBaselineGateError("scientific reference-query gate is absent")
    flags = authorization.get("authorization", {})
    for required in (
        "scientific_reference_bank_access_authorized",
        "reference_query_execution_authorized",
    ):
        if flags.get(required) is not True:
            raise ReferenceBaselineGateError(f"reference gate requires {required}=true")
    for forbidden in (
        "checkpoint_access_authorized",
        "calibration_refit_authorized",
        "model_retraining_or_tuning_authorized",
        "likelihood_gold_claim_authorized",
        "importance_sampling_efficiency_claim_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        if flags.get(forbidden) is not False:
            raise ReferenceBaselineGateError(f"reference gate must keep {forbidden}=false")
    role = str(authorization.get("query_role", ""))
    if role not in QUERY_SPECS:
        raise ReferenceBaselineGateError("reference query role is not preregistered")
    expected_split, expected_count = QUERY_SPECS[role]
    if not (
        int(authorization.get("query_count", -1)) == expected_count
        and authorization.get("reference_id")
        == "selected_prior_em_timing_knn_kde_v1"
        and int(authorization.get("neighbor_count", -1)) == 256
        and int(authorization.get("posterior_draws_per_case", -1))
        == POSTERIOR_DRAW_COUNT
    ):
        raise ReferenceBaselineGateError("reference query count or algorithm changed")
    final_role = role != "validation"
    if final_role:
        if not (
            flags.get("validation_reference_execution_authorized") is False
            and flags.get("final_evaluation_unsealing_authorized") is True
            and flags.get("final_reference_execution_authorized") is True
        ):
            raise ReferenceBaselineGateError("final reference lacks its unsealing gate")
    elif not (
        flags.get("validation_reference_execution_authorized") is True
        and flags.get("final_evaluation_unsealing_authorized") is False
        and flags.get("final_reference_execution_authorized") is False
    ):
        raise ReferenceBaselineGateError("validation reference gate crossed final boundary")

    architecture_id, model = _validate_terminal_and_architecture(
        root,
        authorization,
        terminal_decision_path=terminal_decision_path,
        selected_architecture_decision_path=selected_architecture_decision_path,
    )
    correction = authorization.get("correction_publication", {})
    publication = resolve_corrected_training_publication(
        correction_publication_root,
        stage_a_parent_root=stage_a_publication_root,
        stage_b_parent_root=stage_b_publication_root,
        combined_base_root=combined_publication_root,
        expected_base_generator_commit=str(authorization.get("base_generator_commit", "")),
        expected_base_preregistration_hash=str(
            authorization.get("base_preregistration_hash", "")
        ),
        expected_correction_generator_commit=str(correction.get("generator_commit", "")),
        expected_correction_preregistration_hash=WAVEFORM_CORRECTION_HASH,
        expected_correction_manifest_sha256=str(
            correction.get("parent_manifest_sha256", "")
        ),
        expected_correction_tree_sha256=str(
            correction.get("publication_tree_sha256", "")
        ),
        expected_combined_base_manifest_sha256=str(
            authorization.get("combined_base_manifest_sha256", "")
        ),
    )
    if authorization.get("corrected_combined_train_manifest_sha256") != (
        publication.corrected_combined_train_manifest_sha256
    ):
        raise ReferenceBaselineGateError("reference gate changed the corrected bank")

    configured_preparation = Path(
        str(authorization.get("primary_rung_preparation_path", ""))
    ).resolve()
    preparation = _load_mapping(primary_rung_preparation_path)
    if not (
        primary_rung_preparation_path.resolve() == configured_preparation
        and _sha256(primary_rung_preparation_path)
        == authorization.get("primary_rung_preparation_sha256")
        and preparation.get("status") == "ready_for_authorized_probe_fits"
        and int(preparation.get("rung_count", -1)) == TRAIN_65K_COUNT
        and int(preparation.get("member_count", -1)) == TRAIN_65K_COUNT
        and preparation.get("train_manifest_sha256")
        == publication.corrected_combined_train_manifest_sha256
        and preparation.get("validation_manifest_sha256")
        == publication.stage_a.namespace_manifest_sha256["validation"]
    ):
        raise ReferenceBaselineGateError("reference gate lacks primary preprocessing")
    input_standardizer, _ = _load_standardizers(preparation)
    if standardizer_hash(input_standardizer) != preparation.get(
        "input_standardizer_sha256"
    ):
        raise ReferenceBaselineGateError("reference standardizer hash changed")

    configured_query = Path(str(authorization.get("query_dataset_root", ""))).resolve()
    configured_parent = Path(
        str(authorization.get("query_parent_manifest_path", ""))
    ).resolve()
    if not (
        query_dataset_root.resolve() == configured_query
        and query_parent_manifest_path.resolve() == configured_parent
        and query_dataset_root.is_relative_to(Path("/root/autodl-tmp/lensing-4"))
        and query_parent_manifest_path.is_file()
        and _sha256(query_parent_manifest_path)
        == authorization.get("query_parent_manifest_sha256")
    ):
        raise ReferenceBaselineGateError("reference query parent identity changed")
    dataset_manifest = query_dataset_root / "dataset_manifest.json"
    if not dataset_manifest.is_file() or _sha256(dataset_manifest) != authorization.get(
        "query_dataset_manifest_sha256"
    ):
        raise ReferenceBaselineGateError("reference query dataset manifest changed")
    if role == "validation" and not (
        query_dataset_root.resolve() == publication.stage_a.validation_root.resolve()
        and _sha256(dataset_manifest)
        == publication.stage_a.namespace_manifest_sha256["validation"]
    ):
        raise ReferenceBaselineGateError("validation query is not the locked development set")
    return {
        "authorization": authorization,
        "publication": publication,
        "query_role": role,
        "query_split": expected_split,
        "query_count": expected_count,
        "selected_architecture_id": architecture_id,
        "selected_primary_model": model,
        "input_standardizer": input_standardizer,
        "query_dataset_manifest_sha256": _sha256(dataset_manifest),
    }


@dataclass
class _ScoreAggregate:
    count: int = 0
    nlp_sum: float = 0.0
    crps_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    width_sum: Dict[str, np.ndarray] = field(default_factory=dict)
    marginal_counts: Dict[str, np.ndarray] = field(default_factory=dict)
    joint_counts: Dict[str, int] = field(default_factory=dict)

    def add(self, score: ReferenceCaseScore) -> None:
        self.count += 1
        self.nlp_sum += -float(score.log_probability) / 2.0
        self.crps_sum += np.asarray(score.crps, dtype=np.float64)
        for level in REFERENCE_LEVELS:
            label = f"{level:.2f}"
            self.width_sum.setdefault(label, np.zeros(2, dtype=np.float64))
            self.marginal_counts.setdefault(label, np.zeros(2, dtype=np.int64))
            self.width_sum[label] += np.asarray(score.interval_width[label])
            self.marginal_counts[label] += np.asarray(
                score.marginal_coverage[label], dtype=np.int64
            )
            self.joint_counts[label] = self.joint_counts.get(label, 0) + int(
                score.joint_central_coverage[label]
            )

    def as_mapping(self) -> Mapping[str, Any]:
        if self.count <= 0:
            raise ReferenceBaselineGateError("reference score group is empty")
        coverage = {}
        for level in REFERENCE_LEVELS:
            label = f"{level:.2f}"
            marginal = self.marginal_counts[label]
            joint = self.joint_counts[label]
            coverage[label] = {
                "marginal_success_counts": marginal.tolist(),
                "marginal_rates": (marginal / self.count).tolist(),
                "marginal_wilson_95": [
                    list(wilson_interval(int(value), self.count)) for value in marginal
                ],
                "joint_success_count": joint,
                "joint_rate": joint / self.count,
                "joint_wilson_95": list(wilson_interval(joint, self.count)),
                "mean_interval_width": (self.width_sum[label] / self.count).tolist(),
            }
        return {
            "case_count": self.count,
            "mean_negative_log_probability_nat_per_target_dimension": (
                self.nlp_sum / self.count
            ),
            "mean_crps": (self.crps_sum / self.count).tolist(),
            "coverage": coverage,
        }


def score_reference_query_to_artifacts(
    index: ReferenceBankIndex,
    dataset: PublishedStageADataset,
    standardizer: InputStandardizer,
    *,
    query_role: str,
    expected_count: int,
    score_jsonl_path: Path,
    summary_path: Path,
    bank_manifest_path: Path,
) -> Mapping[str, Any]:
    """Score one query record at a time and atomically publish small products."""

    if query_role not in QUERY_SPECS or QUERY_SPECS[query_role][1] != expected_count:
        raise ReferenceBaselineGateError("reference query role/count changed")
    if len(dataset) != expected_count:
        raise ReferenceBaselineGateError("reference query count is not exact")
    for path in (score_jsonl_path, summary_path, bank_manifest_path):
        if path.exists():
            raise FileExistsError("reference output identity already exists")
    bank_manifest = {
        **index.manifest(),
        "query_role": query_role,
        "query_count": expected_count,
    }
    _atomic_json(bank_manifest_path, bank_manifest)
    partial = score_jsonl_path.with_name(score_jsonl_path.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    overall = _ScoreAggregate()
    by_family: Dict[str, _ScoreAggregate] = {}
    by_cell: Dict[str, _ScoreAggregate] = {}
    query_digest = hashlib.sha256()
    seen = set()
    try:
        with partial.open("x", encoding="utf-8") as stream:
            for position in range(len(dataset)):
                query = standardizer.transform(dataset.metadata_example(position))
                if query.gw_strain.size:
                    raise ReferenceBaselineGateError("reference query opened GW strain")
                if query.physical_system_id in seen:
                    raise ReferenceBaselineGateError("reference query ID is duplicated")
                seen.add(query.physical_system_id)
                score = index.score(query)
                mapping = score.as_mapping()
                stream.write(json.dumps(mapping, sort_keys=True, allow_nan=False) + "\n")
                query_digest.update(query.physical_system_id.encode("utf-8"))
                query_digest.update(b"\0")
                overall.add(score)
                by_family.setdefault(score.lens_family, _ScoreAggregate()).add(score)
                by_cell.setdefault(score.em_cell, _ScoreAggregate()).add(score)
        os.replace(partial, score_jsonl_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    summary = {
        "status": "completed_non_neural_reference_query",
        "reference_id": "selected_prior_em_timing_knn_kde_v1",
        "reference_is_exact_likelihood_or_gold": False,
        "query_role": query_role,
        "query_count": expected_count,
        "query_identity_sha256": query_digest.hexdigest(),
        "reference_bank_identity_sha256": index.identity_sha256,
        "score_jsonl_sha256": _sha256(score_jsonl_path),
        "overall": overall.as_mapping(),
        "by_lens_family": {
            key: value.as_mapping() for key, value in sorted(by_family.items())
        },
        "by_em_cell": {
            key: value.as_mapping() for key, value in sorted(by_cell.items())
        },
        "posterior_draws_persisted": False,
        "gw_strain_opened": False,
        "checkpoint_accessed": False,
        "calibration_refit": False,
        "importance_sampling_efficiency_computed": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(summary_path, summary)
    return summary


def run_authorized_reference_query(
    root: Path,
    *,
    authorization_path: Path,
    stage_a_publication_root: Path,
    stage_b_publication_root: Path,
    combined_publication_root: Path,
    correction_publication_root: Path,
    terminal_decision_path: Path,
    selected_architecture_decision_path: Path,
    primary_rung_preparation_path: Path,
    query_dataset_root: Path,
    query_parent_manifest_path: Path,
    psd_root: Path,
    output_root: Path,
    execution_commit: str,
) -> Mapping[str, Any]:
    """Execute one exact future validation or final RC.7 reference query."""

    gate = validate_reference_query_execution_gate(
        root,
        authorization_path=authorization_path,
        stage_a_publication_root=stage_a_publication_root,
        stage_b_publication_root=stage_b_publication_root,
        combined_publication_root=combined_publication_root,
        correction_publication_root=correction_publication_root,
        terminal_decision_path=terminal_decision_path,
        selected_architecture_decision_path=selected_architecture_decision_path,
        primary_rung_preparation_path=primary_rung_preparation_path,
        query_dataset_root=query_dataset_root,
        query_parent_manifest_path=query_parent_manifest_path,
    )
    authorization = gate["authorization"]
    immutable = authorization.get("immutable_execution", {})
    if immutable.get("git_commit") != execution_commit:
        raise ReferenceBaselineGateError("reference execution commit changed")
    wheel = Path(str(immutable.get("wheel_path", ""))).resolve()
    environment = Path(str(immutable.get("environment_lock_path", ""))).resolve()
    if not (
        wheel.is_file()
        and _sha256(wheel) == immutable.get("wheel_sha256")
        and environment.is_file()
        and _sha256(environment) == immutable.get("environment_lock_sha256")
        and immutable.get("editable_install_authorized") is False
    ):
        raise ReferenceBaselineGateError("reference immutable execution identity failed")
    _verify_training_checkout(root, execution_commit, authorization_path)
    configured_output = Path(str(authorization.get("reference_output_root", ""))).resolve()
    if output_root.resolve() != configured_output or not output_root.is_relative_to(
        Path("/root/autodl-tmp/lensing-4")
    ):
        raise ReferenceBaselineGateError("reference output root changed")
    model = gate["selected_primary_model"]
    load_input_policy(root)
    curves = _verified_curves(model, psd_root)
    publication = gate["publication"]
    if not isinstance(publication, CorrectedTrainingPublication):
        raise ReferenceBaselineGateError("reference gate returned an invalid bank")
    bank_dataset = corrected_65k_training_dataset(publication, curves)
    query_dataset = PublishedStageADataset(
        query_dataset_root,
        expected_split=gate["query_split"],
        detector_curves=curves,
        expected_total_pairs=int(gate["query_count"]),
    )
    if set(bank_dataset.physical_system_ids()) & set(query_dataset.physical_system_ids()):
        raise ReferenceBaselineGateError("reference bank overlaps query identities")
    standardizer = gate["input_standardizer"]
    index = build_standardized_reference_bank(bank_dataset, standardizer)
    role = str(gate["query_role"])
    role_root = output_root / role
    result = score_reference_query_to_artifacts(
        index,
        query_dataset,
        standardizer,
        query_role=role,
        expected_count=int(gate["query_count"]),
        score_jsonl_path=role_root / "reference_scores.jsonl",
        summary_path=role_root / "reference_summary.json",
        bank_manifest_path=role_root / "reference_bank_manifest.json",
    )
    execution_manifest = {
        "status": "completed_authorized_reference_query",
        "execution_commit": execution_commit,
        "immutable_wheel_sha256": immutable.get("wheel_sha256"),
        "environment_lock_sha256": immutable.get("environment_lock_sha256"),
        "selected_architecture_id": gate["selected_architecture_id"],
        "corrected_train_manifest_sha256": (
            publication.corrected_combined_train_manifest_sha256
        ),
        "query_dataset_manifest_sha256": gate["query_dataset_manifest_sha256"],
        "query_role": role,
        "query_count": gate["query_count"],
        "reference_summary_sha256": _sha256(
            role_root / "reference_summary.json"
        ),
        "reference_score_sha256": _sha256(role_root / "reference_scores.jsonl"),
        "checkpoint_accessed": False,
        "final_evaluation_accessed": role != "validation",
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(role_root / "execution_manifest.json", execution_manifest)
    return {**result, "execution_manifest": execution_manifest}
