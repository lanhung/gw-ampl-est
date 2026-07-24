"""Exact, three-seed final-score reporting and release control.

The implementation gate permits synthetic fixtures only.  A future
hash-bound authorization is required before any scientific score artifact may
be opened.  Final records, checkpoints and calibration maps are never inputs
to this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..config import load_yaml
from ..production.final_evaluation import FinalEvaluationNamespace
from .contracts import TrainingGateError
from .final_evaluation import (
    FINAL_ANALYSIS_HASH,
    FINAL_COMMITMENT_SHA256,
    aggregate_seed_metrics,
    coverage_gate,
)
from .final_inference import (
    FINAL_CASE_COUNT,
    FINAL_NAMESPACE_COUNT,
    MODEL_SEEDS,
    _expected_namespaces,
)
from .final_inference_authorization import (
    AUTHORIZATION_STATUS as FINAL_INFERENCE_AUTHORIZATION_STATUS,
)
from .final_inference_authorization import (
    _full_commit,
    _immutable_inference,
    _load_json,
    _project_path,
    _repo_relative,
    _sha256,
    _verify_checkout,
)
from .terminal131 import TRAIN_131K_COUNT

STACK_AUTHORIZATION = (
    "configs/execution/phase7_final_summary_release_stack_authorization.yaml"
)
RELEASE_STATUS = "ready_for_delegated_final_score_summary_review"
REVIEW_STATUS = "delegated_final_score_summary_review_approved"
AUTHORIZATION_STATUS = "authorized_final_score_summary_only"
SCORE_ARTIFACT_COUNT = len(MODEL_SEEDS) * FINAL_NAMESPACE_COUNT
LEVELS = (0.50, 0.80, 0.90, 0.95)
EM_CELLS = (
    "full_precise_spectroscopic",
    "full_photometric_redshifts",
    "no_velocity_dispersion",
    "no_source_redshift",
    "no_lens_redshift",
    "astrometry_redshifts_only",
    "astrometry_kinematics_no_einstein_scale",
    "sparse_astrometry_timing_only",
)
LENS_FAMILIES = {"sie_external_shear", "epl_external_shear"}

_IDENTITY_KEYS = {
    "physical_system_ids",
    "lens_families",
    "em_cells",
    "splits",
    "diagnostic_context_ids",
    "model_seed",
    "architecture_id",
    "namespace_id",
    "checkpoint_sha256",
    "publication_manifest_sha256",
    "calibration_map_sha256",
    "inference_commit",
}
_METRIC_KEYS = {
    "truth",
    "truth_log_density",
    "nlp_nat_per_target_dimension",
    "crps",
    "marginal_region_scores",
    "joint_region_scores",
}
for _level in LEVELS:
    _suffix = f"{int(round(_level * 100)):02d}"
    _METRIC_KEYS.update(
        {
            f"marginal_covered_{_suffix}",
            f"joint_covered_{_suffix}",
            f"marginal_interval_width_{_suffix}",
        }
    )
REQUIRED_SCORE_KEYS = frozenset(_IDENTITY_KEYS | _METRIC_KEYS)

_REVIEW_SCOPE = {
    "locked_training_rung": TRAIN_131K_COUNT,
    "model_seeds": list(MODEL_SEEDS),
    "namespace_count": FINAL_NAMESPACE_COUNT,
    "accepted_case_count": FINAL_CASE_COUNT,
    "score_artifact_count": SCORE_ARTIFACT_COUNT,
    "final_score_artifact_access_authorized": True,
    "final_summary_execution_authorized": True,
    "final_record_access_authorized": False,
    "checkpoint_access_authorized": False,
    "calibration_map_access_authorized": False,
    "model_training_or_tuning_authorized": False,
    "calibration_refit_authorized": False,
    "architecture_or_size_selection_authorized": False,
    "final_result_threshold_change_authorized": False,
    "manuscript_claim_finalization_authorized": False,
    "gwosc_gwtc_access_authorized": False,
}


class FinalSummaryGateError(TrainingGateError):
    """Raised when a final-score summary crosses its frozen boundary."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def load_final_summary_stack_contract(root: Path) -> Mapping[str, Any]:
    """Prove that the current stack is implementation-only."""

    authorization = load_yaml(root / STACK_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_implementation_and_synthetic_fixture_only"
    ):
        raise FinalSummaryGateError("final-summary implementation gate is absent")
    frozen = authorization.get("frozen_contracts", {})
    if (
        int(frozen.get("locked_training_rung", -1)) != TRAIN_131K_COUNT
        or frozen.get("model_seeds") != list(MODEL_SEEDS)
        or frozen.get("final_analysis_hash") != FINAL_ANALYSIS_HASH
        or frozen.get("final_generation_commitment_sha256")
        != FINAL_COMMITMENT_SHA256
        or int(frozen.get("namespace_count", -1)) != FINAL_NAMESPACE_COUNT
        or int(frozen.get("accepted_case_count", -1)) != FINAL_CASE_COUNT
        or int(frozen.get("score_artifact_count", -1)) != SCORE_ARTIFACT_COUNT
        or tuple(float(value) for value in frozen.get("nominal_levels", ()))
        != LEVELS
    ):
        raise FinalSummaryGateError("final-summary parent contract drifted")
    flags = authorization.get("authorization", {})
    allowed = {
        "final_score_summary_implementation_authorized",
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed) or any(
        value is not False for name, value in flags.items() if name not in allowed
    ):
        raise FinalSummaryGateError("final-summary implementation opened execution")
    return authorization


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise FinalSummaryGateError(f"{name} must be one scalar")
    result = str(array.item())
    if not result:
        raise FinalSummaryGateError(f"{name} is empty")
    return result


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise FinalSummaryGateError(f"{name} must be one scalar")
    try:
        return int(array.item())
    except (TypeError, ValueError) as error:
        raise FinalSummaryGateError(f"{name} is not an integer") from error


def load_validated_score_payload(
    path: Path,
    *,
    specification: FinalEvaluationNamespace,
    model_seed: int,
) -> Mapping[str, np.ndarray]:
    """Open one future score artifact and enforce its complete array schema."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(REQUIRED_SCORE_KEYS):
            raise FinalSummaryGateError("final score artifact key set drifted")
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    count = specification.accepted_count
    expected_vector = (count,)
    expected_matrix = (count, 2)
    if (
        payload["truth"].shape != expected_matrix
        or payload["truth_log_density"].shape != expected_vector
        or payload["nlp_nat_per_target_dimension"].shape != expected_vector
        or payload["crps"].shape != expected_matrix
        or payload["marginal_region_scores"].shape != expected_matrix
        or payload["joint_region_scores"].shape != expected_vector
    ):
        raise FinalSummaryGateError("final score core metric shape drifted")
    for name in (
        "truth",
        "truth_log_density",
        "nlp_nat_per_target_dimension",
        "crps",
        "marginal_region_scores",
        "joint_region_scores",
    ):
        if not np.issubdtype(payload[name].dtype, np.number) or not np.all(
            np.isfinite(payload[name])
        ):
            raise FinalSummaryGateError(f"{name} is nonfinite or nonnumeric")
    if (
        not np.allclose(
            payload["nlp_nat_per_target_dimension"],
            -payload["truth_log_density"] / 2.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        or np.any(payload["crps"] < 0.0)
        or np.any(payload["marginal_region_scores"] < 0.0)
        or np.any(payload["marginal_region_scores"] > 1.0)
        or np.any(payload["joint_region_scores"] < 0.0)
        or np.any(payload["joint_region_scores"] > 1.0)
    ):
        raise FinalSummaryGateError("final score metric semantics drifted")
    for level in LEVELS:
        suffix = f"{int(round(level * 100)):02d}"
        marginal = payload[f"marginal_covered_{suffix}"]
        joint = payload[f"joint_covered_{suffix}"]
        width = payload[f"marginal_interval_width_{suffix}"]
        if (
            marginal.shape != expected_matrix
            or joint.shape != expected_vector
            or width.shape != expected_matrix
            or marginal.dtype != np.bool_
            or joint.dtype != np.bool_
            or not np.all(np.isfinite(width))
            or np.any(width < 0.0)
        ):
            raise FinalSummaryGateError("final coverage or interval payload drifted")
    for name in (
        "physical_system_ids",
        "lens_families",
        "em_cells",
        "splits",
        "diagnostic_context_ids",
    ):
        if payload[name].shape != expected_vector:
            raise FinalSummaryGateError(f"{name} shape drifted")
        values = tuple(str(value) for value in payload[name].tolist())
        if any(not value for value in values):
            raise FinalSummaryGateError(f"{name} contains an empty value")
    identifiers = tuple(str(value) for value in payload["physical_system_ids"])
    if len(set(identifiers)) != count:
        raise FinalSummaryGateError("final score physical-system IDs are duplicated")
    if set(str(value) for value in payload["splits"]) != {
        specification.split.value
    }:
        raise FinalSummaryGateError("final score split identity drifted")
    if set(str(value) for value in payload["diagnostic_context_ids"]) != {
        specification.diagnostic_context_id
    }:
        raise FinalSummaryGateError("final score diagnostic context drifted")
    families = set(str(value) for value in payload["lens_families"])
    if not families or not families.issubset(LENS_FAMILIES):
        raise FinalSummaryGateError("final score lens-family labels drifted")
    if specification.split.value == "iid_test":
        cells = tuple(str(value) for value in payload["em_cells"])
        expected_per_cell = count // len(EM_CELLS)
        if (
            count % len(EM_CELLS)
            or set(cells) != set(EM_CELLS)
            or any(cells.count(cell) != expected_per_cell for cell in EM_CELLS)
        ):
            raise FinalSummaryGateError("IID EM-cell balance drifted")
    if (
        _scalar_integer(payload["model_seed"], name="model_seed") != model_seed
        or _scalar_string(payload["namespace_id"], name="namespace_id")
        != specification.namespace_id
    ):
        raise FinalSummaryGateError("final score seed or namespace drifted")
    for name in (
        "architecture_id",
        "checkpoint_sha256",
        "publication_manifest_sha256",
        "calibration_map_sha256",
        "inference_commit",
    ):
        value = _scalar_string(payload[name], name=name)
        if name.endswith("_sha256") and len(value) != 64:
            raise FinalSummaryGateError(f"{name} is not a SHA-256")
        if name == "inference_commit" and len(value) != 40:
            raise FinalSummaryGateError("inference_commit is not full length")
    return payload


def _gate_passed(coverage: Mapping[str, Any]) -> bool:
    return all(
        bool(target["passed"])
        for item in coverage["marginal"].values()
        for target in item.values()
    ) and all(bool(item["passed"]) for item in coverage["joint"].values())


def summarize_score_group(
    payload: Mapping[str, np.ndarray],
    selection: np.ndarray,
    *,
    coverage_floor: float | None,
    joint_coverage_floor: float | None = None,
) -> Mapping[str, Any]:
    """Summarize one frozen case group without selecting a model seed."""

    mask = np.asarray(selection, dtype=bool)
    count = int(np.count_nonzero(mask))
    if mask.shape != payload["truth_log_density"].shape or count == 0:
        raise FinalSummaryGateError("final score group selection is empty or invalid")
    result: Dict[str, Any] = {
        "case_count": count,
        "mean_nlp_nat_per_target_dimension": float(
            np.mean(payload["nlp_nat_per_target_dimension"][mask])
        ),
        "mean_crps": [
            float(value) for value in np.mean(payload["crps"][mask], axis=0)
        ],
        "coverage": {"marginal": {}, "joint": {}},
        "mean_marginal_interval_width": {},
    }
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    marginal_coverage = coverage["marginal"]
    joint_coverage = coverage["joint"]
    assert isinstance(marginal_coverage, dict)
    assert isinstance(joint_coverage, dict)
    for level in LEVELS:
        suffix = f"{int(round(level * 100)):02d}"
        marginal: Dict[str, Any] = {}
        for target_index, target_name in enumerate(
            ("log_abs_mu_primary", "log_abs_mu_secondary")
        ):
            successes = int(
                np.count_nonzero(
                    payload[f"marginal_covered_{suffix}"][mask, target_index]
                )
            )
            if coverage_floor is None:
                marginal[target_name] = {
                    **coverage_gate(successes, count, level, 0.0),
                    "gate_applied": False,
                    "tolerance": None,
                    "passed": None,
                }
            else:
                marginal[target_name] = {
                    **coverage_gate(
                        successes,
                        count,
                        level,
                        coverage_floor,
                    ),
                    "gate_applied": True,
                }
        marginal_coverage[suffix] = marginal
        joint_successes = int(
            np.count_nonzero(payload[f"joint_covered_{suffix}"][mask])
        )
        if joint_coverage_floor is None:
            joint_coverage[suffix] = {
                **coverage_gate(joint_successes, count, level, 0.0),
                "gate_applied": False,
                "tolerance": None,
                "passed": None,
            }
        else:
            joint_coverage[suffix] = {
                **coverage_gate(
                    joint_successes,
                    count,
                    level,
                    joint_coverage_floor,
                ),
                "gate_applied": True,
            }
        result["mean_marginal_interval_width"][suffix] = [
            float(value)
            for value in np.mean(
                payload[f"marginal_interval_width_{suffix}"][mask],
                axis=0,
            )
        ]
    result["all_applied_coverage_gates_passed"] = (
        None if coverage_floor is None else _gate_passed(coverage)
    )
    return result


def summarize_one_seed(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    specifications: Mapping[str, FinalEvaluationNamespace],
) -> Mapping[str, Any]:
    """Create every frozen split/family/cell/stratum report for one seed."""

    if set(payloads) != set(specifications):
        raise FinalSummaryGateError("one seed is missing final score namespaces")
    groups: Dict[str, Mapping[str, Any]] = {}
    for namespace_id, specification in specifications.items():
        payload = payloads[namespace_id]
        full = np.ones(specification.accepted_count, dtype=bool)
        split = specification.split.value
        context = specification.diagnostic_context_id
        if split == "iid_test":
            groups["iid/overall"] = summarize_score_group(
                payload,
                full,
                coverage_floor=0.01,
                joint_coverage_floor=0.015,
            )
            for family in sorted(set(str(value) for value in payload["lens_families"])):
                groups[f"iid/family/{family}"] = summarize_score_group(
                    payload,
                    payload["lens_families"] == family,
                    coverage_floor=None,
                )
            for cell in sorted(set(str(value) for value in payload["em_cells"])):
                groups[f"iid/em_cell/{cell}"] = summarize_score_group(
                    payload,
                    payload["em_cells"] == cell,
                    coverage_floor=0.04,
                    joint_coverage_floor=0.04,
                )
        elif split == "balanced_tail_diagnostic":
            groups[f"balanced_tail/{context}"] = summarize_score_group(
                payload,
                full,
                coverage_floor=0.06,
                joint_coverage_floor=0.06,
            )
        else:
            groups[f"{split}/{context}"] = summarize_score_group(
                payload,
                full,
                coverage_floor=None,
            )
        if split != "iid_test":
            for family in sorted(set(str(value) for value in payload["lens_families"])):
                groups[f"{split}/{context}/family/{family}"] = summarize_score_group(
                    payload,
                    payload["lens_families"] == family,
                    coverage_floor=None,
                )
    iid_required = [
        value
        for name, value in groups.items()
        if name == "iid/overall" or name.startswith("iid/em_cell/")
    ]
    tail_required = [
        value for name, value in groups.items() if name.startswith("balanced_tail/")
    ]
    iid_passed = bool(iid_required) and all(
        value["all_applied_coverage_gates_passed"] is True
        for value in iid_required
    )
    tail_passed = len(tail_required) == 4 and all(
        value["all_applied_coverage_gates_passed"] is True
        for value in tail_required
    )
    return {
        "groups": groups,
        "primary_iid_coverage_gate_passed": iid_passed,
        "balanced_tail_coverage_gate_passed": tail_passed,
        "seed_claim_action": (
            "retain_preregistered_claim_domain"
            if iid_passed and tail_passed
            else "narrow_claims_and_report_failures"
        ),
    }


def _flatten_group_scalars(group: Mapping[str, Any]) -> Mapping[str, float]:
    values: Dict[str, float] = {
        "mean_nlp_nat_per_target_dimension": float(
            group["mean_nlp_nat_per_target_dimension"]
        ),
        "mean_crps_log_abs_mu_primary": float(group["mean_crps"][0]),
        "mean_crps_log_abs_mu_secondary": float(group["mean_crps"][1]),
    }
    coverage = group["coverage"]
    for suffix in ("50", "80", "90", "95"):
        values[f"marginal_coverage_{suffix}_primary"] = float(
            coverage["marginal"][suffix]["log_abs_mu_primary"]["coverage"]
        )
        values[f"marginal_coverage_{suffix}_secondary"] = float(
            coverage["marginal"][suffix]["log_abs_mu_secondary"]["coverage"]
        )
        values[f"joint_coverage_{suffix}"] = float(
            coverage["joint"][suffix]["coverage"]
        )
        values[f"mean_interval_width_{suffix}_primary"] = float(
            group["mean_marginal_interval_width"][suffix][0]
        )
        values[f"mean_interval_width_{suffix}_secondary"] = float(
            group["mean_marginal_interval_width"][suffix][1]
        )
    return values


def summarize_final_scores(
    payloads_by_seed: Mapping[
        int,
        Mapping[str, Mapping[str, np.ndarray]],
    ],
    specifications: Sequence[FinalEvaluationNamespace],
) -> Mapping[str, Any]:
    """Validate cross-seed identity and emit the final three-seed summary."""

    if set(payloads_by_seed) != set(MODEL_SEEDS):
        raise FinalSummaryGateError("final summary requires seeds 0, 1, and 2")
    specification_map = {item.namespace_id: item for item in specifications}
    if len(specification_map) != FINAL_NAMESPACE_COUNT:
        raise FinalSummaryGateError("final namespace set is incomplete")
    fingerprints: Dict[tuple[int, str], str] = {}
    architecture_ids: set[str] = set()
    for seed in MODEL_SEEDS:
        if set(payloads_by_seed[seed]) != set(specification_map):
            raise FinalSummaryGateError("final seed namespace set is incomplete")
        for namespace_id in sorted(specification_map):
            payload = payloads_by_seed[seed][namespace_id]
            identity = hashlib.sha256()
            for name in (
                "physical_system_ids",
                "truth",
                "lens_families",
                "em_cells",
                "splits",
                "diagnostic_context_ids",
            ):
                array = np.ascontiguousarray(payload[name])
                identity.update(name.encode())
                identity.update(str(array.dtype).encode())
                identity.update(json.dumps(array.shape).encode())
                identity.update(array.tobytes())
            fingerprints[(seed, namespace_id)] = identity.hexdigest()
            architecture_ids.add(
                _scalar_string(payload["architecture_id"], name="architecture_id")
            )
    if len(architecture_ids) != 1:
        raise FinalSummaryGateError("final score artifacts mix architectures")
    for namespace_id in specification_map:
        if len(
            {
                fingerprints[(seed, namespace_id)]
                for seed in MODEL_SEEDS
            }
        ) != 1:
            raise FinalSummaryGateError(
                "final score case identity or truth differs across seeds"
            )
    seed_summaries = {
        seed: summarize_one_seed(payloads_by_seed[seed], specification_map)
        for seed in MODEL_SEEDS
    }
    group_names = set(seed_summaries[0]["groups"])
    if any(set(seed_summaries[seed]["groups"]) != group_names for seed in MODEL_SEEDS):
        raise FinalSummaryGateError("final reporting groups differ across seeds")
    aggregate_groups = {
        name: aggregate_seed_metrics(
            {
                seed: _flatten_group_scalars(
                    seed_summaries[seed]["groups"][name]
                )
                for seed in MODEL_SEEDS
            }
        )
        for name in sorted(group_names)
    }
    all_iid = all(
        seed_summaries[seed]["primary_iid_coverage_gate_passed"]
        for seed in MODEL_SEEDS
    )
    all_tail = all(
        seed_summaries[seed]["balanced_tail_coverage_gate_passed"]
        for seed in MODEL_SEEDS
    )
    return {
        "status": "completed_preregistered_three_seed_final_score_summary",
        "locked_training_rung": TRAIN_131K_COUNT,
        "architecture_id": next(iter(architecture_ids)),
        "model_seeds": list(MODEL_SEEDS),
        "namespace_count": len(specification_map),
        "accepted_case_count": sum(
            item.accepted_count for item in specification_map.values()
        ),
        "score_artifact_count": SCORE_ARTIFACT_COUNT,
        "seed_summaries": {str(key): value for key, value in seed_summaries.items()},
        "aggregate_groups": aggregate_groups,
        "all_seed_iid_gate_passed": all_iid,
        "all_seed_balanced_tail_gate_passed": all_tail,
        "claim_action": (
            "retain_preregistered_claim_domain"
            if all_iid and all_tail
            else "narrow_claims_and_report_failures"
        ),
        "diagnostic_splits_used_for_retuning": False,
        "best_seed_selected": False,
        "threshold_changed_after_results": False,
        "manuscript_claim_finalized": False,
        "gwosc_gwtc_accessed": False,
    }


def _score_artifact_catalog(
    root: Path,
    inference_authorization_path: Path,
) -> Mapping[str, Any]:
    inference = load_yaml(inference_authorization_path)
    if inference.get("authorization_status") != FINAL_INFERENCE_AUTHORIZATION_STATUS:
        raise FinalSummaryGateError("final inference authorization is not complete")
    outputs = inference.get("score_outputs", {})
    if set(outputs) != {str(seed) for seed in MODEL_SEEDS}:
        raise FinalSummaryGateError("final inference score outputs omit a seed")
    expected = {item.namespace_id: item for item in _expected_namespaces(root)}
    catalog: Dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        values = outputs[str(seed)]
        if set(values) != set(expected):
            raise FinalSummaryGateError("final inference score outputs omit a namespace")
        seed_catalog: Dict[str, Any] = {}
        for namespace_id in sorted(expected):
            path = _project_path(
                Path(str(values[namespace_id])),
                name=f"seed-{seed} {namespace_id} final score",
                require_file=True,
            )
            summary_path = path.with_suffix(".summary.json")
            if not summary_path.is_file():
                raise FinalSummaryGateError("final score summary is absent")
            summary = _load_json(summary_path)
            if (
                summary.get("status") != "completed_final_score_extraction_only"
                or int(summary.get("model_seed", -1)) != seed
                or summary.get("namespace_id") != namespace_id
                or int(summary.get("case_count", -1))
                != expected[namespace_id].accepted_count
                or summary.get("score_artifact_sha256") != _sha256(path)
                or summary.get("posterior_draws_persisted") is not False
                or summary.get("calibration_refit") is not False
                or summary.get("model_retrained_or_tuned") is not False
                or summary.get("final_result_threshold_changed") is not False
                or summary.get("gwosc_gwtc_accessed") is not False
            ):
                raise FinalSummaryGateError("final score summary identity drifted")
            seed_catalog[namespace_id] = {
                "path": str(path),
                "sha256": _sha256(path),
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
            }
        catalog[str(seed)] = seed_catalog
    return catalog


def build_final_summary_release_packet(
    root: Path,
    *,
    implementation_commit: str,
    final_inference_authorization_path: Path,
    wheel_path: Path,
    exact_wheel_test_result_path: Path,
    environment_lock_path: Path,
    summary_output_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Build a non-authorizing packet after all 45 score artifacts exist."""

    root = root.resolve()
    load_final_summary_stack_contract(root)
    implementation_commit = _full_commit(
        implementation_commit,
        name="final-summary implementation commit",
    )
    _verify_checkout(root, implementation_commit)
    inference_path = _project_path(
        final_inference_authorization_path,
        name="final inference authorization",
        require_file=True,
    )
    summary_output = _project_path(
        summary_output_path,
        name="final summary output",
    )
    if summary_output.exists():
        raise FinalSummaryGateError("final summary output identity already exists")
    relative = _repo_relative(
        root,
        output_path,
        prefix=("results", "phase7"),
        name="final summary release packet",
    )
    if relative != "results/phase7/final_summary_release_packet.json":
        raise FinalSummaryGateError("final summary release-packet path changed")
    return {
        "status": RELEASE_STATUS,
        "authorization_created": False,
        "score_artifact_access_authorized": False,
        "implementation_commit": implementation_commit,
        "frozen_contracts": {
            "locked_training_rung": TRAIN_131K_COUNT,
            "model_seeds": list(MODEL_SEEDS),
            "final_analysis_hash": FINAL_ANALYSIS_HASH,
            "final_generation_commitment_sha256": FINAL_COMMITMENT_SHA256,
            "namespace_count": FINAL_NAMESPACE_COUNT,
            "accepted_case_count": FINAL_CASE_COUNT,
            "score_artifact_count": SCORE_ARTIFACT_COUNT,
        },
        "final_inference_authorization_path": str(inference_path),
        "final_inference_authorization_sha256": _sha256(inference_path),
        "score_artifacts": _score_artifact_catalog(root, inference_path),
        "immutable_execution": _immutable_inference(
            implementation_commit=implementation_commit,
            wheel_path=wheel_path,
            exact_wheel_test_result_path=exact_wheel_test_result_path,
            environment_lock_path=environment_lock_path,
        ),
        "summary_output_path": str(summary_output),
        "review_scope": dict(_REVIEW_SCOPE),
        "future_review_path": "results/phase7/final_summary_delegated_review.json",
        "future_authorization_path": (
            "configs/execution/phase7_final_summary_authorization.yaml"
        ),
        "release_packet_repository_path": relative,
    }


def build_final_summary_authorization(
    root: Path,
    *,
    release_packet_path: Path,
    delegated_review_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Create the exact runtime gate after a separate delegated review."""

    root = root.resolve()
    load_final_summary_stack_contract(root)
    packet = _load_json(release_packet_path)
    review = _load_json(delegated_review_path)
    packet_hash = _sha256(release_packet_path)
    scope = review.get("authorization_scope", {})
    if (
        packet.get("status") != RELEASE_STATUS
        or packet.get("authorization_created") is not False
        or packet.get("score_artifact_access_authorized") is not False
        or review.get("status") != REVIEW_STATUS
        or review.get("reviewed_release_packet_sha256") != packet_hash
        or review.get("reviewed_by")
        != "codex_as_delegated_scientific_and_engineering_reviewer"
        or scope != packet.get("review_scope")
        or scope != _REVIEW_SCOPE
    ):
        raise FinalSummaryGateError("delegated final-summary review is not exact")
    release_relative = _repo_relative(
        root,
        release_packet_path,
        prefix=("results", "phase7"),
        name="final summary release packet",
    )
    review_relative = _repo_relative(
        root,
        delegated_review_path,
        prefix=("results", "phase7"),
        name="final summary delegated review",
    )
    authorization_relative = _repo_relative(
        root,
        output_path,
        prefix=("configs", "execution"),
        name="final summary authorization",
    )
    if (
        release_relative != packet["release_packet_repository_path"]
        or review_relative != packet["future_review_path"]
        or authorization_relative != packet["future_authorization_path"]
    ):
        raise FinalSummaryGateError("final summary release evidence path changed")
    return {
        "phase": "7-final-score-summary",
        "authorization_status": AUTHORIZATION_STATUS,
        "authorized_by": str(review["reviewed_by"]),
        "authorization_date": str(review.get("review_date", "")),
        "authorization_basis": {
            "release_packet_path": release_relative,
            "release_packet_sha256": packet_hash,
            "delegated_review_path": review_relative,
            "delegated_review_sha256": _sha256(delegated_review_path),
        },
        "frozen_contracts": dict(packet["frozen_contracts"]),
        "final_inference_authorization_path": packet[
            "final_inference_authorization_path"
        ],
        "final_inference_authorization_sha256": packet[
            "final_inference_authorization_sha256"
        ],
        "score_artifacts": packet["score_artifacts"],
        "immutable_execution": packet["immutable_execution"],
        "summary_output_path": packet["summary_output_path"],
        "authorization": {
            key: value
            for key, value in _REVIEW_SCOPE.items()
            if key.endswith("_authorized")
        },
        "stop_after_final_score_summary": True,
    }


def _validate_runtime_authorization(
    root: Path,
    *,
    authorization_path: Path,
    output_path: Path,
    execution_commit: str,
) -> Mapping[str, Any]:
    authorization = load_yaml(authorization_path)
    flags = authorization.get("authorization", {})
    expected_flags = {
        key: value
        for key, value in _REVIEW_SCOPE.items()
        if key.endswith("_authorized")
    }
    if (
        authorization.get("authorization_status") != AUTHORIZATION_STATUS
        or flags != expected_flags
        or authorization.get("stop_after_final_score_summary") is not True
        or int(
            authorization.get("frozen_contracts", {}).get(
                "score_artifact_count",
                -1,
            )
        )
        != SCORE_ARTIFACT_COUNT
    ):
        raise FinalSummaryGateError("final score summary is not authorized")
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
        or output_path.resolve()
        != Path(str(authorization.get("summary_output_path", ""))).resolve()
        or output_path.exists()
    ):
        raise FinalSummaryGateError("final summary immutable execution changed")
    _verify_checkout(root, execution_commit)
    return authorization


def run_authorized_final_summary(
    root: Path,
    *,
    authorization_path: Path,
    output_path: Path,
    execution_commit: str,
) -> Mapping[str, Any]:
    """Open exactly 45 derived score artifacts and write one small summary."""

    authorization = _validate_runtime_authorization(
        root,
        authorization_path=authorization_path,
        output_path=output_path,
        execution_commit=execution_commit,
    )
    specifications = tuple(_expected_namespaces(root))
    specification_map = {item.namespace_id: item for item in specifications}
    catalog = authorization.get("score_artifacts", {})
    if set(catalog) != {str(seed) for seed in MODEL_SEEDS}:
        raise FinalSummaryGateError("final summary score catalog omits a seed")
    payloads: Dict[int, Dict[str, Mapping[str, np.ndarray]]] = {}
    for seed in MODEL_SEEDS:
        items = catalog[str(seed)]
        if set(items) != set(specification_map):
            raise FinalSummaryGateError("final summary score catalog omits a namespace")
        payloads[seed] = {}
        for namespace_id in sorted(specification_map):
            item = items[namespace_id]
            path = _project_path(
                Path(str(item.get("path", ""))),
                name="final score artifact",
                require_file=True,
            )
            summary_path = _project_path(
                Path(str(item.get("summary_path", ""))),
                name="final score summary",
                require_file=True,
            )
            if (
                _sha256(path) != item.get("sha256")
                or _sha256(summary_path) != item.get("summary_sha256")
            ):
                raise FinalSummaryGateError("final score artifact hash changed")
            payloads[seed][namespace_id] = load_validated_score_payload(
                path,
                specification=specification_map[namespace_id],
                model_seed=seed,
            )
    result = dict(summarize_final_scores(payloads, specifications))
    result.update(
        {
            "execution_commit": execution_commit,
            "authorization_sha256": _sha256(authorization_path),
            "final_records_opened": False,
            "checkpoints_opened": False,
            "calibration_maps_opened": False,
        }
    )
    _atomic_json(output_path, result)
    return result


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    load_final_summary_stack_contract(root)
    return {
        "status": "implementation_ready_final_score_access_closed",
        "locked_training_rung": TRAIN_131K_COUNT,
        "model_seeds": list(MODEL_SEEDS),
        "namespace_count": FINAL_NAMESPACE_COUNT,
        "accepted_case_count": FINAL_CASE_COUNT,
        "score_artifact_count": SCORE_ARTIFACT_COUNT,
        "final_analysis_hash": FINAL_ANALYSIS_HASH,
        "final_generation_commitment_sha256": FINAL_COMMITMENT_SHA256,
        "score_artifacts_opened": False,
        "final_records_opened": False,
        "checkpoints_opened": False,
        "calibration_maps_opened": False,
        "summary_executed": False,
        "manuscript_claim_finalized": False,
        "gwosc_gwtc_accessed": False,
    }
