"""Frozen calibration and IID-comparison semantics for input ablations.

This module is deliberately side-effect free.  It defines the adapters and
statistics that a later exact execution release may use, but it cannot open a
checkpoint, unseal final data, fit a scientific map, or write an artifact by
itself.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from ..schema import SplitName
from .ablations import ABLATION_VIEWS, apply_ablation_view
from .calibration import LEVELS, fit_region_calibration, wilson_interval
from .contracts import TrainingGateError
from .data import (
    CalibrationSBCCase,
    PublishedStageADataset,
    StandardizedCalibrationSBCDataset,
)
from .final_inference import (
    FinalEvaluationCase,
    StandardizedFinalNamespaceDataset,
)

ABLATION_EVALUATION_CONFIG = (
    "configs/statistics/ablation_calibration_iid_preregistration.yaml"
)
ABLATION_EVALUATION_CONFIG_HASH = (
    "219160f67030bad745b0a4573d78d02f9d0db7536a6490c907196e8570647c9a"
)
ABLATION_EVALUATION_AUTHORIZATION = (
    "configs/execution/phase7_ablation_calibration_iid_stack_authorization.yaml"
)
MODEL_SEEDS = (0, 1, 2)
CALIBRATION_COUNT = 4096
IID_COUNT = 8192
POSTERIOR_DRAW_COUNT = 4096
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED_DOMAIN = "final_iid_ablation_paired_bootstrap_v1"


class AblationEvaluationGateError(TrainingGateError):
    """Raised when an ablation evaluation crosses the frozen RC.8 boundary."""


def load_ablation_evaluation_contract(root: Any) -> Mapping[str, Any]:
    """Validate RC.8 and its implementation-only authorization."""

    config = load_yaml(root / ABLATION_EVALUATION_CONFIG)
    if (
        config.get("preregistration_version") != "1.1.0-rc.8"
        or configuration_hash(config) != ABLATION_EVALUATION_CONFIG_HASH
        or config.get("status") != "design_frozen_execution_disabled"
    ):
        raise AblationEvaluationGateError("ablation calibration/IID RC.8 drifted")
    execution = config.get("execution", {})
    if not isinstance(execution, dict) or any(value is not False for value in execution.values()):
        raise AblationEvaluationGateError("RC.8 must keep every execution flag false")

    authorization = load_yaml(root / ABLATION_EVALUATION_AUTHORIZATION)
    if authorization.get("authorization_status") != (
        "authorized_preregistration_and_pure_implementation_only"
    ):
        raise AblationEvaluationGateError("ablation evaluation implementation gate is absent")
    frozen = authorization.get("frozen_addendum", {})
    if (
        frozen.get("path") != ABLATION_EVALUATION_CONFIG
        or frozen.get("version") != "1.1.0-rc.8"
        or frozen.get("canonical_hash") != ABLATION_EVALUATION_CONFIG_HASH
    ):
        raise AblationEvaluationGateError("implementation gate does not bind RC.8")
    flags = authorization.get("authorization", {})
    allowed_true = {
        "downstream_preregistration_authorized",
        "pure_ablation_score_adapter_implementation_authorized",
        "pure_ablation_calibration_adapter_implementation_authorized",
        "pure_iid_comparison_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed_true):
        raise AblationEvaluationGateError("pure implementation permission is incomplete")
    if any(value is not False for name, value in flags.items() if name not in allowed_true):
        raise AblationEvaluationGateError("implementation gate opened scientific execution")
    return {"config": config, "authorization": authorization}


class AblatedCalibrationDataset:
    """Apply one frozen view after primary standardization on calibration-fit."""

    def __init__(
        self,
        dataset: PublishedStageADataset,
        standardizer: Any,
        view: str,
    ) -> None:
        if dataset.expected_split is not SplitName.CALIBRATION_FIT:
            raise AblationEvaluationGateError(
                "ablation calibration adapter requires calibration_fit"
            )
        if view not in ABLATION_VIEWS:
            raise AblationEvaluationGateError("unknown ablation view")
        self.dataset = StandardizedCalibrationSBCDataset(dataset, standardizer)
        self.view = view

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> CalibrationSBCCase:
        case = self.dataset[index]
        return CalibrationSBCCase(
            example=apply_ablation_view(case.example, self.view),
            em_cell=case.em_cell,
        )


class AblatedIIDDataset:
    """Apply one frozen view to the primary standardized IID publication."""

    def __init__(self, dataset: StandardizedFinalNamespaceDataset, view: str) -> None:
        if (
            dataset.dataset.publication.specification.split
            is not SplitName.IID_TEST
        ):
            raise AblationEvaluationGateError(
                "ablation final adapter is restricted to IID"
            )
        if view not in ABLATION_VIEWS:
            raise AblationEvaluationGateError("unknown ablation view")
        self.dataset = dataset
        self.view = view

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> FinalEvaluationCase:
        case = self.dataset[index]
        return FinalEvaluationCase(
            example=apply_ablation_view(case.example, self.view),
            split=case.split,
            diagnostic_context_id=case.diagnostic_context_id,
        )


def _string_vector(value: Any, *, name: str, count: int) -> Tuple[str, ...]:
    array = np.asarray(value)
    result = tuple(str(item) for item in array.tolist())
    if (
        array.ndim != 1
        or len(result) != count
        or len(set(result)) != count
        or any(not item for item in result)
    ):
        raise AblationEvaluationGateError(f"{name} is not a unique length-{count} vector")
    return result


def fit_ablation_calibration_map(
    score: Mapping[str, np.ndarray],
    *,
    view: str,
    model_seed: int,
    checkpoint_sha256: str,
    primary_calibration_case_ids: Sequence[str],
    expected_count: int = CALIBRATION_COUNT,
    expected_per_cell: int | None = None,
) -> Mapping[str, Any]:
    """Fit one view/seed map on exactly the primary calibration-fit cases."""

    if view not in ABLATION_VIEWS or model_seed not in MODEL_SEEDS:
        raise AblationEvaluationGateError("ablation calibration identity is invalid")
    if len(checkpoint_sha256) != 64:
        raise AblationEvaluationGateError("ablation checkpoint hash is invalid")
    identifiers = _string_vector(
        score.get("physical_system_ids"), name="calibration IDs", count=expected_count
    )
    expected_ids = tuple(str(value) for value in primary_calibration_case_ids)
    if identifiers != expected_ids or len(set(expected_ids)) != expected_count:
        raise AblationEvaluationGateError(
            "ablation and primary calibration cases are not byte-order identical"
        )
    marginal = np.asarray(score.get("marginal_scores"), dtype=np.float64)
    joint = np.asarray(score.get("joint_scores"), dtype=np.float64)
    cells = tuple(str(value) for value in np.asarray(score.get("em_cells")).tolist())
    if (
        marginal.shape != (expected_count, 2)
        or joint.shape != (expected_count,)
        or len(cells) != expected_count
        or not np.all(np.isfinite(marginal))
        or not np.all(np.isfinite(joint))
    ):
        raise AblationEvaluationGateError("ablation calibration scores are invalid")
    per_cell = expected_per_cell
    if per_cell is None:
        if expected_count % 8:
            raise AblationEvaluationGateError("calibration count is not divisible by 8")
        per_cell = expected_count // 8
    fitted = dict(
        fit_region_calibration(
            marginal,
            joint,
            cells,
            expected_total=expected_count,
            expected_per_cell=per_cell,
        )
    )
    fitted["ablation_identity"] = {
        "view": view,
        "model_seed": model_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_case_ids_sha256": hashlib.sha256(
            "\n".join(identifiers).encode()
        ).hexdigest(),
        "same_cases_as_primary_calibration": True,
        "map_shared_across_views_or_seeds": False,
        "primary_calibration_map_reused": False,
        "iid_used_to_fit_map": False,
    }
    return fitted


def validate_matching_ablation_map(
    calibration_map: Mapping[str, Any],
    *,
    view: str,
    model_seed: int,
    checkpoint_sha256: str,
) -> None:
    """Reject a primary, pooled, wrong-view or wrong-seed calibration map."""

    identity = calibration_map.get("ablation_identity", {})
    if (
        not isinstance(identity, dict)
        or identity.get("view") != view
        or int(identity.get("model_seed", -1)) != model_seed
        or identity.get("checkpoint_sha256") != checkpoint_sha256
        or identity.get("same_cases_as_primary_calibration") is not True
        or identity.get("map_shared_across_views_or_seeds") is not False
        or identity.get("primary_calibration_map_reused") is not False
        or identity.get("iid_used_to_fit_map") is not False
        or calibration_map.get("status")
        != "fitted_split_conformal_region_level_maps"
    ):
        raise AblationEvaluationGateError(
            "calibration map does not match the ablation view, seed and checkpoint"
        )


def _coverage_counts(values: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=bool)
    if array.ndim not in (1, 2) or len(array) == 0:
        raise AblationEvaluationGateError("coverage array has invalid shape")
    columns = array[:, None] if array.ndim == 1 else array
    counts = [int(np.count_nonzero(columns[:, index])) for index in range(columns.shape[1])]
    return {
        "case_count": len(array),
        "covered_count": counts,
        "coverage": [value / len(array) for value in counts],
        "wilson_95": [list(wilson_interval(value, len(array))) for value in counts],
    }


def summarize_ablation_iid_scores(
    score: Mapping[str, np.ndarray],
    *,
    view: str,
    model_seed: int,
    expected_count: int = IID_COUNT,
) -> Mapping[str, Any]:
    """Summarize one immutable ablation IID score artifact without selection."""

    if view not in ABLATION_VIEWS or model_seed not in MODEL_SEEDS:
        raise AblationEvaluationGateError("ablation IID identity is invalid")
    identifiers = _string_vector(
        score.get("physical_system_ids"), name="IID IDs", count=expected_count
    )
    families = tuple(str(value) for value in np.asarray(score.get("lens_families")).tolist())
    cells = tuple(str(value) for value in np.asarray(score.get("em_cells")).tolist())
    splits = tuple(str(value) for value in np.asarray(score.get("splits")).tolist())
    nlp = np.asarray(score.get("nlp_nat_per_target_dimension"), dtype=np.float64)
    crps = np.asarray(score.get("crps"), dtype=np.float64)
    if (
        len(families) != expected_count
        or len(cells) != expected_count
        or set(splits) != {SplitName.IID_TEST.value}
        or nlp.shape != (expected_count,)
        or crps.shape != (expected_count, 2)
        or not np.all(np.isfinite(nlp))
        or not np.all(np.isfinite(crps))
        or set(families) != {"sie_external_shear", "epl_external_shear"}
        or len(set(cells)) != 8
    ):
        raise AblationEvaluationGateError("IID score artifact violates RC.8")

    def summarize_indices(indices: np.ndarray) -> Mapping[str, Any]:
        result: Dict[str, Any] = {
            "case_count": int(len(indices)),
            "mean_nlp_nat_per_target_dimension": float(np.mean(nlp[indices])),
            "mean_crps": [
                float(value) for value in np.mean(crps[indices], axis=0)
            ],
            "coverage": {},
        }
        for level in LEVELS:
            suffix = f"{int(round(level * 100)):02d}"
            marginal = np.asarray(score.get(f"marginal_covered_{suffix}"))[indices]
            joint = np.asarray(score.get(f"joint_covered_{suffix}"))[indices]
            width = np.asarray(score.get(f"marginal_interval_width_{suffix}"))[indices]
            if (
                marginal.shape != (len(indices), 2)
                or joint.shape != (len(indices),)
                or width.shape != (len(indices), 2)
                or not np.all(np.isfinite(width))
            ):
                raise AblationEvaluationGateError("IID calibrated metric shape is invalid")
            result["coverage"][f"{level:.2f}"] = {
                "marginal": _coverage_counts(marginal),
                "joint": _coverage_counts(joint),
                "mean_marginal_interval_width": [
                    float(value) for value in np.mean(width, axis=0)
                ],
            }
        return result

    family_array = np.asarray(families, dtype=np.str_)
    cell_array = np.asarray(cells, dtype=np.str_)
    all_indices = np.arange(expected_count)
    return {
        "status": "completed_descriptive_ablation_iid_summary",
        "view": view,
        "model_seed": model_seed,
        "case_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
        "overall": summarize_indices(all_indices),
        "lens_families": {
            family: summarize_indices(np.flatnonzero(family_array == family))
            for family in sorted(set(families))
        },
        "em_cells": {
            cell: summarize_indices(np.flatnonzero(cell_array == cell))
            for cell in sorted(set(cells))
        },
        "best_seed_selected": False,
        "result_can_trigger_retraining_or_tuning": False,
        "sbc_executed": False,
        "ood_or_mismatch_executed": False,
    }


def _aligned_metric(
    primary: Mapping[str, np.ndarray],
    ablation: Mapping[str, np.ndarray],
    key: str,
    *,
    expected_count: int,
) -> np.ndarray:
    left = np.asarray(primary.get(key))
    right = np.asarray(ablation.get(key))
    if left.shape != right.shape or left.shape[0] != expected_count:
        raise AblationEvaluationGateError(f"paired IID metric {key} is misaligned")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise AblationEvaluationGateError(f"paired IID metric {key} is nonfinite")
    return np.asarray(right - left, dtype=np.float64)


def _bootstrap_mean_difference(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    matrix = values[:, None] if values.ndim == 1 else values
    if matrix.ndim != 2 or len(matrix) == 0 or replicates != BOOTSTRAP_REPLICATES:
        raise AblationEvaluationGateError("paired bootstrap contract changed")
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        estimates[start:stop] = np.mean(matrix[indices], axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975), axis=0)
    point = np.mean(matrix, axis=0)
    return {
        "replicates": replicates,
        "point_estimate": [float(value) for value in point],
        "lower_95": [float(value) for value in lower],
        "upper_95": [float(value) for value in upper],
    }


def paired_ablation_iid_comparison(
    primary: Mapping[str, np.ndarray],
    ablation: Mapping[str, np.ndarray],
    *,
    view: str,
    model_seed: int,
    expected_count: int = IID_COUNT,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> Mapping[str, Any]:
    """Compare one ablation to the same-seed primary on identical IID cases."""

    if view not in ABLATION_VIEWS or model_seed not in MODEL_SEEDS:
        raise AblationEvaluationGateError("paired IID comparison identity is invalid")
    primary_ids = _string_vector(
        primary.get("physical_system_ids"), name="primary IID IDs", count=expected_count
    )
    ablation_ids = _string_vector(
        ablation.get("physical_system_ids"), name="ablation IID IDs", count=expected_count
    )
    if primary_ids != ablation_ids:
        raise AblationEvaluationGateError(
            "primary and ablation IID physical-system ordering differs"
        )
    if not np.array_equal(
        np.asarray(primary.get("truth")), np.asarray(ablation.get("truth"))
    ):
        raise AblationEvaluationGateError("primary and ablation IID truths differ")
    for key in ("lens_families", "em_cells", "splits", "diagnostic_context_ids"):
        if not np.array_equal(
            np.asarray(primary.get(key)), np.asarray(ablation.get(key))
        ):
            raise AblationEvaluationGateError(f"primary and ablation IID {key} differ")

    differences: Dict[str, np.ndarray] = {
        "nlp_nat_per_target_dimension": _aligned_metric(
            primary,
            ablation,
            "nlp_nat_per_target_dimension",
            expected_count=expected_count,
        ),
        "crps": _aligned_metric(
            primary, ablation, "crps", expected_count=expected_count
        ),
    }
    for level in LEVELS:
        suffix = f"{int(round(level * 100)):02d}"
        key = f"marginal_interval_width_{suffix}"
        differences[key] = _aligned_metric(
            primary, ablation, key, expected_count=expected_count
        )
    bootstrap = {}
    for metric, values in differences.items():
        seed_payload = (
            f"{BOOTSTRAP_SEED_DOMAIN}\0{view}\0{model_seed}\0{metric}".encode()
        )
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        bootstrap[metric] = _bootstrap_mean_difference(
            values, seed=seed, replicates=replicates
        )
    return {
        "status": "completed_descriptive_paired_iid_ablation_comparison",
        "view": view,
        "model_seed": model_seed,
        "case_count": expected_count,
        "case_ids_sha256": hashlib.sha256("\n".join(primary_ids).encode()).hexdigest(),
        "difference_direction": "ablation_minus_same_seed_primary",
        "paired_bootstrap": bootstrap,
        "confidence_level": 0.95,
        "resampling_unit": "physical_system_id",
        "best_seed_selected": False,
        "superiority_gate": None,
        "result_can_trigger_retraining_or_tuning": False,
    }


def aggregate_ablation_iid_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate all six point estimates without selecting or pooling seeds."""

    by_identity: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    common_case_hash: str | None = None
    for comparison in comparisons:
        view = str(comparison.get("view", ""))
        seed = int(comparison.get("model_seed", -1))
        identity = (view, seed)
        case_hash = str(comparison.get("case_ids_sha256", ""))
        if (
            comparison.get("status")
            != "completed_descriptive_paired_iid_ablation_comparison"
            or view not in ABLATION_VIEWS
            or seed not in MODEL_SEEDS
            or identity in by_identity
            or comparison.get("best_seed_selected") is not False
            or comparison.get("superiority_gate") is not None
            or comparison.get("result_can_trigger_retraining_or_tuning") is not False
            or len(case_hash) != 64
        ):
            raise AblationEvaluationGateError(
                "ablation IID aggregate received an incomplete comparison"
            )
        if common_case_hash is None:
            common_case_hash = case_hash
        elif case_hash != common_case_hash:
            raise AblationEvaluationGateError(
                "ablation IID comparisons used different physical systems"
            )
        by_identity[identity] = comparison
    expected = {
        (view, seed) for view in ABLATION_VIEWS for seed in MODEL_SEEDS
    }
    if set(by_identity) != expected:
        raise AblationEvaluationGateError(
            "ablation IID aggregate requires both views and all three seeds"
        )
    views: Dict[str, Any] = {}
    for view in ABLATION_VIEWS:
        seed_results = [by_identity[(view, seed)] for seed in MODEL_SEEDS]
        metric_names = set(seed_results[0]["paired_bootstrap"])
        if any(set(item["paired_bootstrap"]) != metric_names for item in seed_results):
            raise AblationEvaluationGateError(
                "ablation IID seed comparisons report different metrics"
            )
        aggregates: Dict[str, Any] = {}
        for metric in sorted(metric_names):
            point_estimates = np.asarray(
                [
                    item["paired_bootstrap"][metric]["point_estimate"]
                    for item in seed_results
                ],
                dtype=np.float64,
            )
            if (
                point_estimates.ndim != 2
                or point_estimates.shape[0] != len(MODEL_SEEDS)
                or not np.all(np.isfinite(point_estimates))
            ):
                raise AblationEvaluationGateError(
                    "ablation IID seed point estimates are invalid"
                )
            aggregates[metric] = {
                "mean_across_seeds": [
                    float(value) for value in np.mean(point_estimates, axis=0)
                ],
                "sample_standard_deviation_across_seeds": [
                    float(value)
                    for value in np.std(point_estimates, axis=0, ddof=1)
                ],
            }
        views[view] = {
            "seeds": list(MODEL_SEEDS),
            "seed_results": seed_results,
            "aggregate_point_estimates": aggregates,
        }
    return {
        "status": "completed_all_seed_descriptive_ablation_iid_comparison",
        "comparison_count": len(comparisons),
        "case_ids_sha256": common_case_hash,
        "views": views,
        "best_seed_selected": False,
        "seed_pooling_used_for_model_selection": False,
        "result_can_trigger_retraining_or_tuning": False,
    }


def dry_run_ablation_evaluation_plan(root: Any) -> Mapping[str, Any]:
    """Return the complete future workload while proving execution is closed."""

    load_ablation_evaluation_contract(root)
    return {
        "status": "implementation_ready_execution_disabled",
        "views": list(ABLATION_VIEWS),
        "model_seeds": list(MODEL_SEEDS),
        "calibration_map_count": len(ABLATION_VIEWS) * len(MODEL_SEEDS),
        "calibration_case_count_per_map": CALIBRATION_COUNT,
        "iid_score_artifact_count": len(ABLATION_VIEWS) * len(MODEL_SEEDS),
        "iid_case_count_per_artifact": IID_COUNT,
        "paired_comparison_count": len(ABLATION_VIEWS) * len(MODEL_SEEDS),
        "best_seed_selection_authorized": False,
        "ablation_sbc_authorized": False,
        "ablation_ood_or_mismatch_authorized": False,
        "scientific_execution_authorized": False,
    }
