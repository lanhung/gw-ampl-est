"""Pure, fail-closed contracts for the frozen final-evaluation analysis.

This module deliberately contains no dataset reader, checkpoint loader, optimizer, or
materialization entry point.  It freezes executable metric and counterfactual-family
semantics before final data can be generated or unsealed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from .calibration import wilson_interval
from .features import PreparedExample

FINAL_ANALYSIS_CONFIG = "configs/statistics/final_evaluation_analysis_preregistration.yaml"
FINAL_ANALYSIS_AUTHORIZATION = (
    "configs/execution/phase7_final_evaluation_analysis_stack_authorization.yaml"
)
FINAL_GENERATOR_CONFIG = "configs/data/phase4_final_evaluation.yaml"
FINAL_COMMITMENT = "results/phase4/final_evaluation_commitment.json"
FINAL_ANALYSIS_HASH = "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
PARENT_CALIBRATION_HASH = (
    "033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e"
)
FINAL_GENERATOR_CONFIG_HASH = (
    "11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66"
)
FINAL_COMMITMENT_SHA256 = (
    "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
)
FINAL_SPLIT_COUNTS = {
    "iid_test": 8192,
    "balanced_tail_diagnostic": 4096,
    "cross_family_misspecification_test": 2048,
    "parameter_region_ood": 2048,
    "waveform_mismatch_test": 2048,
    "psd_mismatch_test": 2048,
}
SEEDS = (0, 1, 2)
FAMILY_CONDITIONS = {
    "sie_external_shear": np.asarray((1.0, 0.0), dtype=np.float32),
    "epl_external_shear": np.asarray((0.0, 1.0), dtype=np.float32),
}


class FinalEvaluationGateError(ValueError):
    """Raised when a frozen final-evaluation contract is inconsistent."""


@dataclass(frozen=True)
class CrossFamilyAnalysisContext:
    """Executable analysis meaning of one immutable materialization context."""

    materialization_context_id: str
    analysis_cell_id: str
    truth_family: str
    inference_mode: str
    case_count: int


_CROSS_FAMILY_CONTEXTS = {
    "sie_truth_epl_assumed": CrossFamilyAnalysisContext(
        "sie_truth_epl_assumed",
        "sie_truth_epl_prior_marginalized_assumed",
        "sie_external_shear",
        "epl_family_condition_with_frozen_training_slope_prior_marginalized",
        512,
    ),
    "epl_truth_sie_assumed": CrossFamilyAnalysisContext(
        "epl_truth_sie_assumed",
        "epl_truth_sie_assumed",
        "epl_external_shear",
        "sie_family_condition",
        512,
    ),
    "sie_truth_family_marginalized": CrossFamilyAnalysisContext(
        "sie_truth_family_marginalized",
        "sie_truth_equal_family_mixture",
        "sie_external_shear",
        "equal_density_mixture_of_sie_and_epl_family_conditions",
        512,
    ),
    "epl_truth_family_marginalized": CrossFamilyAnalysisContext(
        "epl_truth_family_marginalized",
        "epl_truth_equal_family_mixture",
        "epl_external_shear",
        "equal_density_mixture_of_sie_and_epl_family_conditions",
        512,
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_evaluation_analysis_contract(
    root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the design-only RC.6 contract and prove every execution gate is closed."""

    config = load_yaml(root / FINAL_ANALYSIS_CONFIG)
    authorization = load_yaml(root / FINAL_ANALYSIS_AUTHORIZATION)
    if configuration_hash(config) != FINAL_ANALYSIS_HASH:
        raise FinalEvaluationGateError("final-analysis preregistration hash mismatch")
    if config.get("preregistration_version") != "1.1.0-rc.6":
        raise FinalEvaluationGateError("final-analysis preregistration version mismatch")
    parent = load_yaml(root / "configs/statistics/calibration_sbc_preregistration.yaml")
    if configuration_hash(parent) != PARENT_CALIBRATION_HASH:
        raise FinalEvaluationGateError("parent calibration/SBC contract hash mismatch")
    frozen = config.get("frozen_generation_commitment", {})
    if (
        configuration_hash(load_yaml(root / FINAL_GENERATOR_CONFIG))
        != FINAL_GENERATOR_CONFIG_HASH
        or _sha256(root / FINAL_COMMITMENT) != FINAL_COMMITMENT_SHA256
        or frozen.get("truth_generation_contexts_changed") is not False
        or frozen.get("namespace_ids_seeds_counts_and_distributions_changed") is not False
    ):
        raise FinalEvaluationGateError("frozen final-generation commitment drifted")
    if config.get("final_pool") != {
        "total_accepted_physical_systems": 20480,
        **FINAL_SPLIT_COUNTS,
        "group_disjoint_from_training_development_and_each_other": True,
        "final_data_cannot_affect_model_or_calibration": True,
    }:
        raise FinalEvaluationGateError("final-evaluation split counts or safety changed")
    inference = config.get("posterior_inference", {})
    if (
        tuple(inference.get("seeds", ())) != SEEDS
        or inference.get("posterior_draws_per_case") != 4096
        or not 0 < int(inference.get("maximum_draw_microbatch", 0)) <= 512
        or 4096 % int(inference["maximum_draw_microbatch"]) != 0
        or inference.get("pooling_seed_draws_for_case_level_regions") is not False
    ):
        raise FinalEvaluationGateError("posterior inference contract drifted")
    cells = config.get("cross_family_executable_semantics", {}).get("cells", ())
    if sum(int(cell.get("cases", 0)) for cell in cells) != 2048:
        raise FinalEvaluationGateError("cross-family cell arithmetic drifted")
    for cell in cells:
        resolved = cross_family_analysis_context(str(cell.get("materialization_context_id")))
        if (
            cell.get("id") != resolved.analysis_cell_id
            or cell.get("truth") != resolved.truth_family
            or cell.get("inference") != resolved.inference_mode
            or cell.get("cases") != resolved.case_count
        ):
            raise FinalEvaluationGateError("cross-family executable mapping drifted")
    if any(value is not False for value in config.get("execution", {}).values()):
        raise FinalEvaluationGateError("RC.6 must keep every execution flag false")
    if authorization.get("authorization_status") != (
        "authorized_preregistration_and_pure_implementation_only"
    ):
        raise FinalEvaluationGateError("pure implementation authorization is absent")
    if authorization.get("frozen_addendum", {}).get("canonical_hash") != FINAL_ANALYSIS_HASH:
        raise FinalEvaluationGateError("authorization references the wrong RC.6 hash")
    flags = authorization.get("authorization", {})
    for name in (
        "downstream_preregistration_authorized",
        "pure_metric_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    ):
        if flags.get(name) is not True:
            raise FinalEvaluationGateError("pure implementation authorization is incomplete")
    forbidden = set(flags) - {
        "downstream_preregistration_authorized",
        "pure_metric_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not False for name in forbidden):
        raise FinalEvaluationGateError("implementation authorization opened an execution gate")
    return config, authorization


def cross_family_analysis_context(materialization_context_id: str) -> CrossFamilyAnalysisContext:
    """Map a committed generator namespace to its executable RC.6 analysis semantics."""

    try:
        return _CROSS_FAMILY_CONTEXTS[materialization_context_id]
    except KeyError as error:
        raise FinalEvaluationGateError("unknown cross-family materialization context") from error


def condition_on_lens_family(example: PreparedExample, family: str) -> PreparedExample:
    """Return a counterfactual copy changing only the deployable family condition."""

    if family not in FAMILY_CONDITIONS:
        raise FinalEvaluationGateError("unsupported lens-family condition")
    return replace(
        example,
        lens_family_condition=FAMILY_CONDITIONS[family].copy(),
    )


def equal_family_log_mixture(
    sie_log_probability: np.ndarray, epl_log_probability: np.ndarray
) -> np.ndarray:
    """Evaluate log(0.5 p_SIE + 0.5 p_EPL) stably, case by case."""

    sie = np.asarray(sie_log_probability, dtype=np.float64)
    epl = np.asarray(epl_log_probability, dtype=np.float64)
    if sie.shape != epl.shape or not np.all(np.isfinite(sie)) or not np.all(np.isfinite(epl)):
        raise FinalEvaluationGateError("family log densities must be finite and shape matched")
    return np.logaddexp(sie, epl) - math.log(2.0)


def equal_family_draw_mixture(sie_draws: np.ndarray, epl_draws: np.ndarray) -> np.ndarray:
    """Create exactly 4,096 equal-family draws without mixing model seeds."""

    sie = np.asarray(sie_draws)
    epl = np.asarray(epl_draws)
    expected_tail = (2048, 2)
    if (
        sie.ndim != 3
        or epl.shape != sie.shape
        or sie.shape[1:] != expected_tail
        or not np.all(np.isfinite(sie))
        or not np.all(np.isfinite(epl))
    ):
        raise FinalEvaluationGateError("family mixture requires cases by 2048 by 2 draws")
    result = np.empty((len(sie), 4096, 2), dtype=np.result_type(sie, epl))
    result[:, 0::2, :] = sie
    result[:, 1::2, :] = epl
    return result


def gw_only_example(example: PreparedExample) -> PreparedExample:
    """Remove every EM input while retaining GW strain, timing, and family condition."""

    scalars = np.asarray(example.scalar_features).copy()
    scalar_mask = np.asarray(example.scalar_mask).copy()
    scalars[2:] = 0.0
    scalar_mask[2:] = 0.0
    return replace(
        example,
        astrometry_items=np.zeros_like(example.astrometry_items),
        astrometry_mask=np.zeros_like(example.astrometry_mask),
        scalar_features=scalars,
        scalar_mask=scalar_mask,
        modality_mask=np.zeros_like(example.modality_mask),
    )


def em_only_example(example: PreparedExample) -> PreparedExample:
    """Remove GW strain, detector availability, and observed GW timing."""

    scalars = np.asarray(example.scalar_features).copy()
    scalar_mask = np.asarray(example.scalar_mask).copy()
    scalars[:2] = 0.0
    scalar_mask[:2] = 0.0
    return replace(
        example,
        gw_strain=np.zeros_like(example.gw_strain),
        detector_mask=np.zeros_like(example.detector_mask),
        scalar_features=scalars,
        scalar_mask=scalar_mask,
    )


def coverage_gate(
    successes: int,
    total: int,
    nominal_level: float,
    absolute_floor: float,
) -> Mapping[str, float | int | bool]:
    """Apply max(floor, three binomial standard errors) to a raw coverage count."""

    if (
        total <= 0
        or successes < 0
        or successes > total
        or not 0.0 < nominal_level < 1.0
        or absolute_floor < 0.0
    ):
        raise FinalEvaluationGateError("coverage gate inputs are invalid")
    standard_error = math.sqrt(nominal_level * (1.0 - nominal_level) / total)
    tolerance = max(float(absolute_floor), 3.0 * standard_error)
    coverage = successes / total
    error = abs(coverage - nominal_level)
    wilson_lower, wilson_upper = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "nominal_level": nominal_level,
        "coverage": coverage,
        "wilson_95_lower": wilson_lower,
        "wilson_95_upper": wilson_upper,
        "absolute_error": error,
        "three_binomial_standard_errors": 3.0 * standard_error,
        "tolerance": tolerance,
        "passed": error <= tolerance,
    }


def aggregate_seed_metrics(
    metrics_by_seed: Mapping[int, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    """Aggregate all three seeds without permitting best-seed selection."""

    if set(metrics_by_seed) != set(SEEDS):
        raise FinalEvaluationGateError("final evaluation requires seeds 0, 1, and 2")
    names = set(metrics_by_seed[0])
    if not names or any(set(metrics_by_seed[seed]) != names for seed in SEEDS):
        raise FinalEvaluationGateError("seed metric fields are missing or inconsistent")
    result: Dict[str, Mapping[str, float]] = {}
    for name in sorted(names):
        values = np.asarray([metrics_by_seed[seed][name] for seed in SEEDS], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FinalEvaluationGateError("seed metric is nonfinite")
        result[name] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
        }
    return result


def validate_final_split_ids(ids_by_split: Mapping[str, Sequence[str]]) -> Mapping[str, int]:
    """Prove exact final counts and cross-split physical-system disjointness."""

    if set(ids_by_split) != set(FINAL_SPLIT_COUNTS):
        raise FinalEvaluationGateError("final split names are incomplete or unexpected")
    seen: set[str] = set()
    counts: Dict[str, int] = {}
    for split, expected in FINAL_SPLIT_COUNTS.items():
        values = tuple(str(value) for value in ids_by_split[split])
        if (
            len(values) != expected
            or len(set(values)) != expected
            or any(not value for value in values)
        ):
            raise FinalEvaluationGateError(f"{split} IDs violate exact count or uniqueness")
        overlap = seen.intersection(values)
        if overlap:
            raise FinalEvaluationGateError("final physical-system IDs overlap across splits")
        seen.update(values)
        counts[split] = len(values)
    counts["total"] = len(seen)
    return counts


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    """Return an execution-disabled plan without resolving data or checkpoint identities."""

    config, _ = load_final_evaluation_analysis_contract(root)
    return {
        "status": "implementation_ready_execution_closed",
        "preregistration_version": config["preregistration_version"],
        "preregistration_hash": FINAL_ANALYSIS_HASH,
        "final_case_count": sum(FINAL_SPLIT_COUNTS.values()),
        "model_seeds": list(SEEDS),
        "posterior_draws_per_case": 4096,
        "official_final_data_identity": None,
        "selected_checkpoint_identities": None,
        "calibration_map_identities": None,
        "final_data_accessed": False,
        "checkpoint_accessed": False,
        "calibration_refit": False,
        "metric_computed": False,
        "ablation_trained": False,
        "baseline_executed": False,
        "gwosc_gwtc_accessed": False,
    }


def write_dry_run_plan(root: Path, output: Path) -> None:
    """Write the small design-only plan for review."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dry_run_plan(root), indent=2, sort_keys=True) + "\n")
