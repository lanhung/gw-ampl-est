"""Frozen split-conformal credible-region calibration and independent SBC."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

LEVELS = (0.50, 0.80, 0.90, 0.95)
TARGETS = ("log_abs_mu_primary", "log_abs_mu_secondary")
SBC_STATISTICS = (
    "log_abs_mu_primary",
    "log_abs_mu_secondary",
    "log_abs_mu_sum",
    "log_abs_mu_difference",
    "joint_log_density_rank",
)


def _finite(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def empirical_pit_scores(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return two-sided central scores from empirical posterior midranks."""

    draws = _finite(samples, name="posterior samples")
    targets = _finite(truth, name="calibration truth")
    if draws.ndim != 3 or draws.shape[2] != 2 or targets.shape != (len(draws), 2):
        raise ValueError("calibration samples must be cases by draws by two targets")
    less = np.sum(draws < targets[:, None, :], axis=1)
    equal = np.sum(draws == targets[:, None, :], axis=1)
    pit = (less + 0.5 * equal) / draws.shape[1]
    return 2.0 * np.abs(pit - 0.5)


def joint_hpd_scores(
    posterior_draw_log_density: np.ndarray, truth_log_density: np.ndarray
) -> np.ndarray:
    """Return raw HPD mass needed to include each calibration truth."""

    draws = _finite(posterior_draw_log_density, name="draw log density")
    truth = _finite(truth_log_density, name="truth log density")
    if draws.ndim != 2 or truth.shape != (len(draws),):
        raise ValueError("joint HPD inputs have incompatible shapes")
    return np.mean(draws >= truth[:, None], axis=1)


def conformal_order_statistic(
    scores: np.ndarray, level: float
) -> Mapping[str, float | int]:
    """Finite-sample split-conformal threshold with no interpolation."""

    values = _finite(scores, name="conformal scores")
    if values.ndim != 1 or not len(values) or not 0.0 < level < 1.0:
        raise ValueError("conformal quantile requires a vector and level in (0,1)")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("credible-region scores must lie in [0,1]")
    rank = min(len(values), math.ceil((len(values) + 1) * level))
    threshold = float(np.partition(values, rank - 1)[rank - 1])
    return {
        "nominal_level": float(level),
        "calibration_count": int(len(values)),
        "one_based_order_statistic": int(rank),
        "raw_region_mass_threshold": threshold,
    }


def _fit_one_map(
    marginal_scores: np.ndarray, joint_scores: np.ndarray
) -> Mapping[str, Any]:
    return {
        "marginal": {
            target: {
                f"{level:.2f}": conformal_order_statistic(
                    marginal_scores[:, index], level
                )
                for level in LEVELS
            }
            for index, target in enumerate(TARGETS)
        },
        "joint": {
            f"{level:.2f}": conformal_order_statistic(joint_scores, level)
            for level in LEVELS
        },
    }


def fit_region_calibration(
    marginal_scores: np.ndarray,
    joint_scores: np.ndarray,
    em_cells: Sequence[str],
    *,
    expected_total: int = 4096,
    expected_per_cell: int = 512,
) -> Mapping[str, Any]:
    """Fit separate global and discrete EM-cell conformal level maps."""

    marginal = _finite(marginal_scores, name="marginal calibration scores")
    joint = _finite(joint_scores, name="joint calibration scores")
    cells = np.asarray(tuple(em_cells), dtype=object)
    if marginal.shape != (expected_total, 2) or joint.shape != (expected_total,):
        raise ValueError("calibration-fit split has the wrong score dimensions or count")
    if cells.shape != (expected_total,) or any(not str(value) for value in cells):
        raise ValueError("calibration-fit EM-cell labels are invalid")
    unique = tuple(sorted(str(value) for value in set(cells)))
    if len(unique) != 8:
        raise ValueError("calibration-fit requires exactly eight EM cells")
    cell_maps = {}
    for cell in unique:
        selected = cells == cell
        if int(np.count_nonzero(selected)) != expected_per_cell:
            raise ValueError("calibration-fit EM cells are not exactly balanced")
        cell_maps[cell] = _fit_one_map(marginal[selected], joint[selected])
    return {
        "status": "fitted_split_conformal_region_level_maps",
        "method": "split_conformal_raw_posterior_region_level_map",
        "calibration_case_count": expected_total,
        "calibration_cases_per_em_cell": expected_per_cell,
        "nominal_levels": list(LEVELS),
        "global": _fit_one_map(marginal, joint),
        "em_cells": cell_maps,
        "primary_case_level_map": "matching_em_cell",
        "posterior_samples_or_density_changed": False,
        "tail_specific_map_fitted": False,
    }


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> Tuple[float, float]:
    """Two-sided Wilson score interval with the frozen 95% Gaussian quantile."""

    if total <= 0 or successes < 0 or successes > total or z <= 0:
        raise ValueError("Wilson interval counts or quantile are invalid")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def _coverage_summary(
    marginal_covered: np.ndarray, joint_covered: np.ndarray
) -> Mapping[str, Any]:
    count = len(joint_covered)
    marginal_counts = [
        int(np.count_nonzero(marginal_covered[:, 0])),
        int(np.count_nonzero(marginal_covered[:, 1])),
    ]
    joint_count = int(np.count_nonzero(joint_covered))
    return {
        "case_count": count,
        "marginal_covered_count": marginal_counts,
        "marginal_coverage": [value / count for value in marginal_counts],
        "marginal_wilson_95": [
            list(wilson_interval(value, count)) for value in marginal_counts
        ],
        "joint_covered_count": joint_count,
        "joint_coverage": joint_count / count,
        "joint_wilson_95": list(wilson_interval(joint_count, count)),
    }


def calibrated_region_coverage(
    calibration_map: Mapping[str, Any],
    marginal_scores: np.ndarray,
    joint_scores: np.ndarray,
    em_cells: Sequence[str],
) -> Mapping[str, Any]:
    """Apply frozen EM-cell maps to an independent set of region scores."""

    marginal = _finite(marginal_scores, name="marginal evaluation scores")
    joint = _finite(joint_scores, name="joint evaluation scores")
    cells = tuple(str(value) for value in em_cells)
    if marginal.ndim != 2 or marginal.shape[1] != 2:
        raise ValueError("marginal score array must have two columns")
    if joint.shape != (len(marginal),) or len(cells) != len(marginal):
        raise ValueError("calibrated coverage inputs have incompatible lengths")
    by_level = {}
    global_by_level = {}
    cell_by_level: dict[str, dict[str, Any]] = {
        cell: {} for cell in sorted(set(cells))
    }
    for level in LEVELS:
        key = f"{level:.2f}"
        marginal_covered = np.empty_like(marginal, dtype=bool)
        joint_covered = np.empty(len(joint), dtype=bool)
        for index, cell in enumerate(cells):
            try:
                cell_map = calibration_map["em_cells"][cell]
            except KeyError as error:
                raise ValueError(f"calibration map has no EM cell: {cell}") from error
            for target_index, target in enumerate(TARGETS):
                threshold = cell_map["marginal"][target][key][
                    "raw_region_mass_threshold"
                ]
                marginal_covered[index, target_index] = (
                    marginal[index, target_index] <= threshold
                )
            joint_covered[index] = joint[index] <= cell_map["joint"][key][
                "raw_region_mass_threshold"
            ]
        by_level[key] = _coverage_summary(marginal_covered, joint_covered)
        global_map = calibration_map["global"]
        global_marginal = np.column_stack(
            [
                marginal[:, index]
                <= global_map["marginal"][target][key][
                    "raw_region_mass_threshold"
                ]
                for index, target in enumerate(TARGETS)
            ]
        )
        global_joint = joint <= global_map["joint"][key][
            "raw_region_mass_threshold"
        ]
        global_by_level[key] = _coverage_summary(global_marginal, global_joint)
        cell_array = np.asarray(cells, dtype=object)
        for cell in cell_by_level:
            selected = cell_array == cell
            cell_by_level[cell][key] = _coverage_summary(
                marginal_covered[selected], joint_covered[selected]
            )
    return {
        "case_count": len(marginal),
        "coverage": by_level,
        "coverage_using_em_cell_maps": by_level,
        "coverage_using_global_map": global_by_level,
        "em_cell_coverage_using_cell_maps": cell_by_level,
        "wilson_interval": "two_sided_95_percent",
    }


def deterministic_sbc_subset(
    physical_system_ids: Sequence[str], *, root_seed: int, count: int = 1024
) -> Tuple[str, ...]:
    identifiers = tuple(str(value) for value in physical_system_ids)
    if len(identifiers) != len(set(identifiers)) or count > len(identifiers):
        raise ValueError("SBC subset requires unique IDs and an attainable count")
    return tuple(
        sorted(
            identifiers,
            key=lambda value: (
                hashlib.sha256(f"{root_seed}\0{value}".encode()).digest(),
                value,
            ),
        )[:count]
    )


def _randomized_rank(
    draws: np.ndarray, truth: float, *, physical_system_id: str, statistic: str
) -> int:
    values = _finite(draws, name=f"{statistic} SBC draws")
    if values.ndim != 1 or not math.isfinite(truth):
        raise ValueError("SBC rank inputs are invalid")
    less = int(np.count_nonzero(values < truth))
    equal = int(np.count_nonzero(values == truth))
    if equal == 0:
        return less
    digest = hashlib.sha256(
        f"sbc_tie_v1\0{physical_system_id}\0{statistic}".encode()
    ).digest()
    return less + int.from_bytes(digest[:8], "big") % (equal + 1)


def sbc_ranks(
    posterior_draws: np.ndarray,
    truth: np.ndarray,
    physical_system_ids: Sequence[str],
    *,
    posterior_draw_log_density: np.ndarray,
    truth_log_density: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Compute five preregistered exchangeability ranks for each SBC replicate."""

    draws = _finite(posterior_draws, name="SBC posterior draws")
    targets = _finite(truth, name="SBC truths")
    draw_density = _finite(posterior_draw_log_density, name="SBC draw log density")
    truth_density = _finite(truth_log_density, name="SBC truth log density")
    identifiers = tuple(str(value) for value in physical_system_ids)
    if (
        draws.ndim != 3
        or draws.shape[2] != 2
        or targets.shape != (len(draws), 2)
        or draw_density.shape != draws.shape[:2]
        or truth_density.shape != (len(draws),)
        or len(identifiers) != len(draws)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("SBC arrays or physical-system IDs have incompatible shapes")
    transformed_draws = {
        "log_abs_mu_primary": draws[:, :, 0],
        "log_abs_mu_secondary": draws[:, :, 1],
        "log_abs_mu_sum": draws[:, :, 0] + draws[:, :, 1],
        "log_abs_mu_difference": draws[:, :, 0] - draws[:, :, 1],
        "joint_log_density_rank": draw_density,
    }
    transformed_truth = {
        "log_abs_mu_primary": targets[:, 0],
        "log_abs_mu_secondary": targets[:, 1],
        "log_abs_mu_sum": targets[:, 0] + targets[:, 1],
        "log_abs_mu_difference": targets[:, 0] - targets[:, 1],
        "joint_log_density_rank": truth_density,
    }
    return {
        statistic: np.asarray(
            [
                _randomized_rank(
                    transformed_draws[statistic][index],
                    float(transformed_truth[statistic][index]),
                    physical_system_id=identifier,
                    statistic=statistic,
                )
                for index, identifier in enumerate(identifiers)
            ],
            dtype=np.int64,
        )
        for statistic in SBC_STATISTICS
    }


def _discrete_uniform_bin_probabilities(
    posterior_draw_count: int, histogram_bins: int
) -> np.ndarray:
    possible = np.arange(posterior_draw_count + 1, dtype=np.int64)
    assignments = np.minimum(
        histogram_bins - 1,
        possible * histogram_bins // (posterior_draw_count + 1),
    )
    counts = np.bincount(assignments, minlength=histogram_bins)
    return counts.astype(np.float64) / (posterior_draw_count + 1)


def holm_step_down(
    p_values: Mapping[str, float], *, familywise_alpha: float
) -> Mapping[str, Mapping[str, float | bool | int]]:
    """Return monotone Holm-adjusted p-values and step-down rejections."""

    if not 0.0 < familywise_alpha < 1.0 or not p_values:
        raise ValueError("Holm correction requires p-values and alpha in (0,1)")
    ordered = sorted((float(value), str(name)) for name, value in p_values.items())
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("Holm correction received an invalid p-value")
    total = len(ordered)
    adjusted_previous = 0.0
    still_rejecting = True
    result = {}
    for index, (p_value, name) in enumerate(ordered):
        multiplier = total - index
        adjusted = min(1.0, max(adjusted_previous, multiplier * p_value))
        adjusted_previous = adjusted
        threshold = familywise_alpha / multiplier
        rejected = still_rejecting and p_value <= threshold
        if not rejected:
            still_rejecting = False
        result[name] = {
            "raw_p_value": p_value,
            "holm_step": index + 1,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": adjusted,
            "rejected": rejected,
        }
    return result


def evaluate_sbc_histograms(
    ranks: Mapping[str, np.ndarray],
    *,
    posterior_draw_count: int = 1024,
    histogram_bins: int = 20,
    familywise_alpha: float = 0.01,
    expected_replicate_count: int = 1024,
) -> Mapping[str, Any]:
    """Apply the frozen discrete-uniform chi-square and Holm familywise test."""

    if expected_replicate_count <= 0:
        raise ValueError("SBC expected replicate count must be positive")
    if set(ranks) != set(SBC_STATISTICS):
        raise ValueError("SBC rank statistics differ from the frozen family")
    probabilities = _discrete_uniform_bin_probabilities(
        posterior_draw_count, histogram_bins
    )
    summaries = {}
    p_values = {}
    replicate_count: int | None = None
    for statistic in SBC_STATISTICS:
        values = np.asarray(ranks[statistic], dtype=np.int64)
        if values.ndim != 1 or np.any((values < 0) | (values > posterior_draw_count)):
            raise ValueError("SBC ranks fall outside the allowed discrete support")
        if replicate_count is None:
            replicate_count = len(values)
        elif len(values) != replicate_count:
            raise ValueError("SBC statistics use different replicate counts")
        if len(values) != expected_replicate_count:
            raise ValueError("SBC statistics do not contain exactly 1,024 replicates")
        assignments = np.minimum(
            histogram_bins - 1,
            values * histogram_bins // (posterior_draw_count + 1),
        )
        observed = np.bincount(assignments, minlength=histogram_bins)
        expected = probabilities * len(values)
        statistic_value = float(np.sum((observed - expected) ** 2 / expected))
        p_value = float(chi2.sf(statistic_value, histogram_bins - 1))
        p_values[statistic] = p_value
        summaries[statistic] = {
            "observed_bin_counts": observed.tolist(),
            "expected_bin_counts": expected.tolist(),
            "chi_square": statistic_value,
            "degrees_of_freedom": histogram_bins - 1,
            "raw_p_value": p_value,
        }
    holm = holm_step_down(p_values, familywise_alpha=familywise_alpha)
    for statistic in SBC_STATISTICS:
        summaries[statistic]["holm"] = holm[statistic]
    return {
        "status": "completed_independent_sbc_rank_tests",
        "replicate_count": replicate_count,
        "posterior_draws_per_replicate": posterior_draw_count,
        "histogram_bins": histogram_bins,
        "familywise_alpha": familywise_alpha,
        "any_holm_rejection": any(bool(item["rejected"]) for item in holm.values()),
        "statistics": summaries,
        "calibration_map_fitted_from_sbc": False,
    }
