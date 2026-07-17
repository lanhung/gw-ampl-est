"""Preregistered paired 16k-to-32k learning-curve decision."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import TrainingGateError

LEVELS = (0.50, 0.80, 0.90, 0.95)
SEEDS = (0, 1, 2)


def _load_cases(path: Path) -> Tuple[Mapping[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    if not rows:
        raise TrainingGateError(f"learning-curve case table is empty: {path}")
    identifiers = [row["physical_system_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise TrainingGateError("learning-curve table has duplicate physical-system IDs")
    return rows


def _aligned(
    smaller: Sequence[Mapping[str, str]], larger: Sequence[Mapping[str, str]]
) -> Tuple[Tuple[Mapping[str, str], Mapping[str, str]], ...]:
    smaller_by_id = {row["physical_system_id"]: row for row in smaller}
    larger_by_id = {row["physical_system_id"]: row for row in larger}
    if set(smaller_by_id) != set(larger_by_id):
        raise TrainingGateError("learning-curve rungs did not use identical validation cases")
    return tuple((smaller_by_id[key], larger_by_id[key]) for key in sorted(smaller_by_id))


def _coverage_errors(rows: Sequence[Mapping[str, str]]) -> Mapping[str, float]:
    result: Dict[str, float] = {}
    for level in LEVELS:
        key = f"{level:.2f}"
        for target in ("primary", "secondary"):
            coverage = np.mean(
                [row[f"covered_{target}_{key}"].lower() == "true" for row in rows]
            )
            result[f"marginal_{target}_{key}"] = float(abs(coverage - level))
        joint = np.mean([row[f"covered_joint_{key}"].lower() == "true" for row in rows])
        result[f"joint_{key}"] = float(abs(joint - level))
    return result


def _em_cell_errors(rows: Sequence[Mapping[str, str]]) -> Mapping[str, Mapping[str, float]]:
    result: Dict[str, Mapping[str, float]] = {}
    for cell in sorted({row["em_cell_signature"] for row in rows}):
        selected = [row for row in rows if row["em_cell_signature"] == cell]
        result[cell] = {
            "case_count": float(len(selected)),
            "primary_error_0.90": float(
                abs(
                    np.mean(
                        [row["covered_primary_0.90"].lower() == "true" for row in selected]
                    )
                    - 0.90
                )
            ),
            "secondary_error_0.90": float(
                abs(
                    np.mean(
                        [
                            row["covered_secondary_0.90"].lower() == "true"
                            for row in selected
                        ]
                    )
                    - 0.90
                )
            ),
            "joint_error_0.90": float(
                abs(
                    np.mean(
                        [row["covered_joint_0.90"].lower() == "true" for row in selected]
                    )
                    - 0.90
                )
            ),
        }
    return result


def _paired_bootstrap(
    improvements: np.ndarray, *, replicates: int, seed: int
) -> Mapping[str, float]:
    if improvements.ndim != 2 or improvements.shape[1] != 3:
        raise ValueError("paired bootstrap requires cases by three seeds")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    case_count = improvements.shape[0]
    chunk = 250
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, case_count, size=(stop - start, case_count))
        values[start:stop] = np.mean(improvements[indices], axis=(1, 2))
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {
        "replicates": replicates,
        "point_estimate": float(np.mean(improvements)),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _compare_rungs(
    smaller_root: Path,
    larger_root: Path,
    *,
    smaller_rung: int,
    larger_rung: int,
    comparison_label: str,
    saturation_decision: str,
    nonsaturation_decision: str,
    bootstrap_replicates: int = 10000,
    bootstrap_seed_domain: str = "adaptive_learning_curve_bootstrap_v1",
) -> Mapping[str, Any]:
    """Apply the frozen all-conditions comparison to two three-seed rungs."""

    aligned_by_seed = {}
    nlp_improvements = []
    seed_summaries: Dict[str, Any] = {}
    all_identifiers: Optional[Tuple[str, ...]] = None
    tail_counts: Optional[Mapping[str, int]] = None
    conditions: list[bool] = []
    for seed in SEEDS:
        smaller = _load_cases(
            smaller_root
            / f"rung-{smaller_rung}"
            / f"seed-{seed}"
            / "development_cases.csv"
        )
        larger = _load_cases(
            larger_root
            / f"rung-{larger_rung}"
            / f"seed-{seed}"
            / "development_cases.csv"
        )
        aligned = _aligned(smaller, larger)
        identifiers = tuple(pair[0]["physical_system_id"] for pair in aligned)
        if all_identifiers is None:
            all_identifiers = identifiers
        elif identifiers != all_identifiers:
            raise TrainingGateError("different seeds used different validation ordering")
        aligned_by_seed[seed] = aligned
        nlp_delta = np.asarray(
            [
                float(pair[0]["nlp_nat_per_target_dimension"])
                - float(pair[1]["nlp_nat_per_target_dimension"])
                for pair in aligned
            ]
        )
        nlp_improvements.append(nlp_delta)
        median_crps_smaller = float(
            np.median([float(pair[0]["crps_mean"]) for pair in aligned])
        )
        median_crps_larger = float(
            np.median([float(pair[1]["crps_mean"]) for pair in aligned])
        )
        crps_relative = (median_crps_smaller - median_crps_larger) / median_crps_smaller
        smaller_errors = _coverage_errors(smaller)
        larger_errors = _coverage_errors(larger)
        maximum_marginal_improvement = max(
            smaller_errors[key] - larger_errors[key]
            for key in smaller_errors
            if key.startswith("marginal_")
        )
        smaller_cells = _em_cell_errors(smaller)
        larger_cells = _em_cell_errors(larger)
        if set(smaller_cells) != set(larger_cells):
            raise TrainingGateError("EM-cell membership changed between learning-curve rungs")
        cell_tolerance_passed = all(
            max(values["primary_error_0.90"], values["secondary_error_0.90"]) <= 0.04
            and values["joint_error_0.90"] <= 0.05
            for values in larger_cells.values()
        )
        maximum_cell_degradation = max(
            larger_cells[cell][metric] - smaller_cells[cell][metric]
            for cell in larger_cells
            for metric in (
                "primary_error_0.90",
                "secondary_error_0.90",
                "joint_error_0.90",
            )
        )
        observed_tail_counts = {
            group: sum(row["tail_view"] == group for row in larger)
            for group in sorted({row["tail_view"] for row in larger} - {"none"})
        }
        if tail_counts is None:
            tail_counts = observed_tail_counts
        elif observed_tail_counts != tail_counts:
            raise TrainingGateError("validation tail membership changed across seeds")
        seed_summary = {
            "mean_nlp_improvement": float(np.mean(nlp_delta)),
            "median_crps_relative_improvement": float(crps_relative),
            "maximum_marginal_coverage_error_improvement": float(
                maximum_marginal_improvement
            ),
            "em_cell_tolerances_passed": cell_tolerance_passed,
            "maximum_em_cell_coverage_error_degradation": float(maximum_cell_degradation),
        }
        seed_summaries[str(seed)] = seed_summary
        conditions.extend(
            (
                seed_summary["mean_nlp_improvement"] < 0.01,
                seed_summary["median_crps_relative_improvement"] < 0.01,
                seed_summary["maximum_marginal_coverage_error_improvement"] < 0.005,
                cell_tolerance_passed,
                maximum_cell_degradation <= 0.02,
            )
        )
    improvements = np.stack(nlp_improvements, axis=1)
    assert all_identifiers is not None
    identifier_hash = hashlib.sha256("\n".join(all_identifiers).encode()).hexdigest()
    seed_payload = (bootstrap_seed_domain + "\0" + identifier_hash).encode()
    bootstrap_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    bootstrap = _paired_bootstrap(
        improvements, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    assert tail_counts is not None
    tail_requirement_passed = bool(tail_counts) and all(
        count >= 128 for count in tail_counts.values()
    )
    conditions.extend((bootstrap["upper_95"] < 0.01, tail_requirement_passed))
    passed = all(bool(value) for value in conditions)
    return {
        "status": "learning_curve_decision_complete",
        "comparison": comparison_label,
        "validation_case_count": len(all_identifiers),
        "seed_summaries": seed_summaries,
        "paired_nlp_bootstrap": bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "tail_view_case_counts": tail_counts or {},
        "tail_minimum_case_requirement_passed": tail_requirement_passed,
        "all_saturation_conditions_passed": passed,
        "decision": saturation_decision if passed else nonsaturation_decision,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }


def compare_16k_to_32k(
    output_root: Path,
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_seed_domain: str = "adaptive_learning_curve_bootstrap_v1",
) -> Mapping[str, Any]:
    """Apply the frozen 16k-to-32k decision."""

    return _compare_rungs(
        output_root,
        output_root,
        smaller_rung=16384,
        larger_rung=32768,
        comparison_label="train_16k_probe_subset_to_train_32k",
        saturation_decision="lock_train_32k",
        nonsaturation_decision="continue_to_train_65k",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed_domain=bootstrap_seed_domain,
    )


def compare_32k_to_65k(
    smaller_output_root: Path,
    larger_output_root: Path,
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_seed_domain: str = "adaptive_learning_curve_bootstrap_v1",
) -> Mapping[str, Any]:
    """Apply the terminal 32k-to-65k rule without authorizing a larger rung."""

    result = dict(
        _compare_rungs(
            smaller_output_root,
            larger_output_root,
            smaller_rung=32768,
            larger_rung=65536,
            comparison_label="train_32k_to_train_65k",
            saturation_decision="lock_train_65k",
            nonsaturation_decision="stop_inconclusive_and_new_preregistration",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed_domain=bootstrap_seed_domain,
        )
    )
    if not result["all_saturation_conditions_passed"]:
        bootstrap = result["paired_nlp_bootstrap"]
        if float(bootstrap["lower_95"]) >= 0.01:
            result["decision"] = "stop_data_limited_and_new_preregistration"
    result["extension_above_65536_authorized"] = False
    return result


def write_learning_curve_decision(output_root: Path, path: Path) -> Mapping[str, Any]:
    result = compare_16k_to_32k(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(path)
    return result
