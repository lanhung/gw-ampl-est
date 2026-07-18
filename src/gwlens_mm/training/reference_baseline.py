"""Pure selected-prior non-neural reference baseline.

The reference intentionally uses only deployable EM/timing observations for
neighbor selection.  Training-rung targets supply the empirical posterior
support, as in any supervised baseline, but query truth is never inspected.
It is a simulation reference, not an exact likelihood or a gold posterior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from .features import PreparedExample

REFERENCE_CONFIG = "configs/statistics/reference_baseline_preregistration.yaml"
REFERENCE_AUTHORIZATION = "configs/execution/phase7_reference_baseline_stack_authorization.yaml"
REFERENCE_CONFIG_HASH = "1df98c89fc418eddfd9ec766cb04311e0f3d9f40836a0d9ba1dd691d6bc1724e"
PARENT_RC6_HASH = "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
NEIGHBOR_COUNT = 256
POSTERIOR_DRAW_COUNT = 4096
VARIANCE_FLOOR = 1.0e-6
SEED_DOMAIN = "non_neural_reference_posterior_sampling_v1"


class ReferenceBaselineGateError(ValueError):
    """Raised when the reference baseline would violate the frozen contract."""


@dataclass(frozen=True)
class ReferencePosterior:
    """Weighted target KDE from one exact-family, exact-EM-cell neighbor set."""

    query_physical_system_id: str
    neighbor_physical_system_ids: Tuple[str, ...]
    neighbor_targets: np.ndarray
    normalized_weights: np.ndarray
    kernel_covariance: np.ndarray
    effective_neighbor_count: float
    bandwidth_squared: float

    def __post_init__(self) -> None:
        if (
            not self.query_physical_system_id
            or len(self.neighbor_physical_system_ids) != NEIGHBOR_COUNT
            or len(set(self.neighbor_physical_system_ids)) != NEIGHBOR_COUNT
            or self.query_physical_system_id in self.neighbor_physical_system_ids
            or np.asarray(self.neighbor_targets).shape != (NEIGHBOR_COUNT, 2)
            or np.asarray(self.normalized_weights).shape != (NEIGHBOR_COUNT,)
            or np.asarray(self.kernel_covariance).shape != (2, 2)
        ):
            raise ReferenceBaselineGateError("reference posterior identity or shape is invalid")
        arrays = (
            np.asarray(self.neighbor_targets),
            np.asarray(self.normalized_weights),
            np.asarray(self.kernel_covariance),
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ReferenceBaselineGateError("reference posterior contains NaN or Inf")
        if (
            np.any(self.normalized_weights <= 0.0)
            or not math.isclose(float(np.sum(self.normalized_weights)), 1.0, abs_tol=1e-12)
            or self.effective_neighbor_count <= 1.0
            or self.bandwidth_squared <= 0.0
        ):
            raise ReferenceBaselineGateError("reference weights or bandwidth are invalid")
        sign, _ = np.linalg.slogdet(self.kernel_covariance)
        if sign <= 0 or not np.allclose(
            self.kernel_covariance, self.kernel_covariance.T, atol=1e-12
        ):
            raise ReferenceBaselineGateError("reference KDE covariance is not positive definite")

    def log_probability(self, targets: np.ndarray) -> np.ndarray:
        """Evaluate the normalized two-dimensional Gaussian mixture density."""

        values = np.asarray(targets, dtype=np.float64)
        scalar = values.ndim == 1
        if scalar:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
            raise ReferenceBaselineGateError("reference density targets must be finite Nx2")
        covariance = np.asarray(self.kernel_covariance, dtype=np.float64)
        inverse = np.linalg.inv(covariance)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ReferenceBaselineGateError("reference covariance is singular")
        delta = values[:, None, :] - np.asarray(self.neighbor_targets)[None, :, :]
        quadratic = np.einsum("nki,ij,nkj->nk", delta, inverse, delta)
        component = (
            np.log(np.asarray(self.normalized_weights))[None, :]
            - 0.5 * (2.0 * math.log(2.0 * math.pi) + log_determinant + quadratic)
        )
        maximum = np.max(component, axis=1)
        result = maximum + np.log(np.sum(np.exp(component - maximum[:, None]), axis=1))
        return result[0] if scalar else result

    def sample(self, count: int = POSTERIOR_DRAW_COUNT) -> np.ndarray:
        """Draw deterministically for this query from the frozen KDE."""

        if count != POSTERIOR_DRAW_COUNT:
            raise ReferenceBaselineGateError("reference posterior draw count must remain 4096")
        digest = hashlib.sha256(
            f"{SEED_DOMAIN}\0{self.query_physical_system_id}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
        selected = rng.choice(
            NEIGHBOR_COUNT,
            size=count,
            replace=True,
            p=np.asarray(self.normalized_weights),
        )
        jitter = rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            np.asarray(self.kernel_covariance, dtype=np.float64),
            size=count,
        )
        result = np.asarray(self.neighbor_targets)[selected] + jitter
        if result.shape != (count, 2) or not np.all(np.isfinite(result)):
            raise ReferenceBaselineGateError("reference posterior sampling failed")
        return result


def load_reference_baseline_contract(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate the prospective RC.7 contract while keeping execution closed."""

    config = load_yaml(root / REFERENCE_CONFIG)
    authorization = load_yaml(root / REFERENCE_AUTHORIZATION)
    if configuration_hash(config) != REFERENCE_CONFIG_HASH:
        raise ReferenceBaselineGateError("reference preregistration hash mismatch")
    if config.get("preregistration_version") != "1.1.0-rc.7":
        raise ReferenceBaselineGateError("reference preregistration version mismatch")
    parent = load_yaml(root / "configs/statistics/final_evaluation_analysis_preregistration.yaml")
    if configuration_hash(parent) != PARENT_RC6_HASH:
        raise ReferenceBaselineGateError("parent RC.6 hash mismatch")
    superseded = config.get("superseded_non_executable_obligation", {})
    if (
        superseded.get("full_latent_proposal_implemented") is not False
        or superseded.get("importance_sampling_efficiency_claim_authorized") is not False
        or superseded.get("exact_likelihood_correction_claim_authorized") is not False
    ):
        raise ReferenceBaselineGateError("RC.7 must not claim unavailable likelihood correction")
    reference = config.get("non_neural_reference", {})
    if (
        reference.get("id") != "selected_prior_em_timing_knn_kde_v1"
        or reference.get("neighbors", {}).get("exact_count") != NEIGHBOR_COUNT
        or reference.get("target_kde", {}).get("posterior_draws_per_case")
        != POSTERIOR_DRAW_COUNT
        or float(reference.get("target_kde", {}).get("diagonal_variance_floor", 0.0))
        != VARIANCE_FLOOR
        or reference.get("hyperparameter_tuning_authorized") is not False
    ):
        raise ReferenceBaselineGateError("non-neural reference contract drifted")
    if any(value is not False for value in config.get("execution", {}).values()):
        raise ReferenceBaselineGateError("RC.7 execution must remain disabled")
    if authorization.get("authorization_status") != (
        "authorized_preregistration_and_pure_implementation_only"
    ):
        raise ReferenceBaselineGateError("reference implementation authorization is absent")
    if authorization.get("frozen_addendum", {}).get("canonical_hash") != REFERENCE_CONFIG_HASH:
        raise ReferenceBaselineGateError("authorization references the wrong RC.7 hash")
    flags = authorization.get("authorization", {})
    allowed_true = {
        "downstream_preregistration_authorized",
        "pure_reference_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    if any(flags.get(name) is not True for name in allowed_true):
        raise ReferenceBaselineGateError(
            "pure reference implementation authorization is incomplete"
        )
    if any(value is not False for name, value in flags.items() if name not in allowed_true):
        raise ReferenceBaselineGateError("reference implementation opened an execution gate")
    return config, authorization


def reference_feature_vector(example: PreparedExample) -> np.ndarray:
    """Flatten deployable EM/timing values and masks, never GW or target truth."""

    scalar_mask = np.asarray(example.scalar_mask, dtype=np.float64)
    scalar = np.asarray(example.scalar_features, dtype=np.float64) * scalar_mask
    astrometry_mask = np.asarray(example.astrometry_mask, dtype=np.float64)
    astrometry = np.asarray(example.astrometry_items, dtype=np.float64) * astrometry_mask[:, None]
    modality = np.asarray(example.modality_mask, dtype=np.float64)
    family = np.asarray(example.lens_family_condition, dtype=np.float64)
    if (
        scalar.shape != (22,)
        or scalar_mask.shape != (22,)
        or astrometry.ndim != 2
        or astrometry.shape[1] != 9
        or astrometry_mask.shape != (len(astrometry),)
        or family.shape != (2,)
    ):
        raise ReferenceBaselineGateError("prepared reference feature shapes are invalid")
    result = np.concatenate(
        (
            scalar,
            scalar_mask,
            astrometry.reshape(-1),
            astrometry_mask,
            modality,
            family,
        )
    )
    if not np.all(np.isfinite(result)):
        raise ReferenceBaselineGateError("reference feature vector is nonfinite")
    return result


def build_reference_posterior(
    query: PreparedExample,
    reference_bank: Sequence[PreparedExample],
) -> ReferencePosterior:
    """Construct the frozen 256-neighbor selected-prior simulation reference."""

    query_feature = reference_feature_vector(query)
    candidates = []
    for reference in reference_bank:
        if reference.physical_system_id == query.physical_system_id:
            continue
        if reference.lens_family != query.lens_family:
            continue
        if not query.em_cell or reference.em_cell != query.em_cell:
            continue
        feature = reference_feature_vector(reference)
        if feature.shape != query_feature.shape:
            raise ReferenceBaselineGateError("reference and query feature dimensions differ")
        distance_squared = float(np.sum((feature - query_feature) ** 2))
        target = np.asarray(reference.target, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ReferenceBaselineGateError("reference target is invalid")
        candidates.append((distance_squared, reference.physical_system_id, target))
    if len(candidates) < NEIGHBOR_COUNT:
        raise ReferenceBaselineGateError("fewer than 256 exact-family/cell references")
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[:NEIGHBOR_COUNT]
    distances = np.asarray([item[0] for item in selected], dtype=np.float64)
    bandwidth_squared = max(float(distances[-1]), float(np.finfo(np.float64).eps))
    raw_weights = np.exp(-0.5 * distances / bandwidth_squared)
    weights = raw_weights / np.sum(raw_weights)
    targets = np.stack([item[2] for item in selected])
    mean = np.sum(weights[:, None] * targets, axis=0)
    centered = targets - mean
    effective_count = float(1.0 / np.sum(weights**2))
    denominator = 1.0 - float(np.sum(weights**2))
    if denominator <= 0.0:
        raise ReferenceBaselineGateError("weighted covariance denominator is invalid")
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered) / denominator
    scott = effective_count ** (-1.0 / 6.0)
    kernel_covariance = scott**2 * covariance + VARIANCE_FLOOR * np.eye(2)
    return ReferencePosterior(
        query.physical_system_id,
        tuple(item[1] for item in selected),
        targets,
        weights,
        kernel_covariance,
        effective_count,
        bandwidth_squared,
    )


def dry_run_plan(root: Path) -> Mapping[str, Any]:
    """Return a plan with no scientific bank, query, or final identity resolved."""

    config, _ = load_reference_baseline_contract(root)
    return {
        "status": "implementation_ready_execution_closed",
        "preregistration_version": config["preregistration_version"],
        "preregistration_hash": REFERENCE_CONFIG_HASH,
        "reference_id": config["non_neural_reference"]["id"],
        "neighbor_count": NEIGHBOR_COUNT,
        "posterior_draw_count": POSTERIOR_DRAW_COUNT,
        "scientific_reference_bank_identity": None,
        "query_publication_identity": None,
        "reference_bank_accessed": False,
        "validation_baseline_executed": False,
        "final_evaluation_accessed": False,
        "likelihood_gold_claimed": False,
        "importance_sampling_efficiency_computed": False,
        "gwosc_gwtc_accessed": False,
    }


def write_dry_run_plan(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dry_run_plan(root), indent=2, sort_keys=True) + "\n")
