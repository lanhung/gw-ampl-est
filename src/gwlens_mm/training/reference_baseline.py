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
from types import MappingProxyType
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

import numpy as np

from ..config import load_yaml
from ..provenance import configuration_hash
from .features import InputStandardizer, PreparedExample
from .metrics import empirical_crps

REFERENCE_CONFIG = "configs/statistics/reference_baseline_preregistration.yaml"
REFERENCE_AUTHORIZATION = "configs/execution/phase7_reference_baseline_stack_authorization.yaml"
REFERENCE_CONFIG_HASH = "1df98c89fc418eddfd9ec766cb04311e0f3d9f40836a0d9ba1dd691d6bc1724e"
PARENT_RC6_HASH = "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
NEIGHBOR_COUNT = 256
POSTERIOR_DRAW_COUNT = 4096
VARIANCE_FLOOR = 1.0e-6
SEED_DOMAIN = "non_neural_reference_posterior_sampling_v1"
REFERENCE_LEVELS = (0.50, 0.80, 0.90, 0.95)


class ReferenceBaselineGateError(ValueError):
    """Raised when the reference baseline would violate the frozen contract."""


class MetadataOnlyDataset(Protocol):
    """Reader surface required by the reference without any strain access."""

    def __len__(self) -> int: ...

    def metadata_example(self, index: int) -> PreparedExample: ...


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


@dataclass(frozen=True)
class ReferenceCaseScore:
    """Small per-case score product; no posterior draws are persisted."""

    physical_system_id: str
    lens_family: str
    em_cell: str
    target_log_abs_mu: Tuple[float, float]
    log_probability: float
    crps: Tuple[float, float]
    marginal_coverage: Mapping[str, Tuple[bool, bool]]
    joint_central_coverage: Mapping[str, bool]
    interval_width: Mapping[str, Tuple[float, float]]
    effective_neighbor_count: float
    neighbor_identity_sha256: str

    def as_mapping(self) -> Mapping[str, Any]:
        return {
            "physical_system_id": self.physical_system_id,
            "lens_family": self.lens_family,
            "em_cell": self.em_cell,
            "target_log_abs_mu": list(self.target_log_abs_mu),
            "log_probability": self.log_probability,
            "negative_log_probability_per_target_dimension": (
                -self.log_probability / 2.0
            ),
            "crps": list(self.crps),
            "marginal_coverage": {
                level: list(value) for level, value in self.marginal_coverage.items()
            },
            "joint_central_coverage": dict(self.joint_central_coverage),
            "interval_width": {
                level: list(value) for level, value in self.interval_width.items()
            },
            "effective_neighbor_count": self.effective_neighbor_count,
            "neighbor_identity_sha256": self.neighbor_identity_sha256,
            "posterior_draws_persisted": False,
            "reference_is_exact_likelihood_or_gold": False,
        }


@dataclass(frozen=True)
class _ReferenceGroup:
    physical_system_ids: np.ndarray
    features: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        identifiers = np.asarray(self.physical_system_ids)
        features = np.asarray(self.features, dtype=np.float64)
        targets = np.asarray(self.targets, dtype=np.float64)
        if (
            identifiers.ndim != 1
            or features.ndim != 2
            or targets.shape != (len(identifiers), 2)
            or len(identifiers) != len(features)
            or len(set(str(value) for value in identifiers)) != len(identifiers)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(targets))
        ):
            raise ReferenceBaselineGateError("reference-bank group is invalid")
        for array in (identifiers, features, targets):
            array.setflags(write=False)


@dataclass(frozen=True)
class ReferenceBankIndex:
    """Immutable vectorized index for the frozen family/cell reference bank."""

    groups: Mapping[Tuple[str, str], _ReferenceGroup]
    physical_system_ids: frozenset[str]
    feature_dimension: int
    identity_sha256: str

    @classmethod
    def build(cls, examples: Sequence[PreparedExample]) -> "ReferenceBankIndex":
        """Build a deterministic index from standardized training metadata only."""

        ordered = sorted(examples, key=lambda item: item.physical_system_id)
        if not ordered:
            raise ReferenceBaselineGateError("reference bank is empty")
        identifiers = [item.physical_system_id for item in ordered]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ReferenceBaselineGateError("reference-bank IDs are empty or duplicated")
        grouped: Dict[Tuple[str, str], list[Tuple[str, np.ndarray, np.ndarray]]] = {}
        dimension = -1
        digest = hashlib.sha256()
        digest.update(b"selected_prior_em_timing_knn_kde_v1\0")
        for example in ordered:
            if not example.lens_family or not example.em_cell:
                raise ReferenceBaselineGateError(
                    "reference bank requires exact lens-family and EM-cell labels"
                )
            feature = reference_feature_vector(example)
            target = np.asarray(example.target, dtype=np.float64)
            if target.shape != (2,) or not np.all(np.isfinite(target)):
                raise ReferenceBaselineGateError("reference target is invalid")
            if dimension < 0:
                dimension = len(feature)
            elif len(feature) != dimension:
                raise ReferenceBaselineGateError(
                    "reference-bank feature dimensions are inconsistent"
                )
            key = (example.lens_family, example.em_cell)
            grouped.setdefault(key, []).append(
                (example.physical_system_id, feature, target)
            )
            for value in (example.physical_system_id, *key):
                digest.update(value.encode("utf-8"))
                digest.update(b"\0")
            digest.update(np.asarray(feature, dtype="<f8").tobytes())
            digest.update(np.asarray(target, dtype="<f8").tobytes())
        groups = {
            key: _ReferenceGroup(
                physical_system_ids=np.asarray(
                    [value[0] for value in values], dtype=np.str_
                ),
                features=np.stack([value[1] for value in values]),
                targets=np.stack([value[2] for value in values]),
            )
            for key, values in grouped.items()
        }
        return cls(
            groups=MappingProxyType(groups),
            physical_system_ids=frozenset(identifiers),
            feature_dimension=dimension,
            identity_sha256=digest.hexdigest(),
        )

    def posterior(
        self,
        query: PreparedExample,
        *,
        require_group_disjoint: bool = True,
    ) -> ReferencePosterior:
        """Resolve one exact-family/cell posterior without scanning other strata."""

        if require_group_disjoint and query.physical_system_id in self.physical_system_ids:
            raise ReferenceBaselineGateError("reference bank overlaps the query split")
        if not query.em_cell:
            raise ReferenceBaselineGateError("reference query has no exact EM-cell label")
        group = self.groups.get((query.lens_family, query.em_cell))
        if group is None:
            raise ReferenceBaselineGateError("reference query stratum is absent")
        feature = reference_feature_vector(query)
        if len(feature) != self.feature_dimension:
            raise ReferenceBaselineGateError("reference and query feature dimensions differ")
        keep = group.physical_system_ids != query.physical_system_id
        identifiers = group.physical_system_ids[keep]
        features = group.features[keep]
        targets = group.targets[keep]
        if len(identifiers) < NEIGHBOR_COUNT:
            raise ReferenceBaselineGateError("fewer than 256 exact-family/cell references")
        distances = np.einsum(
            "ni,ni->n",
            features - feature[None, :],
            features - feature[None, :],
        )
        order = np.lexsort((identifiers, distances))[:NEIGHBOR_COUNT]
        return _reference_posterior_from_neighbors(
            query.physical_system_id,
            identifiers[order],
            targets[order],
            distances[order],
        )

    def manifest(self) -> Mapping[str, Any]:
        """Return a small deterministic identity without exposing bank targets."""

        strata = {
            f"{family}::{cell}": len(group.physical_system_ids)
            for (family, cell), group in sorted(self.groups.items())
        }
        return {
            "reference_id": "selected_prior_em_timing_knn_kde_v1",
            "identity_sha256": self.identity_sha256,
            "physical_system_count": len(self.physical_system_ids),
            "feature_dimension": self.feature_dimension,
            "stratum_counts": strata,
            "neighbor_count": NEIGHBOR_COUNT,
            "posterior_draws_per_case": POSTERIOR_DRAW_COUNT,
            "gw_strain_opened": False,
            "targets_exposed_as_deployable_inputs": False,
        }

    def score(self, query: PreparedExample) -> ReferenceCaseScore:
        """Score one disjoint query using the frozen deterministic draw contract."""

        posterior = self.posterior(query, require_group_disjoint=True)
        target = np.asarray(query.target, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ReferenceBaselineGateError("reference query target is invalid")
        draws = posterior.sample()
        crps = empirical_crps(draws[None, :, :], target[None, :])[0]
        marginal: Dict[str, Tuple[bool, bool]] = {}
        joint: Dict[str, bool] = {}
        width: Dict[str, Tuple[float, float]] = {}
        for level in REFERENCE_LEVELS:
            label = f"{level:.2f}"
            alpha = (1.0 - level) / 2.0
            lower = np.quantile(draws, alpha, axis=0)
            upper = np.quantile(draws, 1.0 - alpha, axis=0)
            contained = (target >= lower) & (target <= upper)
            marginal[label] = (bool(contained[0]), bool(contained[1]))
            joint[label] = bool(np.all(contained))
            widths = upper - lower
            width[label] = (float(widths[0]), float(widths[1]))
        neighbor_digest = hashlib.sha256()
        for identifier in posterior.neighbor_physical_system_ids:
            neighbor_digest.update(identifier.encode("utf-8"))
            neighbor_digest.update(b"\0")
        return ReferenceCaseScore(
            physical_system_id=query.physical_system_id,
            lens_family=query.lens_family,
            em_cell=str(query.em_cell),
            target_log_abs_mu=(float(target[0]), float(target[1])),
            log_probability=float(posterior.log_probability(target)),
            crps=(float(crps[0]), float(crps[1])),
            marginal_coverage=marginal,
            joint_central_coverage=joint,
            interval_width=width,
            effective_neighbor_count=posterior.effective_neighbor_count,
            neighbor_identity_sha256=neighbor_digest.hexdigest(),
        )

    def score_queries(
        self, queries: Sequence[PreparedExample]
    ) -> Tuple[ReferenceCaseScore, ...]:
        """Score a bounded query sequence and reject duplicate query identities."""

        identifiers = tuple(query.physical_system_id for query in queries)
        if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
            raise ReferenceBaselineGateError("reference query IDs are empty or duplicated")
        return tuple(self.score(query) for query in queries)


def build_standardized_reference_bank(
    dataset: MetadataOnlyDataset,
    standardizer: InputStandardizer,
) -> ReferenceBankIndex:
    """Build the locked-rung bank through the metadata-only reader surface."""

    if len(dataset) < NEIGHBOR_COUNT:
        raise ReferenceBaselineGateError("reference dataset has fewer than 256 systems")
    examples = tuple(
        standardizer.transform(dataset.metadata_example(index))
        for index in range(len(dataset))
    )
    if any(example.gw_strain.size for example in examples):
        raise ReferenceBaselineGateError("reference-bank construction opened GW strain")
    return ReferenceBankIndex.build(examples)


def score_standardized_reference_queries(
    index: ReferenceBankIndex,
    dataset: MetadataOnlyDataset,
    standardizer: InputStandardizer,
) -> Tuple[ReferenceCaseScore, ...]:
    """Score a query publication through metadata only and training-rung scales."""

    queries = tuple(
        standardizer.transform(dataset.metadata_example(position))
        for position in range(len(dataset))
    )
    if any(query.gw_strain.size for query in queries):
        raise ReferenceBaselineGateError("reference query scoring opened GW strain")
    return index.score_queries(queries)


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


def _reference_posterior_from_neighbors(
    query_physical_system_id: str,
    identifiers: Sequence[str],
    targets: np.ndarray,
    distances: np.ndarray,
) -> ReferencePosterior:
    """Apply the frozen weighting and KDE rule to exactly 256 ordered neighbors."""

    neighbor_ids = tuple(str(value) for value in identifiers)
    target_values = np.asarray(targets, dtype=np.float64)
    distance_values = np.asarray(distances, dtype=np.float64)
    if (
        len(neighbor_ids) != NEIGHBOR_COUNT
        or target_values.shape != (NEIGHBOR_COUNT, 2)
        or distance_values.shape != (NEIGHBOR_COUNT,)
        or not np.all(np.isfinite(target_values))
        or not np.all(np.isfinite(distance_values))
        or np.any(distance_values < 0.0)
    ):
        raise ReferenceBaselineGateError("selected reference neighbors are invalid")
    bandwidth_squared = max(
        float(distance_values[-1]), float(np.finfo(np.float64).eps)
    )
    raw_weights = np.exp(-0.5 * distance_values / bandwidth_squared)
    weights = raw_weights / np.sum(raw_weights)
    mean = np.sum(weights[:, None] * target_values, axis=0)
    centered = target_values - mean
    effective_count = float(1.0 / np.sum(weights**2))
    denominator = 1.0 - float(np.sum(weights**2))
    if denominator <= 0.0:
        raise ReferenceBaselineGateError("weighted covariance denominator is invalid")
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered) / denominator
    scott = effective_count ** (-1.0 / 6.0)
    kernel_covariance = scott**2 * covariance + VARIANCE_FLOOR * np.eye(2)
    return ReferencePosterior(
        query_physical_system_id,
        neighbor_ids,
        target_values,
        weights,
        kernel_covariance,
        effective_count,
        bandwidth_squared,
    )


def build_reference_posterior(
    query: PreparedExample,
    reference_bank: Sequence[PreparedExample],
) -> ReferencePosterior:
    """Construct the frozen 256-neighbor selected-prior simulation reference."""

    return ReferenceBankIndex.build(reference_bank).posterior(
        query,
        require_group_disjoint=False,
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
