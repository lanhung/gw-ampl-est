"""Exact proposal-v2 mixture on the frozen RC.5 pre-selection latent space."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..physics.quantities import LensFamily


class Component(str, Enum):
    RC5_BROAD = "rc5_broad"
    WIDE = "wide"
    NARROW = "narrow"
    LOW_Z = "low_z"


COMPONENT_ORDER = (
    Component.RC5_BROAD,
    Component.WIDE,
    Component.NARROW,
    Component.LOW_Z,
)


@dataclass(frozen=True)
class ProposalSpecification:
    component_weights: Tuple[float, float, float, float]
    source_lower: float
    source_upper: float
    wide_sigma: float
    narrow_sigma: float
    z_max: float
    preflight_draw_count: int
    preflight_seed: int
    mean_weight_bounds: Tuple[float, float]
    overall_ess_minimum: float
    family_ess_minimum: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ProposalSpecification":
        mixture = config["mixture"]["components"]
        source = config["source_plane"]
        low_z = config["low_redshift"]
        preflight = config["latent_preflight"]
        spec = cls(
            component_weights=tuple(
                float(mixture[item.value]["weight"]) for item in COMPONENT_ORDER
            ),  # type: ignore[arg-type]
            source_lower=float(source["lower"]),
            source_upper=float(source["upper"]),
            wide_sigma=float(source["wide_sigma"]),
            narrow_sigma=float(source["narrow_sigma"]),
            z_max=float(low_z["z_max"]),
            preflight_draw_count=int(preflight["draw_count"]),
            preflight_seed=int(preflight["seed"]),
            mean_weight_bounds=(
                float(preflight["mean_weight_minimum"]),
                float(preflight["mean_weight_maximum"]),
            ),
            overall_ess_minimum=float(preflight["overall_relative_ess_minimum"]),
            family_ess_minimum=float(
                preflight["per_lens_family_relative_ess_minimum"]
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if len(self.component_weights) != len(COMPONENT_ORDER):
            raise ValueError("proposal-v2 requires exactly four components")
        if any(weight <= 0.0 for weight in self.component_weights):
            raise ValueError("all mixture weights must be positive")
        if not math.isclose(sum(self.component_weights), 1.0, abs_tol=1.0e-15):
            raise ValueError("mixture weights must sum exactly to one")
        if not math.isclose(self.component_weights[0], 0.2, abs_tol=1.0e-15):
            raise ValueError("RC.5 broad-support component weight must be 0.2")
        if not self.source_lower < self.source_upper:
            raise ValueError("source-plane support is invalid")
        if self.wide_sigma <= 0.0 or self.narrow_sigma <= 0.0:
            raise ValueError("source-plane sigmas must be positive")
        if self.z_max <= 1.1:
            raise ValueError("source-redshift upper bound is invalid")
        if self.preflight_draw_count < 200_000 or self.preflight_seed < 0:
            raise ValueError("latent preflight requires at least 200,000 deterministic draws")


@dataclass(frozen=True)
class LatentDraw:
    component: Component
    lens_family: LensFamily
    z_lens: float
    z_source: float
    einstein_radius_arcsec: float
    axis_ratio: float
    position_angle_rad: float
    shear_amplitude: float
    shear_angle_rad: float
    density_slope: float
    external_convergence: float
    source_u_x: float
    source_u_y: float
    mass_1_source: float
    mass_ratio: float
    a_1: float
    a_2: float
    tilt_1: float
    tilt_2: float
    phi_12: float
    phi_jl: float
    theta_jn: float
    phase: float
    ra: float
    dec: float
    psi: float

    @property
    def mass_2_source(self) -> float:
        return self.mass_1_source * self.mass_ratio

    @property
    def source_position_arcsec(self) -> Tuple[float, float]:
        return (
            self.source_u_x * self.einstein_radius_arcsec,
            self.source_u_y * self.einstein_radius_arcsec,
        )

    def replay_bytes(self) -> bytes:
        component_index = COMPONENT_ORDER.index(self.component)
        family_index = 0 if self.lens_family is LensFamily.SIE_EXTERNAL_SHEAR else 1
        values = (
            self.z_lens,
            self.z_source,
            self.einstein_radius_arcsec,
            self.axis_ratio,
            self.position_angle_rad,
            self.shear_amplitude,
            self.shear_angle_rad,
            self.density_slope,
            self.external_convergence,
            self.source_u_x,
            self.source_u_y,
            self.mass_1_source,
            self.mass_ratio,
            self.a_1,
            self.a_2,
            self.tilt_1,
            self.tilt_2,
            self.phi_12,
            self.phi_jl,
            self.theta_jn,
            self.phase,
            self.ra,
            self.dec,
            self.psi,
        )
        return struct.pack(">BB24d", component_index, family_index, *values)


def _inside_half_open(value: float, lower: float, upper: float) -> bool:
    return math.isfinite(value) and lower <= value < upper and lower < upper


def _uniform_log(value: float, lower: float, upper: float) -> float:
    if not _inside_half_open(value, lower, upper):
        return -math.inf
    return -math.log(upper - lower)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def truncated_normal_log_density(
    value: float, mean: float, sigma: float, lower: float, upper: float
) -> float:
    if sigma <= 0.0 or not _inside_half_open(value, lower, upper):
        return -math.inf
    normalization = _normal_cdf((upper - mean) / sigma) - _normal_cdf(
        (lower - mean) / sigma
    )
    if normalization <= 0.0:
        raise ValueError("truncated-normal normalization is invalid")
    standardized = (value - mean) / sigma
    return (
        -0.5 * standardized * standardized
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(sigma)
        - math.log(normalization)
    )


def _sample_truncated_normal(
    rng: np.random.Generator, mean: float, sigma: float, lower: float, upper: float
) -> float:
    if sigma <= 0.0 or lower >= upper:
        raise ValueError("truncated-normal specification is invalid")
    while True:
        value = float(rng.normal(mean, sigma))
        if lower <= value < upper:
            return value


def _log_uniform_log(value: float, lower: float, upper: float) -> float:
    if not _inside_half_open(value, lower, upper) or value <= 0.0:
        return -math.inf
    return -math.log(value) - math.log(math.log(upper / lower))


def _log_uniform_sample(rng: np.random.Generator, lower: float, upper: float) -> float:
    return float(math.exp(rng.uniform(math.log(lower), math.log(upper))))


def _power_law_sample(
    rng: np.random.Generator, exponent: float, lower: float, upper: float
) -> float:
    power = exponent + 1.0
    return float(
        (rng.random() * (upper**power - lower**power) + lower**power)
        ** (1.0 / power)
    )


def _power_law_log(value: float, exponent: float, lower: float, upper: float) -> float:
    if not _inside_half_open(value, lower, upper):
        return -math.inf
    power = exponent + 1.0
    return math.log(power / (upper**power - lower**power)) + exponent * math.log(value)


def _beta_scaled_log(
    value: float, maximum: float, alpha: float = 2.0, beta: float = 5.0
) -> float:
    if not 0.0 < value < maximum:
        return -math.inf
    unit = value / maximum
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        (alpha - 1.0) * math.log(unit)
        + (beta - 1.0) * math.log1p(-unit)
        - log_beta
        - math.log(maximum)
    )


def _source_redshift_lower(z_lens: float, z_max: float) -> float:
    lower = max(0.5, z_lens + 0.1)
    if not math.isfinite(lower) or lower >= z_max:
        raise ValueError("conditional source-redshift interval is invalid")
    return lower


def low_z_log_density(z_source: float, z_lens: float, z_max: float = 3.0) -> float:
    lower = _source_redshift_lower(z_lens, z_max)
    if not _inside_half_open(z_source, lower, z_max):
        return -math.inf
    width = z_max - lower
    unit = (z_source - lower) / width
    density = 2.0 * (1.0 - unit) / width
    return math.log(density) if density > 0.0 else -math.inf


def _sample_component(rng: np.random.Generator, spec: ProposalSpecification) -> Component:
    value = float(rng.random())
    cumulative = 0.0
    for component, weight in zip(COMPONENT_ORDER, spec.component_weights):
        cumulative += weight
        if value < cumulative:
            return component
    return COMPONENT_ORDER[-1]


def _sample_source_coordinate(
    rng: np.random.Generator, component: Component, spec: ProposalSpecification
) -> float:
    if component is Component.RC5_BROAD:
        return float(rng.uniform(spec.source_lower, spec.source_upper))
    sigma = spec.narrow_sigma if component is Component.NARROW else spec.wide_sigma
    return _sample_truncated_normal(
        rng, 0.0, sigma, spec.source_lower, spec.source_upper
    )


def sample_latent(rng: np.random.Generator, spec: ProposalSpecification) -> LatentDraw:
    component = _sample_component(rng, spec)
    family = (
        LensFamily.SIE_EXTERNAL_SHEAR
        if rng.random() < 0.5
        else LensFamily.EPL_EXTERNAL_SHEAR
    )
    z_lens = float(rng.uniform(0.1, 1.0))
    z_lower = _source_redshift_lower(z_lens, spec.z_max)
    if component is Component.LOW_Z:
        unit = float(rng.beta(1.0, 2.0))
        z_source = z_lower + unit * (spec.z_max - z_lower)
    else:
        z_source = float(rng.uniform(z_lower, spec.z_max))
    theta_e = _log_uniform_sample(rng, 0.3, 3.0)
    mass_1 = _power_law_sample(rng, -2.3, 20.0, 80.0)
    mass_ratio_lower = max(0.25, 10.0 / mass_1)
    return LatentDraw(
        component=component,
        lens_family=family,
        z_lens=z_lens,
        z_source=z_source,
        einstein_radius_arcsec=theta_e,
        axis_ratio=float(rng.uniform(0.4, 1.0)),
        position_angle_rad=float(rng.uniform(0.0, math.pi)),
        shear_amplitude=float(rng.uniform(0.0, 0.15)),
        shear_angle_rad=float(rng.uniform(0.0, math.pi)),
        density_slope=(
            2.0
            if family is LensFamily.SIE_EXTERNAL_SHEAR
            else float(rng.uniform(1.6, 2.5))
        ),
        external_convergence=float(rng.uniform(-0.15, 0.15)),
        source_u_x=_sample_source_coordinate(rng, component, spec),
        source_u_y=_sample_source_coordinate(rng, component, spec),
        mass_1_source=mass_1,
        mass_ratio=float(rng.uniform(mass_ratio_lower, 1.0)),
        a_1=float(0.99 * rng.beta(2.0, 5.0)),
        a_2=float(0.99 * rng.beta(2.0, 5.0)),
        tilt_1=float(math.acos(rng.uniform(-1.0, 1.0))),
        tilt_2=float(math.acos(rng.uniform(-1.0, 1.0))),
        phi_12=float(rng.uniform(0.0, 2.0 * math.pi)),
        phi_jl=float(rng.uniform(0.0, 2.0 * math.pi)),
        theta_jn=float(math.acos(rng.uniform(-1.0, 1.0))),
        phase=float(rng.uniform(0.0, 2.0 * math.pi)),
        ra=float(rng.uniform(0.0, 2.0 * math.pi)),
        dec=float(math.asin(rng.uniform(-1.0, 1.0))),
        psi=float(rng.uniform(0.0, math.pi)),
    )


def _source_log_density(draw: LatentDraw) -> float:
    mass_ratio_lower = max(0.25, 10.0 / draw.mass_1_source)
    angular_log = (
        math.log(math.sin(draw.tilt_1) / 2.0)
        if 0.0 < draw.tilt_1 < math.pi
        else -math.inf
    )
    angular_log += (
        math.log(math.sin(draw.tilt_2) / 2.0)
        if 0.0 < draw.tilt_2 < math.pi
        else -math.inf
    )
    angular_log += (
        math.log(math.sin(draw.theta_jn) / 2.0)
        if 0.0 < draw.theta_jn < math.pi
        else -math.inf
    )
    angular_log += (
        math.log(math.cos(draw.dec) / 2.0)
        if -0.5 * math.pi < draw.dec < 0.5 * math.pi
        else -math.inf
    )
    return (
        _power_law_log(draw.mass_1_source, -2.3, 20.0, 80.0)
        + _uniform_log(draw.mass_ratio, mass_ratio_lower, 1.0)
        + _beta_scaled_log(draw.a_1, 0.99)
        + _beta_scaled_log(draw.a_2, 0.99)
        + angular_log
        + _uniform_log(draw.phi_12, 0.0, 2.0 * math.pi)
        + _uniform_log(draw.phi_jl, 0.0, 2.0 * math.pi)
        + _uniform_log(draw.phase, 0.0, 2.0 * math.pi)
        + _uniform_log(draw.ra, 0.0, 2.0 * math.pi)
        + _uniform_log(draw.psi, 0.0, math.pi)
    )


def _broad_lens_log_without_redshift_or_source(draw: LatentDraw) -> float:
    if draw.lens_family not in {
        LensFamily.SIE_EXTERNAL_SHEAR,
        LensFamily.EPL_EXTERNAL_SHEAR,
    }:
        return -math.inf
    result = (
        -math.log(2.0)
        + _uniform_log(draw.z_lens, 0.1, 1.0)
        + _log_uniform_log(draw.einstein_radius_arcsec, 0.3, 3.0)
        + _uniform_log(draw.axis_ratio, 0.4, 1.0)
        + _uniform_log(draw.position_angle_rad, 0.0, math.pi)
        + _uniform_log(draw.shear_amplitude, 0.0, 0.15)
        + _uniform_log(draw.shear_angle_rad, 0.0, math.pi)
        + _uniform_log(draw.external_convergence, -0.15, 0.15)
    )
    if draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR:
        result += _uniform_log(draw.density_slope, 1.6, 2.5)
    elif draw.density_slope != 2.0:
        return -math.inf
    return result


def _source_plane_log(
    draw: LatentDraw, component: Component, spec: ProposalSpecification
) -> float:
    if component is Component.RC5_BROAD:
        x_log = _uniform_log(draw.source_u_x, spec.source_lower, spec.source_upper)
        y_log = _uniform_log(draw.source_u_y, spec.source_lower, spec.source_upper)
    else:
        sigma = spec.narrow_sigma if component is Component.NARROW else spec.wide_sigma
        x_log = truncated_normal_log_density(
            draw.source_u_x, 0.0, sigma, spec.source_lower, spec.source_upper
        )
        y_log = truncated_normal_log_density(
            draw.source_u_y, 0.0, sigma, spec.source_lower, spec.source_upper
        )
    if draw.einstein_radius_arcsec <= 0.0:
        return -math.inf
    return x_log + y_log - 2.0 * math.log(draw.einstein_radius_arcsec)


def log_component_density(
    draw: LatentDraw, component: Component, spec: ProposalSpecification
) -> float:
    lens_log = _broad_lens_log_without_redshift_or_source(draw)
    z_lower = _source_redshift_lower(draw.z_lens, spec.z_max)
    z_log = (
        low_z_log_density(draw.z_source, draw.z_lens, spec.z_max)
        if component is Component.LOW_Z
        else _uniform_log(draw.z_source, z_lower, spec.z_max)
    )
    return lens_log + z_log + _source_plane_log(draw, component, spec) + _source_log_density(draw)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def log_mixture_density(draw: LatentDraw, spec: ProposalSpecification) -> float:
    return _logsumexp(
        tuple(
            math.log(weight) + log_component_density(draw, component, spec)
            for component, weight in zip(COMPONENT_ORDER, spec.component_weights)
        )
    )


def log_evaluation_density(draw: LatentDraw, spec: ProposalSpecification) -> float:
    if draw.lens_family not in {
        LensFamily.SIE_EXTERNAL_SHEAR,
        LensFamily.EPL_EXTERNAL_SHEAR,
    }:
        return -math.inf
    theta_log = truncated_normal_log_density(
        math.log(draw.einstein_radius_arcsec),
        0.0,
        0.35,
        math.log(0.3),
        math.log(3.0),
    ) - math.log(draw.einstein_radius_arcsec)
    result = (
        -math.log(2.0)
        + _uniform_log(draw.z_lens, 0.1, 1.0)
        + _uniform_log(
            draw.z_source,
            _source_redshift_lower(draw.z_lens, spec.z_max),
            spec.z_max,
        )
        + theta_log
        + truncated_normal_log_density(draw.axis_ratio, 0.7, 0.15, 0.4, 1.0)
        + _uniform_log(draw.position_angle_rad, 0.0, math.pi)
        + truncated_normal_log_density(draw.shear_amplitude, 0.0, 0.05, 0.0, 0.15)
        + _uniform_log(draw.shear_angle_rad, 0.0, math.pi)
        + truncated_normal_log_density(
            draw.external_convergence, 0.0, 0.05, -0.15, 0.15
        )
        + _uniform_log(draw.source_u_x, spec.source_lower, spec.source_upper)
        + _uniform_log(draw.source_u_y, spec.source_lower, spec.source_upper)
        - 2.0 * math.log(draw.einstein_radius_arcsec)
        + _source_log_density(draw)
    )
    if draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR:
        result += truncated_normal_log_density(draw.density_slope, 2.08, 0.16, 1.6, 2.5)
    elif draw.density_slope != 2.0:
        return -math.inf
    return result


def log_importance_weight(draw: LatentDraw, spec: ProposalSpecification) -> float:
    return log_evaluation_density(draw, spec) - log_mixture_density(draw, spec)


def _relative_ess(weights: np.ndarray) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    squared = float(np.sum(np.square(weights), dtype=np.float64))
    return total * total / (len(weights) * squared) if squared > 0.0 else 0.0


def _replay_hash(spec: ProposalSpecification, count: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(sample_latent(rng, spec).replay_bytes())
    return digest.hexdigest()


def evaluate_latent_preflight(spec: ProposalSpecification) -> Dict[str, Any]:
    count = spec.preflight_draw_count
    rng = np.random.default_rng(spec.preflight_seed)
    log_q = np.empty(count, dtype=np.float64)
    log_p = np.empty(count, dtype=np.float64)
    family_epl = np.empty(count, dtype=np.bool_)
    component_counts = {component.value: 0 for component in COMPONENT_ORDER}
    digest = hashlib.sha256()
    support_holes = 0
    for index in range(count):
        draw = sample_latent(rng, spec)
        digest.update(draw.replay_bytes())
        component_counts[draw.component.value] += 1
        log_q[index] = log_mixture_density(draw, spec)
        log_p[index] = log_evaluation_density(draw, spec)
        family_epl[index] = draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR
        if math.isfinite(log_p[index]) and not math.isfinite(log_q[index]):
            support_holes += 1
    log_weights = log_p - log_q
    weights = np.exp(log_weights)
    finite_q = np.isfinite(log_q)
    finite_p = np.isfinite(log_p)
    finite_weights = np.isfinite(weights)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    normalized = weights / weight_sum
    replay_hash = _replay_hash(spec, count, spec.preflight_seed)
    first_hash = digest.hexdigest()
    family_ess = {
        LensFamily.SIE_EXTERNAL_SHEAR.value: _relative_ess(weights[~family_epl]),
        LensFamily.EPL_EXTERNAL_SHEAR.value: _relative_ess(weights[family_epl]),
    }
    mean_weight = float(np.mean(weights))
    overall_ess = _relative_ess(weights)
    reasons = []
    if not (bool(np.all(finite_q)) and bool(np.all(finite_p)) and bool(np.all(finite_weights))):
        reasons.append("nonfinite_log_density_or_weight")
    if not spec.mean_weight_bounds[0] <= mean_weight <= spec.mean_weight_bounds[1]:
        reasons.append("mean_importance_weight_outside_frozen_interval")
    if overall_ess < spec.overall_ess_minimum:
        reasons.append("overall_relative_ess_below_frozen_minimum")
    if any(value < spec.family_ess_minimum for value in family_ess.values()):
        reasons.append("lens_family_relative_ess_below_frozen_minimum")
    if support_holes:
        reasons.append("proposal_support_hole")
    if replay_hash != first_hash:
        reasons.append("deterministic_replay_mismatch")
    return {
        "status": "passed" if not reasons else "failed_hard_stop",
        "draw_count": count,
        "seed": spec.preflight_seed,
        "waveform_pair_count": 0,
        "accepted_pair_generator_called": False,
        "proposal_ab_qualification_run": False,
        "finite_fraction": {
            "log_q": float(np.mean(finite_q)),
            "log_p": float(np.mean(finite_p)),
            "weight": float(np.mean(finite_weights)),
        },
        "mean_unnormalized_importance_weight": mean_weight,
        "overall_relative_ess": overall_ess,
        "relative_ess_by_lens_family": family_ess,
        "maximum_normalized_weight": float(np.max(normalized)),
        "log_weight_quantiles": {
            str(quantile): float(np.quantile(log_weights, quantile))
            for quantile in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "weight_quantiles": {
            str(quantile): float(np.quantile(weights, quantile))
            for quantile in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "component_counts": component_counts,
        "support_hole_count": support_holes,
        "deterministic_replay_sha256": first_hash,
        "replay_byte_identical": replay_hash == first_hash,
        "frozen_thresholds": {
            "mean_weight_bounds": list(spec.mean_weight_bounds),
            "overall_relative_ess_minimum": spec.overall_ess_minimum,
            "per_lens_family_relative_ess_minimum": spec.family_ess_minimum,
        },
        "failure_reasons": reasons,
    }
