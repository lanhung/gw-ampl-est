"""Frozen RC.4 proposal/evaluation population sampling and exact log densities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from ..physics.quantities import LensFamily
from .source_plane import normalized_source_log_density, sample_source_position


def _uniform_log(value: float, lower: float, upper: float) -> float:
    if not lower <= value <= upper or upper <= lower:
        return -math.inf
    return -math.log(upper - lower)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _truncated_normal_log(
    value: float, mean: float, standard_deviation: float, lower: float, upper: float
) -> float:
    if not lower <= value <= upper:
        return -math.inf
    z = (value - mean) / standard_deviation
    normalization = _normal_cdf((upper - mean) / standard_deviation) - _normal_cdf(
        (lower - mean) / standard_deviation
    )
    return (
        -0.5 * z**2
        - math.log(standard_deviation)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(normalization)
    )


def _log_uniform_sample(rng: np.random.Generator, lower: float, upper: float) -> float:
    return float(math.exp(rng.uniform(math.log(lower), math.log(upper))))


def _log_uniform_log(value: float, lower: float, upper: float) -> float:
    if not lower <= value <= upper:
        return -math.inf
    return -math.log(value) - math.log(math.log(upper / lower))


def _power_law_sample(
    rng: np.random.Generator, exponent: float, lower: float, upper: float
) -> float:
    power = exponent + 1.0
    return float(
        (rng.random() * (upper**power - lower**power) + lower**power) ** (1.0 / power)
    )


def _power_law_log(value: float, exponent: float, lower: float, upper: float) -> float:
    if not lower <= value <= upper:
        return -math.inf
    power = exponent + 1.0
    normalization = power / (upper**power - lower**power)
    return math.log(normalization) + exponent * math.log(value)


def _beta_scaled_log(value: float, maximum: float, alpha: float = 2.0, beta: float = 5.0) -> float:
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


@dataclass(frozen=True)
class PopulationDraw:
    lens_family: LensFamily
    z_lens: float
    z_source: float
    lens_parameters: Mapping[str, float]
    source_position_arcsec: tuple[float, float]
    external_convergence: float
    source_parameters: Mapping[str, float]
    proposal_log_probability: float
    evaluation_log_probability: float

    @property
    def importance_weight(self) -> float:
        difference = self.evaluation_log_probability - self.proposal_log_probability
        result = math.exp(difference)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("importance weight must be positive and finite")
        return result


def _evaluation_lens_log(draw: Dict[str, float], family: LensFamily) -> float:
    theta = draw["einstein_radius_arcsec"]
    log_theta = math.log(theta)
    theta_log = _truncated_normal_log(
        log_theta, 0.0, 0.35, math.log(0.3), math.log(3.0)
    ) - math.log(theta)
    shear_log = _truncated_normal_log(draw["shear_amplitude"], 0.0, 0.05, 0.0, 0.15)
    result = (
        -math.log(2.0)
        + _uniform_log(draw["z_lens"], 0.1, 1.0)
        + _uniform_log(draw["z_source"], max(0.5, draw["z_lens"] + 0.1), 3.0)
        + theta_log
        + _truncated_normal_log(draw["axis_ratio"], 0.7, 0.15, 0.4, 1.0)
        + _uniform_log(draw["position_angle_rad"], 0.0, math.pi)
        + shear_log
        + _uniform_log(draw["shear_angle_rad"], 0.0, math.pi)
        + _truncated_normal_log(draw["external_convergence"], 0.0, 0.05, -0.15, 0.15)
        + normalized_source_log_density(theta)
    )
    if family is LensFamily.EPL_EXTERNAL_SHEAR:
        result += _truncated_normal_log(draw["density_slope"], 2.08, 0.16, 1.6, 2.5)
    return result


def sample_population(
    rng: np.random.Generator,
    source_rng: np.random.Generator | None = None,
) -> PopulationDraw:
    family = (
        LensFamily.SIE_EXTERNAL_SHEAR
        if rng.random() < 0.5
        else LensFamily.EPL_EXTERNAL_SHEAR
    )
    z_lens = float(rng.uniform(0.1, 1.0))
    z_source_lower = max(0.5, z_lens + 0.1)
    z_source = float(rng.uniform(z_source_lower, 3.0))
    theta_e = _log_uniform_sample(rng, 0.3, 3.0)
    axis_ratio = float(rng.uniform(0.4, 1.0))
    position_angle = float(rng.uniform(0.0, math.pi))
    shear = float(rng.uniform(0.0, 0.15))
    shear_angle = float(rng.uniform(0.0, math.pi))
    density_slope = 2.0 if family is LensFamily.SIE_EXTERNAL_SHEAR else float(
        rng.uniform(1.6, 2.5)
    )
    external_convergence = float(rng.uniform(-0.15, 0.15))
    source_position = sample_source_position(rng, theta_e)
    lens_values = {
        "z_lens": z_lens,
        "z_source": z_source,
        "einstein_radius_arcsec": theta_e,
        "axis_ratio": axis_ratio,
        "position_angle_rad": position_angle,
        "shear_amplitude": shear,
        "shear_angle_rad": shear_angle,
        "shear_gamma1": shear * math.cos(2.0 * shear_angle),
        "shear_gamma2": shear * math.sin(2.0 * shear_angle),
        "density_slope": density_slope,
        "external_convergence": external_convergence,
    }
    proposal_lens_log = (
        -math.log(2.0)
        + _uniform_log(z_lens, 0.1, 1.0)
        + _uniform_log(z_source, z_source_lower, 3.0)
        + _log_uniform_log(theta_e, 0.3, 3.0)
        + _uniform_log(axis_ratio, 0.4, 1.0)
        + _uniform_log(position_angle, 0.0, math.pi)
        + _uniform_log(shear, 0.0, 0.15)
        + _uniform_log(shear_angle, 0.0, math.pi)
        + _uniform_log(external_convergence, -0.15, 0.15)
        + normalized_source_log_density(theta_e)
    )
    if family is LensFamily.EPL_EXTERNAL_SHEAR:
        proposal_lens_log += _uniform_log(density_slope, 1.6, 2.5)

    source_generator = rng if source_rng is None else source_rng
    mass_1 = _power_law_sample(source_generator, -2.3, 20.0, 80.0)
    q_lower = max(0.25, 10.0 / mass_1)
    mass_ratio = float(source_generator.uniform(q_lower, 1.0))
    mass_2 = mass_1 * mass_ratio
    a_1 = float(0.99 * source_generator.beta(2.0, 5.0))
    a_2 = float(0.99 * source_generator.beta(2.0, 5.0))
    tilt_1 = float(math.acos(source_generator.uniform(-1.0, 1.0)))
    tilt_2 = float(math.acos(source_generator.uniform(-1.0, 1.0)))
    theta_jn = float(math.acos(source_generator.uniform(-1.0, 1.0)))
    dec = float(math.asin(source_generator.uniform(-1.0, 1.0)))
    source = {
        "mass_1_source": mass_1,
        "mass_2_source": mass_2,
        "mass_ratio": mass_ratio,
        "a_1": a_1,
        "a_2": a_2,
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "phi_12": float(source_generator.uniform(0.0, 2.0 * math.pi)),
        "phi_jl": float(source_generator.uniform(0.0, 2.0 * math.pi)),
        "theta_jn": theta_jn,
        "phase": float(source_generator.uniform(0.0, 2.0 * math.pi)),
        "ra": float(source_generator.uniform(0.0, 2.0 * math.pi)),
        "dec": dec,
        "psi": float(source_generator.uniform(0.0, math.pi)),
    }
    source_log = (
        _power_law_log(mass_1, -2.3, 20.0, 80.0)
        + _uniform_log(mass_ratio, q_lower, 1.0)
        + _beta_scaled_log(a_1, 0.99)
        + _beta_scaled_log(a_2, 0.99)
        + math.log(max(math.sin(tilt_1), 1.0e-300) / 2.0)
        + math.log(max(math.sin(tilt_2), 1.0e-300) / 2.0)
        + math.log(max(math.sin(theta_jn), 1.0e-300) / 2.0)
        + math.log(max(math.cos(dec), 1.0e-300) / 2.0)
        - math.log(2.0 * math.pi) * 4.0
        - math.log(math.pi)
    )
    return PopulationDraw(
        family,
        z_lens,
        z_source,
        lens_values,
        source_position,
        external_convergence,
        source,
        proposal_lens_log + source_log,
        _evaluation_lens_log(lens_values, family) + source_log,
    )
