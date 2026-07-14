"""Exact frozen latent distributions for final-evaluation diagnostics."""

from __future__ import annotations

import math
from dataclasses import replace
from enum import Enum
from typing import Tuple

import numpy as np

from ..physics.quantities import LensFamily
from .exact_mixture import LatentDraw, _sample_truncated_normal, truncated_normal_log_density
from .target_anchored import (
    TargetAnchoredSpecification,
    log_target_density,
    sample_evaluation_target,
)


class ParameterOODStratum(str, Enum):
    SLOPE_OUTSIDE_TRAINING = "slope_outside_training"
    EXTREME_FLATTENING = "extreme_flattening"
    HIGH_EXTERNAL_SHEAR = "high_external_shear"
    EXTREME_EXTERNAL_CONVERGENCE = "extreme_external_convergence"


def _uniform_log(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value) or not lower <= value < upper:
        return -math.inf
    return -math.log(upper - lower)


def sample_target_conditioned_family(
    rng: np.random.Generator,
    specification: TargetAnchoredSpecification,
    family: LensFamily,
) -> Tuple[LatentDraw, float]:
    """Draw the evaluation target conditional on one declared lens family."""

    draw = sample_evaluation_target(rng, specification)
    if family is LensFamily.SIE_EXTERNAL_SHEAR:
        draw = replace(draw, lens_family=family, density_slope=2.0)
    else:
        draw = replace(
            draw,
            lens_family=family,
            density_slope=_sample_truncated_normal(rng, 2.08, 0.16, 1.6, 2.5),
        )
    log_density = log_target_density(draw, specification) + math.log(2.0)
    if not math.isfinite(log_density):
        raise ValueError("conditional-family target draw has nonfinite density")
    return draw, log_density


def _sample_signed_outer_interval(rng: np.random.Generator) -> float:
    magnitude = _sample_left_open_right_closed(rng, 0.15, 0.25)
    return magnitude if rng.random() < 0.5 else -magnitude


def _sample_left_open_right_closed(
    rng: np.random.Generator, lower: float, upper: float
) -> float:
    """Sample the preregistered ``(lower, upper]`` floating-point interval."""

    return float(upper - float(rng.random()) * (upper - lower))


def sample_parameter_ood(
    rng: np.random.Generator,
    specification: TargetAnchoredSpecification,
    stratum: ParameterOODStratum,
) -> Tuple[LatentDraw, float]:
    """Draw one exact target-anchored OOD stratum and return its log density."""

    draw = sample_evaluation_target(rng, specification)
    if stratum is ParameterOODStratum.SLOPE_OUTSIDE_TRAINING:
        low_branch = rng.random() < 0.5
        slope = (
            float(rng.uniform(1.4, 1.6))
            if low_branch
            else _sample_left_open_right_closed(rng, 2.5, 2.7)
        )
        draw = replace(draw, lens_family=LensFamily.EPL_EXTERNAL_SHEAR, density_slope=slope)
    elif stratum is ParameterOODStratum.EXTREME_FLATTENING:
        draw = replace(draw, axis_ratio=float(rng.uniform(0.25, 0.4)))
    elif stratum is ParameterOODStratum.HIGH_EXTERNAL_SHEAR:
        draw = replace(
            draw,
            shear_amplitude=_sample_left_open_right_closed(rng, 0.15, 0.25),
        )
    elif stratum is ParameterOODStratum.EXTREME_EXTERNAL_CONVERGENCE:
        draw = replace(draw, external_convergence=_sample_signed_outer_interval(rng))
    else:  # pragma: no cover - exhaustive enum defense
        raise ValueError(f"unknown OOD stratum: {stratum}")
    log_density = log_parameter_ood_density(draw, specification, stratum)
    if not math.isfinite(log_density):
        raise ValueError("parameter-OOD sampler and density disagree")
    return draw, log_density


def log_parameter_ood_density(
    draw: LatentDraw,
    specification: TargetAnchoredSpecification,
    stratum: ParameterOODStratum,
) -> float:
    """Evaluate the normalized OOD density on the complete latent state."""

    if stratum is ParameterOODStratum.SLOPE_OUTSIDE_TRAINING:
        in_support = (1.4 <= draw.density_slope < 1.6) or (
            2.5 < draw.density_slope <= 2.7
        )
        if draw.lens_family is not LensFamily.EPL_EXTERNAL_SHEAR or not in_support:
            return -math.inf
        reference = replace(draw, density_slope=2.08)
        target_factor = -math.log(2.0) + truncated_normal_log_density(
            2.08, 2.08, 0.16, 1.6, 2.5
        )
        return log_target_density(reference, specification) - target_factor + math.log(2.5)
    if stratum is ParameterOODStratum.EXTREME_FLATTENING:
        if not 0.25 <= draw.axis_ratio < 0.4:
            return -math.inf
        reference = replace(draw, axis_ratio=0.7)
        target_factor = truncated_normal_log_density(0.7, 0.7, 0.15, 0.4, 1.0)
        return (
            log_target_density(reference, specification)
            - target_factor
            + _uniform_log(draw.axis_ratio, 0.25, 0.4)
        )
    if stratum is ParameterOODStratum.HIGH_EXTERNAL_SHEAR:
        if not 0.15 < draw.shear_amplitude <= 0.25:
            return -math.inf
        reference = replace(draw, shear_amplitude=0.05)
        target_factor = truncated_normal_log_density(0.05, 0.0, 0.05, 0.0, 0.15)
        return (
            log_target_density(reference, specification)
            - target_factor
            + math.log(10.0)
        )
    if stratum is ParameterOODStratum.EXTREME_EXTERNAL_CONVERGENCE:
        magnitude = abs(draw.external_convergence)
        if not 0.15 < magnitude <= 0.25:
            return -math.inf
        reference = replace(draw, external_convergence=0.0)
        target_factor = truncated_normal_log_density(0.0, 0.0, 0.05, -0.15, 0.15)
        return log_target_density(reference, specification) - target_factor + math.log(5.0)
    raise ValueError(f"unknown OOD stratum: {stratum}")
