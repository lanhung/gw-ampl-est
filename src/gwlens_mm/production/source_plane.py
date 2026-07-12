"""RC.4 source-plane measure and primary/reference solver audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ..physics.lenstronomy_adapter import LenstronomyAdapter
from ..physics.quantities import LensFamily
from ..physics.solver import LensSystemSolution


def normalized_source_log_density(einstein_radius_arcsec: float) -> float:
    theta_e = float(einstein_radius_arcsec)
    if not math.isfinite(theta_e) or theta_e <= 0:
        raise ValueError("Einstein radius must be positive and finite")
    return -math.log(25.0) - 2.0 * math.log(theta_e)


def sample_source_position(
    rng: np.random.Generator, einstein_radius_arcsec: float
) -> Tuple[float, float]:
    theta_e = float(einstein_radius_arcsec)
    normalized_source_log_density(theta_e)
    values = rng.uniform(-2.5, 2.5, size=2) * theta_e
    return float(values[0]), float(values[1])


def boundary_points(points_per_edge: int) -> Tuple[Tuple[float, float], ...]:
    if points_per_edge < 2:
        raise ValueError("source boundary requires at least two points per edge")
    upper = float(np.nextafter(2.5, -math.inf))
    coordinates = np.linspace(-2.5, upper, points_per_edge, endpoint=True)
    points: list[Tuple[float, float]] = []
    for value in coordinates:
        points.extend(
            (
                (-2.5, float(value)),
                (upper, float(value)),
                (float(value), -2.5),
                (float(value), upper),
            )
        )
    return tuple(dict.fromkeys(points))


@dataclass(frozen=True)
class SolverAgreement:
    primary_valid: bool
    reference_valid: bool
    primary_image_count: int
    reference_image_count: int
    maximum_position_difference_arcsec: float
    maximum_relative_magnification_difference: float
    passed: bool
    reason: str


def _nearest_match_metrics(
    primary: LensSystemSolution, reference: LensSystemSolution
) -> Tuple[float, float]:
    unmatched = list(reference.physical_images)
    maximum_position = 0.0
    maximum_magnification = 0.0
    for image in primary.physical_images:
        distances = [
            math.hypot(
                image.position_arcsec[0] - candidate.position_arcsec[0],
                image.position_arcsec[1] - candidate.position_arcsec[1],
            )
            for candidate in unmatched
        ]
        match_index = int(np.argmin(distances))
        match = unmatched.pop(match_index)
        maximum_position = max(maximum_position, distances[match_index])
        scale = max(abs(match.mu_signed), 1.0e-300)
        maximum_magnification = max(
            maximum_magnification,
            abs(image.mu_signed - match.mu_signed) / scale,
        )
    return maximum_position, maximum_magnification


def compare_solver_unions(
    family: LensFamily,
    z_lens: float,
    z_source: float,
    source_position: Tuple[float, float],
    lens_parameters: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
) -> SolverAgreement:
    audit = numerical_contract["support_audit"]
    reference_contract = {
        **numerical_contract,
        "components": audit["reference_components"],
    }
    primary_parameters = {**lens_parameters, "numerical_contract": numerical_contract}
    reference_parameters = {**lens_parameters, "numerical_contract": reference_contract}
    adapter = LenstronomyAdapter(family, z_lens, z_source)
    primary = adapter.solve(source_position, primary_parameters)
    reference = adapter.solve(source_position, reference_parameters)
    primary_count = len(primary.physical_images) if primary.valid else 0
    reference_count = len(reference.physical_images) if reference.valid else 0
    if primary_count != reference_count:
        return SolverAgreement(
            primary.valid,
            reference.valid,
            primary_count,
            reference_count,
            math.inf,
            math.inf,
            False,
            "image_multiplicity_mismatch",
        )
    if primary_count == 0:
        return SolverAgreement(False, False, 0, 0, 0.0, 0.0, True, "both_single_image")
    position, magnification = _nearest_match_metrics(primary, reference)
    passed = position <= float(audit["maximum_position_difference_arcsec"]) and (
        magnification <= float(audit["maximum_relative_magnification_difference"])
    )
    return SolverAgreement(
        primary.valid,
        reference.valid,
        primary_count,
        reference_count,
        position,
        magnification,
        passed,
        "matched" if passed else "image_value_mismatch",
    )


def scaled_boundary_points(
    points: Sequence[Tuple[float, float]], einstein_radius_arcsec: float
) -> Tuple[Tuple[float, float], ...]:
    theta_e = float(einstein_radius_arcsec)
    normalized_source_log_density(theta_e)
    return tuple((x * theta_e, y * theta_e) for x, y in points)
