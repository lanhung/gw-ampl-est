"""Balanced eight-cell noisy EM/environment observation generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ..physics.solver import PhysicalImage
from ..schema import (
    EMObservation,
    ExternalConvergenceObservation,
    ImageAstrometryObservation,
    ScalarObservation,
    TimingObservation,
)
from .dynamics import KinematicsDraw
from .population import PopulationDraw


@dataclass(frozen=True)
class GeneratedObservations:
    em_cell: str
    em: EMObservation
    timing: TimingObservation


def _hash(identifier: str) -> bytes:
    return hashlib.sha256(identifier.encode("utf-8")).digest()


def balanced_em_cell(
    accepted_index: int,
    physical_system_id: str,
    cell_names: Sequence[str],
) -> str:
    """Assign exact one-per-cell blocks using only stable physical-system IDs."""

    if accepted_index < 0 or not physical_system_id or len(cell_names) != 8:
        raise ValueError("balanced EM assignment requires index, system ID, and eight cells")
    block_start = accepted_index - accepted_index % 8
    suffix = f"{accepted_index:06d}"
    if not physical_system_id.endswith(suffix):
        raise ValueError("physical-system ID disagrees with accepted index")
    prefix = physical_system_id[: -len(suffix)]
    block_ids = [
        f"{prefix}{index:06d}"
        for index in range(block_start, block_start + 8)
    ]
    ranked = sorted(block_ids, key=_hash)
    rank = ranked.index(physical_system_id)
    return tuple(cell_names)[rank]


def _redshift_observation(
    rng: np.random.Generator, truth: float, specification: Mapping[str, Any] | None
) -> ScalarObservation | None:
    if specification is None:
        return None
    standard_deviation = float(specification["standard_deviation_scale_1_plus_z"]) * (
        1.0 + truth
    )
    return ScalarObservation(
        max(0.0, truth + float(rng.normal(0.0, standard_deviation))),
        standard_deviation,
    )


def generate_observations(
    *,
    rng: np.random.Generator,
    accepted_index: int,
    physical_system_id: str,
    draw: PopulationDraw,
    images: Sequence[PhysicalImage],
    selected_images: Tuple[PhysicalImage, PhysicalImage],
    kinematics: KinematicsDraw | None,
    em_model: Mapping[str, Any],
) -> GeneratedObservations:
    cells = em_model["availability_cells"]
    cell_name = balanced_em_cell(accepted_index, physical_system_id, tuple(cells))
    cell = cells[cell_name]
    modalities = set(cell["modalities"])
    astrometry_std = float(cell["astrometry_std_arcsec"])
    astrometry = (
        tuple(
            ImageAstrometryObservation(
                image.image_id,
                (
                    image.position_arcsec[0] + float(rng.normal(0.0, astrometry_std)),
                    image.position_arcsec[1] + float(rng.normal(0.0, astrometry_std)),
                ),
                ((astrometry_std**2, 0.0), (0.0, astrometry_std**2)),
            )
            for image in images
        )
        if "image_astrometry" in modalities
        else None
    )
    center_std = float(cell["lens_center_std_arcsec"])
    center = (
        (float(rng.normal(0.0, center_std)), float(rng.normal(0.0, center_std)))
        if "lens_center" in modalities
        else None
    )
    center_covariance = (
        ((center_std**2, 0.0), (0.0, center_std**2)) if center is not None else None
    )
    theta_e = float(draw.lens_parameters["einstein_radius_arcsec"])
    theta_fraction = cell["einstein_radius_fractional_std"]
    einstein = None
    if "einstein_radius" in modalities and theta_fraction is not None:
        theta_std = theta_e * float(theta_fraction)
        einstein = ScalarObservation(
            float(
                max(
                    float(np.finfo(float).tiny),
                    theta_e + float(rng.normal(0.0, theta_std)),
                )
            ),
            theta_std,
        )
    lens_redshift = _redshift_observation(rng, draw.z_lens, cell["lens_redshift"])
    source_redshift = _redshift_observation(rng, draw.z_source, cell["source_redshift"])
    velocity = None
    velocity_fraction = cell["velocity_dispersion_fractional_std"]
    if "velocity_dispersion" in modalities:
        if kinematics is None or velocity_fraction is None:
            raise ValueError("available velocity dispersion requires a Galkin draw")
        velocity_std = kinematics.velocity_dispersion_km_s * float(velocity_fraction)
        velocity = ScalarObservation(
            float(
                max(
                    float(np.finfo(float).tiny),
                    kinematics.velocity_dispersion_km_s
                    + float(rng.normal(0.0, velocity_std)),
                )
            ),
            velocity_std,
        )
    environment = None
    environment_state = str(cell["environment_state"])
    if "environment_convergence" in modalities:
        standard_deviation = float(
            em_model["environment_observation"][
                f"{environment_state}_standard_deviation"
            ]
        )
        environment = ExternalConvergenceObservation(
            draw.external_convergence + float(rng.normal(0.0, standard_deviation)),
            standard_deviation,
            "synthetic_line_of_sight_posterior_v1",
            None,
        )
    values = {
        "image_astrometry": astrometry,
        "lens_center": center,
        "einstein_radius": einstein,
        "lens_redshift": lens_redshift,
        "source_redshift": source_redshift,
        "velocity_dispersion": velocity,
        "environment_convergence": environment,
    }
    em = EMObservation(
        astrometry,
        center,
        center_covariance,
        einstein,
        lens_redshift,
        source_redshift,
        velocity,
        {name: value is not None for name, value in values.items()},
        {image.image_id: False for image in images},
        {
            "aperture_inner_radius_arcsec": 0.0,
            "aperture_outer_radius_arcsec": 1.0,
            "seeing_fwhm_arcsec": 0.7,
        },
        (
            lens_redshift.value < source_redshift.value
            if lens_redshift is not None and source_redshift is not None
            else None
        ),
        environment,
        kinematics.effective_radius_arcsec if velocity is not None and kinematics else None,
        kinematics.model_reference if velocity is not None and kinematics else None,
    )
    primary, secondary = selected_images
    if primary.arrival_time_seconds is None or secondary.arrival_time_seconds is None:
        raise ValueError("timing observation requires physical arrival times")
    timing_std = float(cell["timing_std_seconds"])
    delay = secondary.arrival_time_seconds - primary.arrival_time_seconds
    timing = TimingObservation(
        delay + float(rng.normal(0.0, timing_std)),
        timing_std,
        "synthetic_paired_trigger_geocentric_time",
        None,
    )
    return GeneratedObservations(cell_name, em, timing)
