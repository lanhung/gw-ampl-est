"""Frozen Lenstronomy Galkin stellar-kinematics forward model."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np

from ..physics.quantities import LensFamily


@dataclass(frozen=True)
class KinematicsDraw:
    velocity_dispersion_km_s: float
    effective_radius_arcsec: float
    anisotropy_radius_arcsec: float
    density_slope: float
    monte_carlo_samples: int
    seed: int
    model_reference: str = "lenstronomy_galkin_sphericalpowerlaw_hernquist_om_v1"


def sample_kinematics_nuisance(
    rng: np.random.Generator,
) -> tuple[float, float]:
    effective_radius = float(math.exp(rng.uniform(math.log(0.3), math.log(2.0))))
    anisotropy_ratio = float(math.exp(rng.uniform(math.log(0.5), math.log(5.0))))
    return effective_radius, anisotropy_ratio


def galkin_velocity_dispersion(
    *,
    lens_family: LensFamily,
    z_lens: float,
    z_source: float,
    einstein_radius_arcsec: float,
    density_slope: float,
    effective_radius_arcsec: float,
    anisotropy_radius_over_effective_radius: float,
    monte_carlo_samples: int,
    seed: int,
) -> KinematicsDraw:
    values = (
        z_lens,
        z_source,
        einstein_radius_arcsec,
        density_slope,
        effective_radius_arcsec,
        anisotropy_radius_over_effective_radius,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("kinematics parameters must be finite")
    if not 0 <= z_lens < z_source or min(values[2:]) <= 0 or monte_carlo_samples <= 0:
        raise ValueError("kinematics parameters are outside physical support")
    slope = 2.0 if lens_family is LensFamily.SIE_EXTERNAL_SHEAR else density_slope
    cosmology = importlib.import_module("astropy.cosmology").Planck18
    galkin_class = importlib.import_module("lenstronomy.GalKin.galkin").Galkin
    distance_lens = float(cosmology.angular_diameter_distance(z_lens).value)
    distance_source = float(cosmology.angular_diameter_distance(z_source).value)
    distance_lens_source = float(
        cosmology.angular_diameter_distance_z1z2(z_lens, z_source).value
    )
    galkin = galkin_class(
        {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "OM",
        },
        {
            "aperture_type": "shell",
            "r_in": 0.0,
            "r_out": 1.0,
            "center_ra": 0.0,
            "center_dec": 0.0,
        },
        {"psf_type": "GAUSSIAN", "fwhm": 0.7},
        {
            "d_d": distance_lens,
            "d_s": distance_source,
            "d_ds": distance_lens_source,
        },
    )
    anisotropy_radius = (
        anisotropy_radius_over_effective_radius * effective_radius_arcsec
    )
    # Galkin 1.13.6 uses NumPy's legacy global generator. Phase 3A calls this
    # function only inside an owned worker process, never concurrently in threads.
    np.random.seed(seed % (2**32))
    dispersion = float(
        galkin.dispersion(
            [
                {
                    "theta_E": einstein_radius_arcsec,
                    "gamma": slope,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ],
            [
                {
                    "amp": 1.0,
                    "Rs": effective_radius_arcsec / 1.8153,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ],
            {"r_ani": anisotropy_radius},
            sampling_number=monte_carlo_samples,
        )
    )
    if not math.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("Galkin returned invalid velocity dispersion")
    return KinematicsDraw(
        dispersion,
        effective_radius_arcsec,
        anisotropy_radius,
        slope,
        monte_carlo_samples,
        seed,
    )
