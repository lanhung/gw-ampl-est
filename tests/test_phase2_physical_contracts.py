from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.optional_solver
def test_preregistered_galkin_velocity_dispersion_converges():
    pytest.importorskip("lenstronomy")
    from astropy.cosmology import Planck18
    from lenstronomy.GalKin.galkin import Galkin

    config = load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")
    specification = config["stellar_kinematics_forward_model"]
    assert specification["light_model"] == "HERNQUIST"
    assert specification["anisotropy_model"] == "Osipkov_Merritt"
    assert specification["shortcut_forbidden"] == (
        "direct_inversion_from_einstein_radius_alone"
    )

    lens_redshift, source_redshift = 0.5, 1.5
    distance_lens = Planck18.angular_diameter_distance(lens_redshift).value
    distance_source = Planck18.angular_diameter_distance(source_redshift).value
    distance_lens_source = Planck18.angular_diameter_distance_z1z2(
        lens_redshift, source_redshift
    ).value
    galkin = Galkin(
        {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "OM",
        },
        {
            "aperture_type": "shell",
            "r_in": specification["aperture"]["inner_radius_arcsec"],
            "r_out": specification["aperture"]["outer_radius_arcsec"],
            "center_ra": 0.0,
            "center_dec": 0.0,
        },
        {"psf_type": "GAUSSIAN", "fwhm": specification["psf"]["fwhm_arcsec"]},
        {
            "d_d": distance_lens,
            "d_s": distance_source,
            "d_ds": distance_lens_source,
        },
    )
    kwargs_mass = [{"theta_E": 1.0, "gamma": 2.0, "center_x": 0.0, "center_y": 0.0}]
    kwargs_light = [
        {
            "amp": 1.0,
            "Rs": 1.0 / 1.8153,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    kwargs_anisotropy = {"r_ani": 1.0}

    values = []
    for samples in (
        specification["monte_carlo_samples"],
        specification["phase3a_convergence_check"]["reference_samples"],
    ):
        np.random.seed(20260712)
        values.append(
            float(
                galkin.dispersion(
                    kwargs_mass,
                    kwargs_light,
                    kwargs_anisotropy,
                    sampling_number=samples,
                )
            )
        )
    assert all(np.isfinite(value) and 100.0 < value < 500.0 for value in values)
    relative_difference = abs(values[0] - values[1]) / values[1]
    assert relative_difference <= specification["phase3a_convergence_check"][
        "maximum_relative_difference"
    ]
