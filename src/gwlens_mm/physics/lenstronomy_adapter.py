"""Optional lenstronomy adapters for SIE/EPL plus external shear."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

from .quantities import ImageParity, LensFamily, MorseClass
from .solver import LensSystemSolution, PhysicalImage


def _load_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    lens_model = importlib.import_module("lenstronomy.LensModel.lens_model").LensModel
    solver = importlib.import_module(
        "lenstronomy.LensModel.Solver.lens_equation_solver"
    ).LensEquationSolver
    lens_cosmo = importlib.import_module("lenstronomy.Cosmo.lens_cosmo").LensCosmo
    ellipticity = importlib.import_module("lenstronomy.Util.param_util").phi_q2_ellipticity
    planck18 = importlib.import_module("astropy.cosmology").Planck18
    return lens_model, solver, lens_cosmo, ellipticity, planck18


@dataclass(frozen=True)
class LenstronomyAdapter:
    """Return every physical image, sorted by Fermat-potential arrival order."""

    lens_family: LensFamily
    z_lens: float
    z_source: float

    def __post_init__(self) -> None:
        if self.lens_family not in {
            LensFamily.SIE_EXTERNAL_SHEAR,
            LensFamily.EPL_EXTERNAL_SHEAR,
        }:
            raise ValueError("lenstronomy adapter supports only SIE/EPL plus shear")
        if not (0 <= self.z_lens < self.z_source):
            raise ValueError("lens/source redshifts must satisfy 0 <= z_lens < z_source")

    def _model(self, parameters: Mapping[str, Any]) -> Tuple[Any, list[dict[str, float]]]:
        lens_model_class, _, _, ellipticity_converter, _ = _load_dependencies()
        axis_ratio = float(parameters["axis_ratio"])
        if not 0 < axis_ratio <= 1:
            raise ValueError("axis_ratio must satisfy 0 < q <= 1")
        e1, e2 = ellipticity_converter(
            float(parameters.get("position_angle_rad", 0.0)), axis_ratio
        )
        mass: dict[str, float] = {
            "theta_E": float(parameters["einstein_radius_arcsec"]),
            "e1": float(e1),
            "e2": float(e2),
            "center_x": float(parameters.get("center_x_arcsec", 0.0)),
            "center_y": float(parameters.get("center_y_arcsec", 0.0)),
        }
        profile = "SIE"
        if self.lens_family is LensFamily.EPL_EXTERNAL_SHEAR:
            profile = "EPL"
            mass["gamma"] = float(parameters["density_slope"])
        shear = {
            "gamma1": float(parameters["shear_gamma1"]),
            "gamma2": float(parameters["shear_gamma2"]),
            "ra_0": float(parameters.get("center_x_arcsec", 0.0)),
            "dec_0": float(parameters.get("center_y_arcsec", 0.0)),
        }
        return lens_model_class([profile, "SHEAR"]), [mass, shear]

    @staticmethod
    def _morse_class(
        lens_model: Any,
        x: float,
        y: float,
        kwargs: list[dict[str, float]],
    ) -> MorseClass:
        f_xx, f_xy, f_yx, f_yy = lens_model.hessian(x, y, kwargs)
        arrival_hessian = np.array(
            [[1.0 - f_xx, -f_xy], [-f_yx, 1.0 - f_yy]], dtype=float
        )
        eigenvalues = np.linalg.eigvalsh(arrival_hessian)
        tolerance = 1e-8
        if np.all(eigenvalues > tolerance):
            return MorseClass.MINIMUM
        if np.all(eigenvalues < -tolerance):
            return MorseClass.MAXIMUM
        if eigenvalues[0] < -tolerance < eigenvalues[1]:
            return MorseClass.SADDLE
        raise ValueError("degenerate image Hessian has undefined Morse class")

    def solve(
        self,
        source_position: Tuple[float, float],
        parameters: Mapping[str, Any],
    ) -> LensSystemSolution:
        beta_x, beta_y = (float(value) for value in source_position)
        if not all(math.isfinite(value) for value in (beta_x, beta_y)):
            raise ValueError("source position must be finite")
        lens_model, kwargs_lens = self._model(parameters)
        _, solver_class, lens_cosmo_class, _, planck18 = _load_dependencies()
        solver = solver_class(lens_model)
        x_image, y_image = solver.image_position_from_source(
            beta_x,
            beta_y,
            kwargs_lens,
            search_window=float(parameters.get("search_window_arcsec", 5.0)),
            min_distance=float(parameters.get("minimum_image_separation_arcsec", 0.01)),
            num_iter_max=int(parameters.get("solver_iterations", 200)),
        )
        if len(x_image) < 2:
            return LensSystemSolution(
                lens_family=self.lens_family,
                physical_images=(),
                solver_name="lenstronomy",
                solver_version=str(importlib.metadata.version("lenstronomy")),
                valid=False,
                validity_reason="fewer than two images",
            )
        magnifications = np.asarray(
            lens_model.magnification(x_image, y_image, kwargs_lens), dtype=float
        )
        fermat = np.asarray(
            lens_model.fermat_potential(
                x_image,
                y_image,
                kwargs_lens,
                x_source=beta_x,
                y_source=beta_y,
            ),
            dtype=float,
        )
        order = np.argsort(fermat)
        lens_cosmo = lens_cosmo_class(self.z_lens, self.z_source, cosmo=planck18)
        delay_days = np.asarray(lens_cosmo.time_delay_units(fermat), dtype=float)
        delay_seconds = (delay_days - np.min(delay_days)) * 86400.0
        images = []
        for output_index, raw_index in enumerate(order):
            mu_signed = float(magnifications[raw_index])
            morse = self._morse_class(
                lens_model,
                float(x_image[raw_index]),
                float(y_image[raw_index]),
                kwargs_lens,
            )
            parity = ImageParity.POSITIVE if mu_signed > 0 else ImageParity.NEGATIVE
            if (morse is MorseClass.SADDLE) != (parity is ImageParity.NEGATIVE):
                raise ValueError("Morse class and signed-magnification parity disagree")
            images.append(
                PhysicalImage(
                    image_id=f"image_{output_index}",
                    position_arcsec=(
                        float(x_image[raw_index]),
                        float(y_image[raw_index]),
                    ),
                    mu_signed=mu_signed,
                    parity=parity,
                    morse_class=morse,
                    fermat_potential_dimensionless=float(fermat[raw_index]),
                    arrival_time_seconds=float(delay_seconds[raw_index]),
                )
            )
        return LensSystemSolution(
            lens_family=self.lens_family,
            physical_images=tuple(images),
            solver_name="lenstronomy",
            solver_version=str(importlib.metadata.version("lenstronomy")),
        )
