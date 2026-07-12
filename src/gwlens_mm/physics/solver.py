"""Implementation-independent lens solver and pair-selection contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from .quantities import ImageParity, LensFamily, Magnification, MorseClass, PrimaryDefinition
from .sis import sis_from_source_position


@dataclass(frozen=True)
class PhysicalImage:
    image_id: str
    position_arcsec: Tuple[float, float]
    mu_signed: float
    parity: ImageParity
    morse_class: MorseClass
    fermat_potential_dimensionless: Optional[float] = None
    arrival_time_seconds: Optional[float] = None
    valid: bool = True
    validity_reason: str = "valid"

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id is required")
        if len(self.position_arcsec) != 2 or not all(
            math.isfinite(v) for v in self.position_arcsec
        ):
            raise ValueError("position_arcsec must contain two finite values")
        magnification = Magnification(self.mu_signed)
        if magnification.parity is not self.parity:
            raise ValueError("parity is inconsistent with signed magnification")
        coordinates = (
            self.fermat_potential_dimensionless,
            self.arrival_time_seconds,
        )
        if all(value is None for value in coordinates):
            raise ValueError("a physical image requires a Fermat or physical-time coordinate")
        if any(value is not None and not math.isfinite(value) for value in coordinates):
            raise ValueError("supplied Fermat and physical-time coordinates must be finite")


@dataclass(frozen=True)
class LensSystemSolution:
    lens_family: LensFamily
    physical_images: Tuple[PhysicalImage, ...]
    solver_name: str
    solver_version: str
    valid: bool = True
    validity_reason: str = "valid"

    def __post_init__(self) -> None:
        ids = [image.image_id for image in self.physical_images]
        if len(ids) != len(set(ids)):
            raise ValueError("physical image IDs must be unique")
        if self.valid and len(ids) < 2:
            raise ValueError("a valid strong-lensing solution must contain at least two images")


@dataclass(frozen=True)
class SelectedPair:
    primary_image_id: str
    secondary_image_id: str
    primary_definition: PrimaryDefinition
    selection_reason: str
    detector_visibility: Mapping[str, Tuple[bool, bool]]
    unselected_image_ids: Tuple[str, ...] = ()
    censored_image_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.primary_image_id == self.secondary_image_id:
            raise ValueError("selected image IDs must differ")
        if not self.selection_reason:
            raise ValueError("selection_reason is required")

    def validate_against(self, solution: LensSystemSolution) -> None:
        physical = {image.image_id for image in solution.physical_images}
        selected = {self.primary_image_id, self.secondary_image_id}
        if not selected <= physical:
            raise ValueError("selected pair references an unknown physical image")
        extras = set(self.unselected_image_ids) | set(self.censored_image_ids)
        if not extras <= physical - selected:
            raise ValueError("unselected/censored IDs must be non-selected physical images")
        if set(self.unselected_image_ids) & set(self.censored_image_ids):
            raise ValueError("an extra image cannot be both unselected and censored")


class LensSolver(Protocol):
    """Adapter contract; implementations must return all physical images."""

    @property
    def lens_family(self) -> LensFamily: ...

    def solve(
        self, source_position: Tuple[float, float], parameters: Mapping[str, Any]
    ) -> LensSystemSolution: ...


class SISSolver:
    """Dependency-free SIS control in angular units of theta_E."""

    lens_family = LensFamily.SIS

    def solve(
        self, source_position: Tuple[float, float], parameters: Mapping[str, Any]
    ) -> LensSystemSolution:
        beta_x, beta_y = (float(value) for value in source_position)
        theta_e = float(parameters.get("einstein_radius_arcsec", 1.0))
        if not math.isfinite(theta_e) or theta_e <= 0:
            raise ValueError("einstein_radius_arcsec must be positive and finite")
        beta_radius = math.hypot(beta_x, beta_y)
        if beta_radius == 0.0:
            raise ValueError("the exactly aligned SIS Einstein ring is not a two-image solution")
        y = beta_radius / theta_e
        analytic = sis_from_source_position(y)
        ux, uy = beta_x / beta_radius, beta_y / beta_radius
        x_plus = theta_e * (y + 1.0)
        x_minus = theta_e * (y - 1.0)
        images = (
            PhysicalImage(
                image_id="sis_plus",
                position_arcsec=(x_plus * ux, x_plus * uy),
                mu_signed=analytic.plus.mu_signed,
                parity=ImageParity.POSITIVE,
                morse_class=MorseClass.MINIMUM,
                fermat_potential_dimensionless=-0.5 - y,
            ),
            PhysicalImage(
                image_id="sis_minus",
                position_arcsec=(x_minus * ux, x_minus * uy),
                mu_signed=analytic.minus.mu_signed,
                parity=ImageParity.NEGATIVE,
                morse_class=MorseClass.SADDLE,
                fermat_potential_dimensionless=-0.5 + y,
            ),
        )
        return LensSystemSolution(
            lens_family=self.lens_family,
            physical_images=images,
            solver_name="gwlens_mm.SISSolver",
            solver_version="1",
        )


def validate_solver_contract(
    solver: LensSolver,
    cases: Sequence[Tuple[Tuple[float, float], Mapping[str, Any]]],
) -> None:
    """Run lightweight structural checks for an optional non-SIS adapter."""

    if solver.lens_family not in set(LensFamily):
        raise ValueError("solver reports an unsupported lens family")
    for source_position, parameters in cases:
        solution = solver.solve(source_position, parameters)
        if solution.lens_family is not solver.lens_family:
            raise ValueError("solver result family does not match adapter family")
        physical_times = [image.arrival_time_seconds for image in solution.physical_images]
        fermat_values = [
            image.fermat_potential_dimensionless for image in solution.physical_images
        ]
        if all(value is not None for value in physical_times):
            ordering = [float(value) for value in physical_times if value is not None]
        elif all(value is not None for value in fermat_values):
            ordering = [float(value) for value in fermat_values if value is not None]
        else:
            raise ValueError(
                "all images require one common explicit coordinate for ordering"
            )
        if ordering != sorted(ordering):
            raise ValueError("solver images must be returned in arrival/Fermat order")
        if not all(image.valid for image in solution.physical_images):
            raise ValueError("contract case returned an invalid image")


def apply_mass_sheet_transform(
    solution: LensSystemSolution,
    source_position: Tuple[float, float],
    external_convergence: float,
) -> Tuple[LensSystemSolution, Tuple[float, float]]:
    """Apply the explicit alpha.3 mass-sheet convention.

    ``lambda_mst = 1 - external_convergence``. Image positions, parity, and
    Morse class are invariant; source coordinates and Fermat/time-delay
    differences scale with lambda, and signed magnifications with lambda^-2.
    """

    kappa_ext = float(external_convergence)
    if not math.isfinite(kappa_ext):
        raise ValueError("external convergence must be finite")
    lambda_mst = 1.0 - kappa_ext
    if lambda_mst <= 0.0:
        raise ValueError("mass-sheet lambda must be positive")
    beta = tuple(lambda_mst * float(value) for value in source_position)
    if len(beta) != 2 or not all(math.isfinite(value) for value in beta):
        raise ValueError("source position must contain two finite values")
    images = tuple(
        PhysicalImage(
            image_id=image.image_id,
            position_arcsec=image.position_arcsec,
            mu_signed=image.mu_signed / lambda_mst**2,
            parity=image.parity,
            morse_class=image.morse_class,
            fermat_potential_dimensionless=(
                None
                if image.fermat_potential_dimensionless is None
                else lambda_mst * image.fermat_potential_dimensionless
            ),
            arrival_time_seconds=(
                None
                if image.arrival_time_seconds is None
                else lambda_mst * image.arrival_time_seconds
            ),
            valid=image.valid,
            validity_reason=image.validity_reason,
        )
        for image in solution.physical_images
    )
    return (
        LensSystemSolution(
            lens_family=solution.lens_family,
            physical_images=images,
            solver_name=f"{solution.solver_name}+mass-sheet-transform",
            solver_version=f"{solution.solver_version}+mst-v1",
            valid=solution.valid,
            validity_reason=solution.validity_reason,
        ),
        (beta[0], beta[1]),
    )
