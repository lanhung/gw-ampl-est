import pytest

from gwlens_mm.physics.quantities import (
    ImageParity,
    LensFamily,
    MorseClass,
    PrimaryDefinition,
)
from gwlens_mm.physics.solver import (
    LensSystemSolution,
    PhysicalImage,
    SelectedPair,
    SISSolver,
    apply_mass_sheet_transform,
    validate_solver_contract,
)


def test_sis_solver_order_and_parity():
    solution = SISSolver().solve((0.5, 0.0), {"einstein_radius_arcsec": 1.0})
    plus, minus = solution.physical_images
    assert plus.fermat_potential_dimensionless < minus.fermat_potential_dimensionless
    assert plus.arrival_time_seconds is None
    assert minus.arrival_time_seconds is None
    assert plus.parity is ImageParity.POSITIVE
    assert plus.morse_class is MorseClass.MINIMUM
    assert minus.parity is ImageParity.NEGATIVE
    assert minus.morse_class is MorseClass.SADDLE
    validate_solver_contract(SISSolver(), [((0.5, 0.0), {"einstein_radius_arcsec": 1.0})])


def test_multi_image_system_is_separate_from_selected_pair():
    images = tuple(
        PhysicalImage(
            image_id=f"image_{index}",
            position_arcsec=(float(index), 0.0),
            mu_signed=2.0 if index % 2 == 0 else -1.0,
            parity=ImageParity.POSITIVE if index % 2 == 0 else ImageParity.NEGATIVE,
            morse_class=MorseClass.MINIMUM if index % 2 == 0 else MorseClass.SADDLE,
            fermat_potential_dimensionless=float(index),
        )
        for index in range(4)
    )
    solution = LensSystemSolution(
        lens_family=LensFamily.SIE_EXTERNAL_SHEAR,
        physical_images=images,
        solver_name="contract-fixture",
        solver_version="1",
    )
    pair = SelectedPair(
        primary_image_id="image_0",
        secondary_image_id="image_2",
        primary_definition=PrimaryDefinition.CATALOG_ANCHOR,
        selection_reason="fixture detector visibility",
        detector_visibility={"H1": (True, True)},
        unselected_image_ids=("image_1",),
        censored_image_ids=("image_3",),
    )
    pair.validate_against(solution)
    assert len(solution.physical_images) == 4


def test_selected_pair_rejects_unknown_image():
    solution = SISSolver().solve((0.5, 0.0), {})
    pair = SelectedPair(
        primary_image_id="sis_plus",
        secondary_image_id="not_an_image",
        primary_definition=PrimaryDefinition.EARLIEST_ARRIVING,
        selection_reason="bad fixture",
        detector_visibility={},
    )
    with pytest.raises(ValueError, match="unknown physical image"):
        pair.validate_against(solution)


def test_lightweight_non_sis_adapter_contract():
    class FixtureSIEAdapter:
        lens_family = LensFamily.SIE_EXTERNAL_SHEAR

        def solve(self, source_position, parameters):
            del source_position, parameters
            return LensSystemSolution(
                lens_family=self.lens_family,
                physical_images=(
                    PhysicalImage(
                        image_id="minimum",
                        position_arcsec=(1.0, 0.0),
                        mu_signed=2.0,
                        parity=ImageParity.POSITIVE,
                        morse_class=MorseClass.MINIMUM,
                        arrival_time_seconds=0.0,
                    ),
                    PhysicalImage(
                        image_id="saddle",
                        position_arcsec=(-0.7, 0.1),
                        mu_signed=-1.0,
                        parity=ImageParity.NEGATIVE,
                        morse_class=MorseClass.SADDLE,
                        arrival_time_seconds=1.0,
                    ),
                ),
                solver_name="fixture-sie-adapter",
                solver_version="contract-only",
            )

    validate_solver_contract(FixtureSIEAdapter(), [((0.1, 0.0), {"shear": 0.05})])


def test_physical_image_requires_an_explicit_time_coordinate():
    with pytest.raises(ValueError, match="Fermat or physical-time"):
        PhysicalImage(
            image_id="missing-time",
            position_arcsec=(0.0, 0.0),
            mu_signed=1.0,
            parity=ImageParity.POSITIVE,
            morse_class=MorseClass.MINIMUM,
        )


def test_mass_sheet_transform_scales_source_magnification_and_fermat_difference():
    original = SISSolver().solve((0.5, 0.0), {"einstein_radius_arcsec": 1.0})
    transformed, source = apply_mass_sheet_transform(original, (0.5, 0.0), 0.2)
    assert source == pytest.approx((0.4, 0.0))
    for before, after in zip(original.physical_images, transformed.physical_images):
        assert after.position_arcsec == before.position_arcsec
        assert after.mu_signed == pytest.approx(before.mu_signed / 0.8**2)
        assert after.parity is before.parity
        assert after.morse_class is before.morse_class
    before_values = [image.fermat_potential_dimensionless for image in original.physical_images]
    after_values = [
        image.fermat_potential_dimensionless for image in transformed.physical_images
    ]
    before_delta = float(before_values[1]) - float(before_values[0])
    after_delta = float(after_values[1]) - float(after_values[0])
    assert after_delta == pytest.approx(0.8 * before_delta)


def test_mass_sheet_transform_scales_physical_delay():
    solution = LensSystemSolution(
        lens_family=LensFamily.SIE_EXTERNAL_SHEAR,
        physical_images=(
            PhysicalImage(
                image_id="minimum",
                position_arcsec=(1.0, 0.0),
                mu_signed=2.0,
                parity=ImageParity.POSITIVE,
                morse_class=MorseClass.MINIMUM,
                arrival_time_seconds=0.0,
            ),
            PhysicalImage(
                image_id="saddle",
                position_arcsec=(-1.0, 0.0),
                mu_signed=-1.0,
                parity=ImageParity.NEGATIVE,
                morse_class=MorseClass.SADDLE,
                arrival_time_seconds=100.0,
            ),
        ),
        solver_name="fixture",
        solver_version="1",
    )
    transformed, _ = apply_mass_sheet_transform(solution, (0.1, 0.0), -0.1)
    assert transformed.physical_images[1].arrival_time_seconds == pytest.approx(110.0)


def test_mass_sheet_transform_rejects_nonpositive_lambda():
    solution = SISSolver().solve((0.5, 0.0), {"einstein_radius_arcsec": 1.0})
    with pytest.raises(ValueError, match="lambda must be positive"):
        apply_mass_sheet_transform(solution, (0.5, 0.0), 1.0)
