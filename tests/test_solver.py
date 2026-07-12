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
    validate_solver_contract,
)


def test_sis_solver_order_and_parity():
    solution = SISSolver().solve((0.5, 0.0), {"einstein_radius_arcsec": 1.0})
    plus, minus = solution.physical_images
    assert plus.arrival_time_dimensionless < minus.arrival_time_dimensionless
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
            arrival_time_dimensionless=float(index),
            parity=ImageParity.POSITIVE if index % 2 == 0 else ImageParity.NEGATIVE,
            morse_class=MorseClass.MINIMUM if index % 2 == 0 else MorseClass.SADDLE,
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
                        "minimum",
                        (1.0, 0.0),
                        2.0,
                        0.0,
                        ImageParity.POSITIVE,
                        MorseClass.MINIMUM,
                    ),
                    PhysicalImage(
                        "saddle",
                        (-0.7, 0.1),
                        -1.0,
                        1.0,
                        ImageParity.NEGATIVE,
                        MorseClass.SADDLE,
                    ),
                ),
                solver_name="fixture-sie-adapter",
                solver_version="contract-only",
            )

    validate_solver_contract(FixtureSIEAdapter(), [((0.1, 0.0), {"shear": 0.05})])
