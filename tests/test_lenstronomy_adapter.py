import pytest

from gwlens_mm.physics.lenstronomy_adapter import LenstronomyAdapter
from gwlens_mm.physics.quantities import LensFamily, PrimaryDefinition
from gwlens_mm.physics.solver import SelectedPair, validate_solver_contract

lenstronomy = pytest.importorskip("lenstronomy")


def parameters(family):
    values = {
        "einstein_radius_arcsec": 1.1,
        "axis_ratio": 0.72,
        "position_angle_rad": 0.2,
        "shear_gamma1": 0.04,
        "shear_gamma2": -0.02,
    }
    if family is LensFamily.EPL_EXTERNAL_SHEAR:
        values["density_slope"] = 2.1
        values["axis_ratio"] = 0.75
    return values


@pytest.mark.optional_solver
@pytest.mark.parametrize(
    ("family", "source_position", "expected_images"),
    [
        (LensFamily.SIE_EXTERNAL_SHEAR, (0.5, 0.1), 2),
        (LensFamily.SIE_EXTERNAL_SHEAR, (0.03, 0.02), 4),
        (LensFamily.EPL_EXTERNAL_SHEAR, (0.03, 0.02), 4),
    ],
)
def test_lenstronomy_deterministic_fixtures(family, source_position, expected_images):
    adapter = LenstronomyAdapter(family, z_lens=0.5, z_source=1.5)
    solution = adapter.solve(source_position, parameters(family))
    assert solution.valid
    assert len(solution.physical_images) == expected_images
    assert [image.image_id for image in solution.physical_images] == [
        f"image_{index}" for index in range(expected_images)
    ]
    arrivals = [image.arrival_time_seconds for image in solution.physical_images]
    assert arrivals == sorted(arrivals)
    assert arrivals[0] == pytest.approx(0.0, abs=1e-9)
    assert all(
        image.fermat_potential_dimensionless is not None
        for image in solution.physical_images
    )
    validate_solver_contract(adapter, [(source_position, parameters(family))])


@pytest.mark.optional_solver
def test_selected_pair_need_not_be_first_two_solver_outputs():
    adapter = LenstronomyAdapter(LensFamily.SIE_EXTERNAL_SHEAR, 0.5, 1.5)
    solution = adapter.solve((0.03, 0.02), parameters(adapter.lens_family))
    pair = SelectedPair(
        primary_image_id="image_0",
        secondary_image_id="image_2",
        primary_definition=PrimaryDefinition.EARLIEST_ARRIVING,
        selection_reason="engineering fixture uses a non-consecutive pair",
        detector_visibility={"H1": (True, True)},
        unselected_image_ids=("image_1", "image_3"),
    )
    pair.validate_against(solution)


@pytest.mark.optional_solver
def test_all_fixture_first_two_diagnostics():
    from scripts.phase1b.run_solver_fixtures import run

    result = run()
    diagnostics = {
        item["fixture"]: item["selected_pair_is_first_two"]
        for item in result["fixtures"]
    }
    assert diagnostics == {
        "sis_double": True,
        "sie_double": True,
        "sie_quad_nonconsecutive": False,
        "epl_quad": True,
    }
