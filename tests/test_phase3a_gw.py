import numpy as np

from gwlens_mm.physics.quantities import ImageParity, MorseClass
from gwlens_mm.physics.solver import PhysicalImage
from gwlens_mm.production.gw import (
    ImageProjection,
    detector_frame_newtonian_chirp_time_seconds,
    raised_cosine_guard_window,
    select_earliest_detectable_images,
)


def projection(identifier: str, arrival: float, snrs: tuple[float, float, float]):
    image = PhysicalImage(
        identifier,
        (0.0, 0.0),
        2.0,
        ImageParity.POSITIVE,
        MorseClass.MINIMUM,
        arrival_time_seconds=arrival,
    )
    return ImageProjection(image, arrival, arrival - 6.0, np.zeros((3, 8)), snrs)


def test_selection_uses_earliest_two_images_passing_both_thresholds() -> None:
    projections = (
        projection("first", 0.0, (8.0, 7.0, 1.0)),
        projection("fails", 1.0, (11.0, 1.0, 1.0)),
        projection("third", 2.0, (6.0, 6.0, 6.0)),
    )
    result = select_earliest_detectable_images(projections)
    assert result.selected is not None
    assert tuple(item.image.image_id for item in result.selected) == ("first", "third")
    assert result.passing_image_ids == ("first", "third")


def test_selection_rejects_when_fewer_than_two_images_pass() -> None:
    result = select_earliest_detectable_images(
        (projection("only", 0.0, (8.0, 7.0, 1.0)),)
    )
    assert result.selected is None
    assert result.rejection_reason == "fewer_than_two_images_pass_synthetic_selection"


def test_rc5_conditioning_window_has_exact_guards_and_symmetric_transition() -> None:
    window = raised_cosine_guard_window(16384, 512, 512)
    assert window.shape == (16384,)
    assert np.all(window[:512] == 0.0)
    assert np.all(window[-512:] == 0.0)
    assert np.all(window[1024:-1024] == 1.0)
    assert np.all(np.diff(window[512:1024]) > 0.0)
    assert np.array_equal(window[512:1024], window[-1024:-512][::-1])


def test_rc5_conditioning_window_rejects_invalid_geometry() -> None:
    for values in ((0, 1, 1), (10, 0, 1), (10, 1, 0), (8, 2, 2)):
        with np.testing.assert_raises(ValueError):
            raised_cosine_guard_window(*values)


def test_detector_frame_chirp_time_is_positive_and_decreases_with_mass() -> None:
    low_mass = detector_frame_newtonian_chirp_time_seconds(30.0, 15.0, 20.0)
    high_mass = detector_frame_newtonian_chirp_time_seconds(320.0, 320.0, 20.0)
    assert low_mass > high_mass > 0.0
