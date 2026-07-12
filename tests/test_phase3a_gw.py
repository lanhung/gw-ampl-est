import numpy as np

from gwlens_mm.physics.quantities import ImageParity, MorseClass
from gwlens_mm.physics.solver import PhysicalImage
from gwlens_mm.production.gw import ImageProjection, select_earliest_detectable_images


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
