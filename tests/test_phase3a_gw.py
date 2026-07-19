import numpy as np

from gwlens_mm.physics.quantities import ImageParity, MorseClass
from gwlens_mm.physics.solver import PhysicalImage
from gwlens_mm.production.gw import (
    ImageProjection,
    WaveformNumericalPathology,
    detector_frame_newtonian_chirp_time_seconds,
    raised_cosine_guard_window,
    select_earliest_detectable_images,
    validate_source_polarization_spectrum,
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


def test_source_polarization_spectral_validity_accepts_smooth_support() -> None:
    frequencies = np.arange(0.0, 64.0, 0.25)
    envelope = np.exp(-0.025 * np.maximum(frequencies - 20.0, 0.0))
    envelope[frequencies < 20.0] = 0.0
    result = validate_source_polarization_spectrum(
        {"plus": envelope.astype(complex), "cross": (0.7j * envelope)},
        frequencies,
        minimum_frequency_hz=20.0,
        positive_amplitude_quantile=0.999,
        maximum_peak_to_quantile_ratio=10.0,
    )
    assert result.maximum_peak_to_quantile_ratio < 2.0
    assert set(result.peak_to_quantile_ratio_by_polarization) == {"plus", "cross"}


def test_source_polarization_spectral_validity_rejects_isolated_bin() -> None:
    frequencies = np.arange(0.0, 2048.0, 0.03125)
    envelope = np.ones(frequencies.shape, dtype=complex) * 1.0e-24
    envelope[frequencies < 20.0] = 0.0
    envelope[1211] = 1.0e-18
    with np.testing.assert_raises(WaveformNumericalPathology):
        validate_source_polarization_spectrum(
            {"plus": envelope, "cross": 0.5j * envelope},
            frequencies,
            minimum_frequency_hz=20.0,
            positive_amplitude_quantile=0.999,
            maximum_peak_to_quantile_ratio=10.0,
        )


def test_source_polarization_spectral_validity_ignores_zero_padding() -> None:
    frequencies = np.arange(0.0, 128.0, 0.25)
    support = np.zeros(frequencies.shape, dtype=complex)
    active = (frequencies >= 20.0) & (frequencies < 40.0)
    support[active] = np.linspace(1.0, 2.0, int(np.sum(active))) * 1.0e-24
    result = validate_source_polarization_spectrum(
        {"plus": support, "cross": 0.5j * support},
        frequencies,
        minimum_frequency_hz=20.0,
        positive_amplitude_quantile=0.999,
        maximum_peak_to_quantile_ratio=10.0,
    )
    assert result.positive_in_band_bin_count_by_polarization == {
        "plus": int(np.sum(active)),
        "cross": int(np.sum(active)),
    }
