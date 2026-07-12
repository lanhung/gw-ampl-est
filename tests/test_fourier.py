import numpy as np
import pytest

from gwlens_mm.physics.fourier import (
    apply_geometric_optics_image,
    morse_phase_factor,
    time_shift_factor,
)
from gwlens_mm.physics.quantities import MorseClass


def test_morse_positive_frequency_factors():
    frequencies = np.array([1.0])
    assert morse_phase_factor(frequencies, MorseClass.MINIMUM)[0] == pytest.approx(1.0)
    assert morse_phase_factor(frequencies, MorseClass.SADDLE)[0] == pytest.approx(-1j)
    assert morse_phase_factor(frequencies, MorseClass.MAXIMUM)[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("morse", list(MorseClass))
def test_morse_preserves_conjugate_symmetry(morse):
    frequencies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    spectrum = np.array([2 - 1j, 1 - 2j, 3 + 0j, 1 + 2j, 2 + 1j])
    transformed = apply_geometric_optics_image(
        spectrum,
        frequencies,
        amplitude_factor=2.0,
        delay_seconds=0.125,
        morse=morse,
    )
    assert transformed[0] == pytest.approx(np.conjugate(transformed[-1]))
    assert transformed[1] == pytest.approx(np.conjugate(transformed[-2]))
    assert transformed[2].imag == pytest.approx(0.0)


def test_time_shift_direction_for_fft_convention():
    sample_rate = 64
    sample_count = 64
    time = np.arange(sample_count) / sample_rate
    signal = np.zeros(sample_count)
    signal[5] = 1.0
    frequencies = np.fft.fftfreq(sample_count, d=1 / sample_rate)
    shifted = np.fft.ifft(np.fft.fft(signal) * time_shift_factor(frequencies, 3 / sample_rate)).real
    assert int(np.argmax(shifted)) == 8
    assert time[8] - time[5] == pytest.approx(3 / sample_rate)
