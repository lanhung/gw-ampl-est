"""Fourier-domain time-delay and geometric-optics Morse factors.

Convention: H(f) = integral h(t) exp(-2*pi*i*f*t) dt. A delayed signal
h(t-delay) therefore has H_delayed(f) = exp(-2*pi*i*f*delay) H(f).
For positive frequencies the Morse factor is exp(-i*pi*n), with
n in {0, 1/2, 1}. Negative frequencies use the complex conjugate factor so a
conjugate-symmetric spectrum remains a real time-domain signal.
"""

from __future__ import annotations

import numpy as np

from .quantities import MorseClass


def morse_phase_factor(frequencies_hz: np.ndarray, morse: MorseClass) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    positive_factor = np.exp(-1j * np.pi * morse.half_integer_index)
    factors = np.ones(frequencies.shape, dtype=complex)
    factors[frequencies > 0] = positive_factor
    factors[frequencies < 0] = np.conjugate(positive_factor)
    if morse is MorseClass.MAXIMUM:
        factors[frequencies == 0] = -1.0
    return factors


def time_shift_factor(frequencies_hz: np.ndarray, delay_seconds: float) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if not np.isfinite(delay_seconds):
        raise ValueError("delay_seconds must be finite")
    return np.exp(-2j * np.pi * frequencies * float(delay_seconds))


def apply_geometric_optics_image(
    spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    amplitude_factor: float,
    delay_seconds: float,
    morse: MorseClass,
) -> np.ndarray:
    if not np.isfinite(amplitude_factor) or amplitude_factor <= 0:
        raise ValueError("amplitude_factor must be positive and finite")
    spectrum_array = np.asarray(spectrum, dtype=complex)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if spectrum_array.shape != frequencies.shape:
        raise ValueError("spectrum and frequencies must have identical shapes")
    return (
        spectrum_array
        * amplitude_factor
        * time_shift_factor(frequencies, delay_seconds)
        * morse_phase_factor(frequencies, morse)
    )
