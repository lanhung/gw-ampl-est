"""Bilby-compatible PSD whitening without per-event standardization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ASDCurve:
    """A verified amplitude-spectral-density curve."""

    frequency_hz: np.ndarray
    asd: np.ndarray
    identity: str
    sha256: Optional[str] = None

    def validate(self) -> None:
        frequency = np.asarray(self.frequency_hz, dtype=np.float64)
        asd = np.asarray(self.asd, dtype=np.float64)
        if frequency.ndim != 1 or asd.ndim != 1 or frequency.shape != asd.shape:
            raise ValueError("ASD frequency and value arrays must be matching vectors")
        if len(frequency) < 2 or not np.all(np.isfinite(frequency)):
            raise ValueError("ASD frequency grid is incomplete or nonfinite")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("ASD frequency grid must be strictly increasing")
        if not np.all(np.isfinite(asd)) or not np.all(asd > 0):
            raise ValueError("ASD values must be finite and positive")
        if not self.identity.strip():
            raise ValueError("ASD identity is required")
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError("ASD SHA-256 must be hexadecimal text")

    @classmethod
    def from_file(cls, path: Path, *, expected_sha256: Optional[str] = None) -> "ASDCurve":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"ASD SHA-256 mismatch: {path}")
        table = np.loadtxt(path, comments=("#", "%"), dtype=np.float64)
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError("ASD table must contain frequency and ASD columns")
        curve = cls(table[:, 0], table[:, 1], path.name, digest)
        curve.validate()
        return curve

    def interpolate(self, frequency_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate inside declared support and mark supported bins."""

        self.validate()
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        supported = (frequencies >= self.frequency_hz[0]) & (
            frequencies <= self.frequency_hz[-1]
        )
        values = np.full(frequencies.shape, np.inf, dtype=np.float64)
        values[supported] = np.interp(
            frequencies[supported], self.frequency_hz, self.asd
        )
        return values, supported


def bilby_psd_whiten(
    time_series: np.ndarray,
    *,
    sampling_frequency_hz: float,
    asd_curve: ASDCurve,
    minimum_frequency_hz: float,
    maximum_frequency_hz: Optional[float] = None,
) -> np.ndarray:
    """Match Bilby's ``nfft`` and time-domain whitening normalization.

    The input may have arbitrary leading axes and a final time axis. No observed
    per-event standard deviation is used. Bins outside the declared PSD/frequency
    support are zeroed exactly.
    """

    series = np.asarray(time_series)
    if series.ndim < 1 or series.shape[-1] < 2:
        raise ValueError("time series requires a nontrivial final sample axis")
    if not np.all(np.isfinite(series)):
        raise ValueError("time series contains NaN or Inf")
    sampling_frequency = float(sampling_frequency_hz)
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling frequency must be finite and positive")
    sample_count = series.shape[-1]
    duration = sample_count / sampling_frequency
    nyquist = sampling_frequency / 2.0
    upper = nyquist if maximum_frequency_hz is None else float(maximum_frequency_hz)
    lower = float(minimum_frequency_hz)
    if lower < 0 or upper <= lower or upper > nyquist:
        raise ValueError("whitening frequency interval is invalid")
    frequency = np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency)
    asd, supported = asd_curve.interpolate(frequency)
    mask = supported & (frequency >= lower) & (frequency <= upper)
    retained = int(np.count_nonzero(mask))
    if retained == 0:
        raise ValueError("whitening retained no frequency bins")
    # bilby.core.utils.nfft: rfft(time_series) / sampling_frequency.
    frequency_series = np.fft.rfft(series, axis=-1) / sampling_frequency
    whitened_frequency = np.zeros_like(frequency_series, dtype=np.complex128)
    whitened_frequency[..., mask] = frequency_series[..., mask] / (
        asd[mask] * np.sqrt(duration / 4.0)
    )
    # bilby.gw.detector.Interferometer.whitened_time_domain_strain.
    frequency_window_factor = retained / len(frequency)
    whitened = np.fft.irfft(whitened_frequency, n=sample_count, axis=-1)
    whitened *= np.sqrt(retained) / frequency_window_factor
    if not np.all(np.isfinite(whitened)):
        raise ValueError("whitening produced NaN or Inf")
    return whitened.astype(np.float32, copy=False)


def whiten_detector_grid(
    strain: np.ndarray,
    detector_mask: np.ndarray,
    *,
    sampling_frequency_hz: float,
    detector_curves: Tuple[ASDCurve, ASDCurve, ASDCurve],
    minimum_frequency_hz: float,
    maximum_frequency_hz: Optional[float] = None,
) -> np.ndarray:
    """Whiten a fixed (2 images, 3 detectors, samples) tensor."""

    values = np.asarray(strain)
    mask = np.asarray(detector_mask, dtype=bool)
    if values.ndim != 3 or values.shape[:2] != (2, 3):
        raise ValueError("strain must have shape (2, 3, samples)")
    if mask.shape != (2, 3):
        raise ValueError("detector mask must have shape (2, 3)")
    output = np.zeros(values.shape, dtype=np.float32)
    for detector_index, curve in enumerate(detector_curves):
        available_images = mask[:, detector_index]
        if np.any(available_images):
            output[available_images, detector_index] = bilby_psd_whiten(
                values[available_images, detector_index],
                sampling_frequency_hz=sampling_frequency_hz,
                asd_curve=curve,
                minimum_frequency_hz=minimum_frequency_hz,
                maximum_frequency_hz=maximum_frequency_hz,
            )
    if np.any(output[~mask] != 0.0):
        raise RuntimeError("unavailable detector slots must remain zero")
    return output
