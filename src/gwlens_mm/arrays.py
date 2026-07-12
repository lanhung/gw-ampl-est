"""Small in-memory validators for v2 strain-array contracts."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_strain_array_semantics(
    noisy: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    detector_availability_mask: Sequence[Sequence[bool]],
) -> None:
    """Validate decomposition and zero-fill semantics without writing data."""

    arrays = tuple(np.asarray(array) for array in (noisy, clean, noise))
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("noisy, clean, and noise arrays must have identical shapes")
    shape = arrays[0].shape
    if len(shape) != 3 or shape[:2] != (2, 3) or shape[2] <= 0:
        raise ValueError("strain products must have shape (2 images, 3 detectors, samples)")
    if any(array.dtype != np.dtype("float32") for array in arrays):
        raise ValueError("smoke strain products must use float32")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("strain products must be finite; NaN is not a missing-slot marker")
    mask = np.asarray(detector_availability_mask, dtype=bool)
    if mask.shape != (2, 3):
        raise ValueError("detector availability mask must have shape (2, 3)")
    unavailable = ~mask
    for array in arrays:
        if np.any(array[unavailable] != 0.0):
            raise ValueError("unavailable detector slots must be exactly zero-filled")
    available = mask
    expected_noisy = (arrays[1] + arrays[2]).astype(np.float32)
    if not np.array_equal(arrays[0][available], expected_noisy[available]):
        raise ValueError("available detector slots must satisfy noisy = clean + noise")
