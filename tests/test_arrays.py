import numpy as np
import pytest

from gwlens_mm.arrays import validate_strain_array_semantics


def valid_products():
    mask = np.array([[True, True, False], [True, False, True]])
    clean = np.zeros((2, 3, 8), dtype=np.float32)
    noise = np.zeros_like(clean)
    clean[mask] = 2.0
    noise[mask] = 0.5
    noisy = clean + noise
    return noisy, clean, noise, mask


def test_available_decomposition_and_unavailable_zero_fill_pass():
    validate_strain_array_semantics(*valid_products())


def test_unavailable_slot_cannot_contain_latent_clean_signal():
    noisy, clean, noise, mask = valid_products()
    clean[0, 2, 0] = 1.0
    with pytest.raises(ValueError, match="zero-filled"):
        validate_strain_array_semantics(noisy, clean, noise, mask)


def test_available_slot_requires_noisy_clean_noise_identity():
    noisy, clean, noise, mask = valid_products()
    noisy[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match=r"noisy = clean \+ noise"):
        validate_strain_array_semantics(noisy, clean, noise, mask)


def test_nan_is_not_a_missing_detector_representation():
    noisy, clean, noise, mask = valid_products()
    noisy[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_strain_array_semantics(noisy, clean, noise, mask)


def test_smoke_products_require_float32():
    noisy, clean, noise, mask = valid_products()
    with pytest.raises(ValueError, match="float32"):
        validate_strain_array_semantics(noisy.astype(np.float64), clean, noise, mask)
