import numpy as np

from gwlens_mm.production.dynamics import sample_kinematics_nuisance


def test_kinematics_nuisance_is_deterministic_and_supported() -> None:
    first = sample_kinematics_nuisance(np.random.default_rng(4))
    second = sample_kinematics_nuisance(np.random.default_rng(4))
    assert first == second
    effective_radius, anisotropy_ratio = first
    assert 0.3 <= effective_radius <= 2.0
    assert 0.5 <= anisotropy_ratio <= 5.0
