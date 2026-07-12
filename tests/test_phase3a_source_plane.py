import math

import numpy as np
import pytest

from gwlens_mm.production.source_plane import (
    boundary_points,
    normalized_source_log_density,
    sample_source_position,
    scaled_boundary_points,
)


def test_source_plane_density_integrates_to_one() -> None:
    theta_e = 1.7
    density = math.exp(normalized_source_log_density(theta_e))
    assert density * 25.0 * theta_e**2 == pytest.approx(1.0)


def test_source_sampling_is_deterministic_and_inside_half_open_support() -> None:
    first = sample_source_position(np.random.default_rng(7), 2.0)
    second = sample_source_position(np.random.default_rng(7), 2.0)
    assert first == second
    assert all(-5.0 <= value < 5.0 for value in first)


def test_boundary_points_are_unique_and_scale_with_einstein_radius() -> None:
    points = boundary_points(32)
    assert len(points) == len(set(points)) == 124
    assert all(-2.5 <= coordinate < 2.5 for point in points for coordinate in point)
    scaled = scaled_boundary_points(points, 0.3)
    assert scaled[0][0] == pytest.approx(points[0][0] * 0.3)
