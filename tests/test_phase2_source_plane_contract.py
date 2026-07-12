import numpy as np
import pytest

from gwlens_mm.physics.lenstronomy_adapter import LenstronomyAdapter, _load_dependencies
from gwlens_mm.physics.quantities import LensFamily

pytestmark = pytest.mark.optional_solver


def _deduplicate(model, kwargs_lens, candidates, tolerance=1.0e-7):
    kept = []
    for x, y in candidates:
        beta_x, beta_y = model.ray_shooting(x, y, kwargs_lens)
        assert np.hypot(beta_x, beta_y - 0.1) <= 1.0e-8
        if not any(np.hypot(x - old_x, y - old_y) <= tolerance for old_x, old_y in kept):
            kept.append((float(x), float(y)))
    return sorted(kept)


def _union(solver, model, kwargs_lens, grid_distance, grid_window, nmeas, extra):
    analytical = solver.image_position_from_source(
        0.0,
        0.1,
        kwargs_lens,
        solver="analytical",
        Nmeas=nmeas,
        Nmeas_extra=extra,
        arrival_time_sort=True,
    )
    grid = solver.image_position_from_source(
        0.0,
        0.1,
        kwargs_lens,
        solver="lenstronomy",
        min_distance=grid_distance,
        search_window=grid_window,
        precision_limit=1.0e-10,
        num_iter_max=200,
        arrival_time_sort=True,
        initial_guess_cut=False,
        num_random=0,
    )
    candidates = list(zip(*analytical)) + list(zip(*grid))
    return _deduplicate(model, kwargs_lens, candidates)


def test_rc3_union_retains_shallow_epl_central_image_and_matches_reference():
    pytest.importorskip("lenstronomy")
    parameters = {
        "einstein_radius_arcsec": 1.0,
        "axis_ratio": 0.4,
        "position_angle_rad": 0.0,
        "shear_gamma1": 0.15,
        "shear_gamma2": 0.0,
        "density_slope": 1.6,
    }
    adapter = LenstronomyAdapter(LensFamily.EPL_EXTERNAL_SHEAR, 0.5, 1.5)
    model, kwargs_lens = adapter._model(parameters)
    _, solver_class, *_ = _load_dependencies()
    solver = solver_class(model)
    primary = _union(solver, model, kwargs_lens, 0.02, 8.0, 400, 80)
    reference = _union(solver, model, kwargs_lens, 0.005, 12.0, 800, 160)
    assert len(primary) == len(reference) == 5
    unmatched = list(reference)
    for x_primary, y_primary in primary:
        distances = [
            np.hypot(x_primary - x_reference, y_primary - y_reference)
            for x_reference, y_reference in unmatched
        ]
        match = int(np.argmin(distances))
        assert distances[match] <= 1.0e-6
        unmatched.pop(match)
    assert not unmatched
    magnifications = np.abs(
        model.magnification(
            np.array([position[0] for position in primary]),
            np.array([position[1] for position in primary]),
            kwargs_lens,
        )
    )
    assert np.min(magnifications) < 0.01
