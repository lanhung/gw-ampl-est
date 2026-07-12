import math

import numpy as np

from gwlens_mm.production.population import sample_population


def test_population_draws_obey_frozen_support_and_are_deterministic() -> None:
    first = sample_population(np.random.default_rng(20260712))
    second = sample_population(np.random.default_rng(20260712))
    assert first == second
    assert 20.0 <= first.source_parameters["mass_1_source"] <= 80.0
    assert first.source_parameters["mass_2_source"] >= 10.0
    assert 0.3 <= first.lens_parameters["einstein_radius_arcsec"] <= 3.0
    assert first.z_source - first.z_lens >= 0.1
    assert math.isfinite(first.proposal_log_probability)
    assert math.isfinite(first.evaluation_log_probability)
    assert math.isfinite(first.importance_weight) and first.importance_weight > 0


def test_population_sample_never_violates_conditional_secondary_mass() -> None:
    rng = np.random.default_rng(9)
    for _ in range(1000):
        draw = sample_population(rng)
        assert draw.source_parameters["mass_2_source"] >= 10.0
