import math

import pytest

from gwlens_mm.physics.quantities import (
    Magnification,
    RelativeMagnification,
    absolute_time_delay,
    apparent_luminosity_distance,
    signed_time_delay,
)
from gwlens_mm.physics.sis import sis_from_relative_flux, sis_from_source_position


@pytest.mark.parametrize("y", [1e-8, 1e-5, 0.1, 0.5, 0.99999999])
def test_sis_round_trip(y):
    direct = sis_from_source_position(y)
    inverse = sis_from_relative_flux(direct.relative.relative_flux_magnification)
    assert inverse.source_position_y == pytest.approx(y, rel=2e-9, abs=1e-12)
    assert inverse.plus.mu_abs == pytest.approx(direct.plus.mu_abs)
    assert inverse.minus.mu_signed == pytest.approx(direct.minus.mu_signed)
    assert math.isfinite(direct.plus.log_mu_abs)
    assert math.isfinite(direct.relative.logit_relative_flux)


def test_sis_known_legacy_fixture():
    result = sis_from_source_position(0.5)
    assert result.plus.mu_signed == pytest.approx(3.0)
    assert result.minus.mu_signed == pytest.approx(-1.0)
    assert result.relative.relative_flux_magnification == pytest.approx(1.0 / 3.0)
    assert result.relative.relative_strain_amplitude == pytest.approx(math.sqrt(1.0 / 3.0))


@pytest.mark.parametrize("invalid", [-1.0, 0.0, 1.0, 2.0, math.inf, math.nan])
def test_sis_rejects_invalid_y_without_clipping(invalid):
    with pytest.raises(ValueError):
        sis_from_source_position(invalid)


@pytest.mark.parametrize("invalid", [-1.0, 0.0, 1.0, 2.0, math.inf, math.nan])
def test_sis_rejects_invalid_relative_flux_without_clipping(invalid):
    with pytest.raises(ValueError):
        sis_from_relative_flux(invalid)


def test_magnification_distance_and_delay_conventions():
    primary = Magnification(4.0)
    secondary = Magnification(-1.0)
    relative = RelativeMagnification.from_magnifications(primary=primary, secondary=secondary)
    assert primary.amplitude_factor == 2.0
    assert relative.relative_flux_magnification == 0.25
    assert relative.relative_strain_amplitude == 0.5
    assert apparent_luminosity_distance(1000.0, primary) == 500.0
    assert signed_time_delay(15.0, 10.0) == 5.0
    assert absolute_time_delay(10.0, 15.0) == 5.0


def test_general_pair_ratio_does_not_assume_primary_is_brightest():
    catalog_anchor = Magnification(1.0)
    secondary = Magnification(-4.0)
    relative = RelativeMagnification.from_magnifications(
        primary=catalog_anchor, secondary=secondary
    )
    assert relative.relative_flux_magnification == 4.0
    assert relative.relative_strain_amplitude == 2.0
    with pytest.raises(ValueError, match="logit"):
        _ = relative.logit_relative_flux
