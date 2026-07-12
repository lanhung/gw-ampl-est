"""Numerically stable singular-isothermal-sphere analytic control."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .quantities import Magnification, RelativeMagnification


@dataclass(frozen=True)
class SISResult:
    source_position_y: float
    plus: Magnification
    minus: Magnification
    relative: RelativeMagnification


def _validate_y(source_position_y: float) -> float:
    y = float(source_position_y)
    if not math.isfinite(y) or not 0.0 < y < 1.0:
        raise ValueError("SIS double-image source position must satisfy 0 < y < 1")
    return y


def sis_from_source_position(source_position_y: float) -> SISResult:
    """Return the SIS double-image solution for dimensionless source radius y."""

    y = _validate_y(source_position_y)
    plus = Magnification(1.0 + 1.0 / y)
    minus = Magnification(1.0 - 1.0 / y)
    relative = RelativeMagnification.from_magnifications(primary=plus, secondary=minus)
    return SISResult(y, plus, minus, relative)


def sis_from_relative_flux(relative_flux_magnification: float) -> SISResult:
    """Invert the SIS secondary/primary absolute-flux ratio without clipping."""

    relative = RelativeMagnification(float(relative_flux_magnification))
    ratio = relative.relative_flux_magnification
    if ratio >= 1.0:
        raise ValueError("SIS plus-primary inversion requires relative flux strictly below 1")
    denominator = 1.0 - ratio
    plus = Magnification(2.0 / denominator)
    minus = Magnification(-2.0 * ratio / denominator)
    y = denominator / (1.0 + ratio)
    return SISResult(y, plus, minus, relative)
