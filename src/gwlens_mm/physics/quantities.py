"""Unambiguous names and conversions for lensing observables.

The selected-pair convention is always explicit. No function in this module
infers that the earliest, brightest, minimum, or catalog-anchor image is the
same object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ImageParity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class MorseClass(str, Enum):
    MINIMUM = "minimum"
    SADDLE = "saddle"
    MAXIMUM = "maximum"

    @property
    def half_integer_index(self) -> float:
        return {
            MorseClass.MINIMUM: 0.0,
            MorseClass.SADDLE: 0.5,
            MorseClass.MAXIMUM: 1.0,
        }[self]

    @property
    def integer_index(self) -> int:
        return {
            MorseClass.MINIMUM: 0,
            MorseClass.SADDLE: 1,
            MorseClass.MAXIMUM: 2,
        }[self]


class PrimaryDefinition(str, Enum):
    EARLIEST_ARRIVING = "earliest_arriving"
    BRIGHTEST = "brightest"
    MINIMUM_IMAGE = "minimum_image"
    CATALOG_ANCHOR = "catalog_anchor"


class LensFamily(str, Enum):
    SIS = "sis"
    SIE_EXTERNAL_SHEAR = "sie_external_shear"
    EPL_EXTERNAL_SHEAR = "epl_external_shear"


@dataclass(frozen=True)
class Magnification:
    """Signed lensing magnification and derived amplitude quantities."""

    mu_signed: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu_signed) or self.mu_signed == 0.0:
            raise ValueError("mu_signed must be finite and nonzero")

    @property
    def mu_abs(self) -> float:
        return abs(self.mu_signed)

    @property
    def amplitude_factor(self) -> float:
        return math.sqrt(self.mu_abs)

    @property
    def parity(self) -> ImageParity:
        return ImageParity.POSITIVE if self.mu_signed > 0.0 else ImageParity.NEGATIVE

    @property
    def log_mu_abs(self) -> float:
        return math.log(self.mu_abs)


@dataclass(frozen=True)
class RelativeMagnification:
    """Secondary-over-primary ratios for one explicitly ordered pair."""

    relative_flux_magnification: float

    def __post_init__(self) -> None:
        value = self.relative_flux_magnification
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("relative_flux_magnification must be positive and finite")

    @classmethod
    def from_magnifications(
        cls, *, primary: Magnification, secondary: Magnification
    ) -> "RelativeMagnification":
        return cls(secondary.mu_abs / primary.mu_abs)

    @property
    def relative_strain_amplitude(self) -> float:
        return math.sqrt(self.relative_flux_magnification)

    @property
    def logit_relative_flux(self) -> float:
        value = self.relative_flux_magnification
        if value >= 1.0:
            raise ValueError("logit relative flux requires a ratio strictly between 0 and 1")
        return math.log(value) - math.log1p(-value)


def apparent_luminosity_distance(
    physical_luminosity_distance_mpc: float, magnification: Magnification
) -> float:
    """Return d_L/sqrt(|mu|), the distance measured by one lensed GW image."""

    if not math.isfinite(physical_luminosity_distance_mpc) or physical_luminosity_distance_mpc <= 0:
        raise ValueError("physical luminosity distance must be positive and finite")
    return physical_luminosity_distance_mpc / magnification.amplitude_factor


def signed_time_delay(arrival_time_secondary_s: float, arrival_time_primary_s: float) -> float:
    """Return secondary minus primary arrival time in seconds."""

    if not all(math.isfinite(v) for v in (arrival_time_secondary_s, arrival_time_primary_s)):
        raise ValueError("arrival times must be finite")
    return arrival_time_secondary_s - arrival_time_primary_s


def absolute_time_delay(arrival_time_a_s: float, arrival_time_b_s: float) -> float:
    """Return the nonnegative separation between two finite arrival times."""

    return abs(signed_time_delay(arrival_time_a_s, arrival_time_b_s))
