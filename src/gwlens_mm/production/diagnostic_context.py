"""Selection classifiers for frozen final-evaluation contexts."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence


class MagnifiedImage(Protocol):
    @property
    def mu_signed(self) -> float: ...


class BalancedTailStratum(str, Enum):
    HIGH_ABSOLUTE_MAGNIFICATION = "high_absolute_magnification"
    EXTREME_RELATIVE_MAGNIFICATION = "extreme_relative_magnification"
    SECOND_IMAGE_NEAR_THRESHOLD = "second_image_near_threshold"
    EXTREME_PROFILE_OR_ENVIRONMENT = "extreme_profile_or_environment"


def classify_balanced_tail(
    selected_images: Sequence[MagnifiedImage],
    *,
    secondary_network_snr: float,
    external_convergence: float,
    density_slope: float,
) -> BalancedTailStratum | None:
    """Apply the preregistered first-matching tail priority exactly."""

    if len(selected_images) != 2:
        raise ValueError("tail classification requires exactly two selected images")
    magnifications = tuple(abs(float(image.mu_signed)) for image in selected_images)
    if max(magnifications) >= 20.0:
        return BalancedTailStratum.HIGH_ABSOLUTE_MAGNIFICATION
    if min(magnifications) / max(magnifications) <= 0.10:
        return BalancedTailStratum.EXTREME_RELATIVE_MAGNIFICATION
    if 10.0 <= secondary_network_snr <= 12.0:
        return BalancedTailStratum.SECOND_IMAGE_NEAR_THRESHOLD
    if abs(external_convergence) >= 0.10 or not 1.75 <= density_slope <= 2.40:
        return BalancedTailStratum.EXTREME_PROFILE_OR_ENVIRONMENT
    return None
