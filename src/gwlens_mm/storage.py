"""Storage-size calculations used by the Phase 1 architecture decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageEstimate:
    pair_count: int
    samples_per_image: int
    image_slots: int
    detector_slots: int
    products: int
    bytes_per_sample: int

    @property
    def bytes(self) -> int:
        values = (
            self.pair_count,
            self.samples_per_image,
            self.image_slots,
            self.detector_slots,
            self.products,
            self.bytes_per_sample,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all storage dimensions must be positive")
        result = 1
        for value in values:
            result *= value
        return result

    @property
    def mebibytes(self) -> float:
        return self.bytes / 1024**2

    @property
    def gibibytes(self) -> float:
        return self.bytes / 1024**3


def strain_storage_estimate(pair_count: int, sample_count: int = 4096) -> StorageEstimate:
    return StorageEstimate(
        pair_count=pair_count,
        samples_per_image=sample_count,
        image_slots=2,
        detector_slots=3,
        products=3,
        bytes_per_sample=4,
    )
