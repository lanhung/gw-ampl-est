"""Grouped split assignments and leakage detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .schema import SplitName


class SplitLeakageError(ValueError):
    pass


@dataclass(frozen=True)
class SplitAssignment:
    split: SplitName
    pair_id: str
    source_id: str
    lens_id: str
    physical_system_id: str
    noise_segment_ids: Tuple[str, ...]
    augmentation_parent_id: Optional[str] = None


def validate_grouped_splits(assignments: Iterable[SplitAssignment]) -> None:
    records = tuple(assignments)
    if not records:
        raise ValueError("at least one split assignment is required")
    pair_ids = [record.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise SplitLeakageError("pair_id must identify one record exactly")

    grouped: Dict[str, Dict[str, SplitName]] = {
        "source_id": {},
        "lens_id": {},
        "physical_system_id": {},
        "noise_segment_id": {},
        "augmentation_parent_id": {},
    }

    def register(kind: str, identifier: Optional[str], split: SplitName) -> None:
        if not identifier:
            return
        previous = grouped[kind].get(identifier)
        if previous is not None and previous is not split:
            raise SplitLeakageError(
                f"{kind}={identifier!r} leaks between {previous.value} and {split.value}"
            )
        grouped[kind][identifier] = split

    for record in records:
        register("source_id", record.source_id, record.split)
        register("lens_id", record.lens_id, record.split)
        register("physical_system_id", record.physical_system_id, record.split)
        register("augmentation_parent_id", record.augmentation_parent_id, record.split)
        for noise_segment_id in record.noise_segment_ids:
            register("noise_segment_id", noise_segment_id, record.split)
