import pytest

from gwlens_mm.schema import SplitName
from gwlens_mm.splits import SplitAssignment, SplitLeakageError, validate_grouped_splits


def assignment(split, suffix, **overrides):
    values = {
        "split": split,
        "pair_id": f"pair-{suffix}",
        "source_id": f"source-{suffix}",
        "lens_id": f"lens-{suffix}",
        "physical_system_id": f"system-{suffix}",
        "noise_segment_ids": (f"noise-{suffix}-a", f"noise-{suffix}-b"),
        "augmentation_parent_id": f"augmentation-{suffix}",
    }
    values.update(overrides)
    return SplitAssignment(**values)


def test_disjoint_grouped_splits_pass():
    validate_grouped_splits(
        [assignment(SplitName.TRAIN, "a"), assignment(SplitName.CALIBRATION, "b")]
    )


@pytest.mark.parametrize(
    "override",
    [
        {"source_id": "source-a"},
        {"lens_id": "lens-a"},
        {"physical_system_id": "system-a"},
        {"noise_segment_ids": ("noise-a-a", "noise-b-x")},
        {"augmentation_parent_id": "augmentation-a"},
    ],
)
def test_corrupted_grouped_splits_are_rejected(override):
    records = [
        assignment(SplitName.TRAIN, "a"),
        assignment(SplitName.IID_TEST, "b", **override),
    ]
    with pytest.raises(SplitLeakageError, match="leaks"):
        validate_grouped_splits(records)
