from pathlib import Path

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.production.observations import balanced_em_cell

ROOT = Path(__file__).resolve().parents[1]


def test_em_assignment_is_exactly_balanced_and_deterministic() -> None:
    config = load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")
    cells = tuple(config["em_observation_model"]["availability_cells"])
    assignments = [
        balanced_em_cell(index, f"qualification-system-{index:06d}", cells)
        for index in range(64)
    ]
    assert assignments == [
        balanced_em_cell(index, f"qualification-system-{index:06d}", cells)
        for index in range(64)
    ]
    assert {cell: assignments.count(cell) for cell in cells} == {cell: 8 for cell in cells}


def test_em_assignment_does_not_use_process_random_hash() -> None:
    config = load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")
    cells = tuple(config["em_observation_model"]["availability_cells"])
    first = balanced_em_cell(3, "qualification-system-000003", cells)
    np.random.seed(999)
    assert balanced_em_cell(3, "qualification-system-000003", cells) == first
