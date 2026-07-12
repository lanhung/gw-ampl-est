import json
from pathlib import Path

import pytest

from gwlens_mm.production.run_control import AttemptJournal, AttemptRecord


def attempt(index: int, status: str = "rejected") -> AttemptRecord:
    return AttemptRecord(
        attempt_id=index,
        proposal_seed=100 + index,
        lens_family="sie_external_shear",
        em_cell="full_precise_spectroscopic",
        status=status,
        rejection_reason="selection" if status == "rejected" else None,
        pair_id=f"pair-{index}" if status == "accepted" else None,
        source_id=f"source-{index}",
        lens_id=f"lens-{index}",
        physical_system_id=f"system-{index}",
    )


def test_attempt_journal_is_append_only_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    with AttemptJournal(path) as journal:
        journal.append(attempt(0))
        journal.append(attempt(1, "accepted"))
    with AttemptJournal(path) as journal:
        assert journal.next_attempt_id == 2
        journal.append(attempt(2))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["attempt_id"] for row in rows] == [0, 1, 2]


def test_attempt_journal_rejects_gap_and_duplicate_system(tmp_path: Path) -> None:
    with AttemptJournal(tmp_path / "attempts.jsonl") as journal:
        with pytest.raises(ValueError, match="contiguously"):
            journal.append(attempt(1))
        journal.append(attempt(0))
        duplicate = AttemptRecord(**{**attempt(1).__dict__, "physical_system_id": "system-0"})
        with pytest.raises(ValueError, match="already exists"):
            journal.append(duplicate)
