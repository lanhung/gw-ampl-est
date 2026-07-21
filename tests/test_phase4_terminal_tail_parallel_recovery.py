from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.terminal131 import (
    load_terminal_131k_contract,
    terminal_namespaces,
)
from scripts.phase4 import run_terminal_tail_parallel_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]


def _stopped_evidence(root: Path) -> tuple[Path, int]:
    evidence = root / "stopped-tail"
    zarr = evidence / "dataset" / "shards" / "shard-00000.partial" / "noisy.zarr"
    attempts = evidence / "dataset" / "attempts"
    zarr.mkdir(parents=True)
    attempts.mkdir(parents=True)
    for index in range(15):
        (zarr / f"{index}.0.0.0").write_bytes(b"")
    (attempts / "shard-00000.jsonl").write_text("\n" * 91839, encoding="utf-8")
    retained = sum(path.stat().st_size for path in evidence.rglob("*") if path.is_file())
    return evidence, retained


def test_parallel_tail_changes_only_execution_partition_and_ids() -> None:
    config = load_terminal_131k_contract(ROOT)
    original = terminal_namespaces(config)[1:]
    parallel = recovery._tail_namespaces(config)
    assert len(parallel) == 4
    assert sum(item.accepted_count for item in parallel) == 512
    assert len(original) == len(parallel)
    for before, after in zip(original, parallel):
        assert before.namespace_id == after.namespace_id
        assert before.split == after.split
        assert before.root_seed == after.root_seed
        assert before.accepted_count == after.accepted_count == 128
        assert before.proposal_distribution_id == after.proposal_distribution_id
        assert before.evaluation_distribution_id == after.evaluation_distribution_id
        assert before.balanced_tail_stratum == after.balanced_tail_stratum
        assert (before.shard_count, before.pairs_per_shard) == (1, 128)
        assert (after.shard_count, after.pairs_per_shard) == (32, 4)
        assert after.id_prefix.endswith("-parallel32")
        assert after.attempt_namespace.endswith("-parallel32")


def test_parallel_tail_identities_are_deterministic_and_distinct() -> None:
    config = load_terminal_131k_contract(ROOT)
    first = recovery._recovery_identities(config, "1" * 40)
    second = recovery._recovery_identities(deepcopy(config), "1" * 40)
    assert first == second
    values = {
        first["tail_parent_id"],
        first["combined_train_id"],
        *first["tail_dataset_ids"].values(),
    }
    assert len(values) == 6
    assert all("parallel32" in value for value in first["tail_dataset_ids"].values())


def test_parallel_tail_authorization_preserves_closed_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = load_yaml(ROOT / recovery.AUTHORIZATION)
    evidence, retained = _stopped_evidence(tmp_path)
    authorization["stopped_tail_evidence"]["path"] = str(evidence)
    authorization["stopped_tail_evidence"]["retained_bytes"] = retained

    original_loader = recovery.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original_loader(path)

    monkeypatch.setattr(recovery, "APPROVED_REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "load_yaml", _load)
    observed = recovery._authorization(config)
    execution = observed["parallel_execution_contract"]
    assert execution["worker_processes"] == 32
    assert execution["namespace_count"] == 4
    assert execution["accepted_pairs_per_namespace"] == 128
    assert execution["shards_per_namespace"] == 32
    assert execution["accepted_pairs_per_shard"] == 4
    assert execution["total_accepted_pairs"] == 512
    flags = observed["authorization"]
    assert flags["tail_recovery_authorized"] is True
    assert flags["reuse_published_train_increment_authorized"] is True
    assert flags["scientific_contract_change_authorized"] is False
    assert flags["training_authorized"] is False
    assert flags["final_evaluation_authorized"] is False
    assert flags["gwosc_gwtc_access_authorized"] is False


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("worker_processes", 64),
        ("shards_per_namespace", 16),
        ("accepted_pairs_per_shard", 8),
        ("total_accepted_pairs", 511),
    ],
)
def test_parallel_tail_authorization_rejects_execution_drift(
    field: str,
    invalid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = load_yaml(ROOT / recovery.AUTHORIZATION)
    evidence, retained = _stopped_evidence(tmp_path)
    authorization["stopped_tail_evidence"]["path"] = str(evidence)
    authorization["stopped_tail_evidence"]["retained_bytes"] = retained
    authorization["parallel_execution_contract"][field] = invalid

    original_loader = recovery.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original_loader(path)

    monkeypatch.setattr(recovery, "APPROVED_REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "load_yaml", _load)
    with pytest.raises(ValueError, match="count or scheduler mismatch"):
        recovery._authorization(config)


def test_parallel_tail_does_not_authorize_scientific_execution() -> None:
    authorization = load_yaml(ROOT / recovery.AUTHORIZATION)
    assert authorization["authorization"]["worker_64_authorized"] is False
    assert authorization["authorization"]["training_authorized"] is False
    assert authorization["authorization"]["architecture_selection_authorized"] is False
    assert authorization["authorization"]["calibration_authorized"] is False
    assert authorization["authorization"]["sbc_authorized"] is False
    assert authorization["authorization"]["final_evaluation_authorized"] is False
    assert authorization["authorization"]["extension_above_131072_authorized"] is False
    assert authorization["authorization"]["real_noise_authorized"] is False
    assert authorization["authorization"]["gwosc_gwtc_access_authorized"] is False
