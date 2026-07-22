from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.terminal131 import (
    load_terminal_131k_contract,
    terminal_namespaces,
)
from scripts.phase4 import run_terminal_tail_microshard_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]


def test_microshards_preserve_science_and_remove_fixed_four_case_quota() -> None:
    config = load_terminal_131k_contract(ROOT)
    original = terminal_namespaces(config)[1:]
    micro = recovery._tail_namespaces(config)
    assert len(original) == len(micro) == 4
    assert sum(item.accepted_count for item in micro) == 512
    for before, after in zip(original, micro):
        assert before.namespace_id == after.namespace_id
        assert before.split == after.split
        assert before.root_seed == after.root_seed
        assert before.accepted_count == after.accepted_count == 128
        assert before.proposal_distribution_id == after.proposal_distribution_id
        assert before.evaluation_distribution_id == after.evaluation_distribution_id
        assert before.balanced_tail_stratum == after.balanced_tail_stratum
        assert (before.shard_count, before.pairs_per_shard) == (1, 128)
        assert (after.shard_count, after.pairs_per_shard) == (128, 1)
        assert after.id_prefix.endswith("-micro128")
        assert after.attempt_namespace.endswith("-micro128")


def test_microshard_identities_are_deterministic_and_distinct() -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = load_yaml(ROOT / recovery.AUTHORIZATION)
    first = recovery._recovery_identities(config, "1" * 40, authorization)
    second = recovery._recovery_identities(
        deepcopy(config), "1" * 40, deepcopy(authorization)
    )
    assert first == second
    values = {
        first["tail_parent_id"],
        first["combined_train_id"],
        *first["tail_dataset_ids"].values(),
    }
    assert len(values) == 6
    assert all("micro128" in value for value in first["tail_dataset_ids"].values())


def test_microshard_config_freezes_dynamic_resource_caps() -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = recovery._authorization(config)
    namespace = recovery._tail_namespaces(config)[0]
    observed = recovery._namespace_config(config, namespace, authorization)
    execution = observed["execution"]
    assert execution["qualification_worker_processes"] == 32
    assert execution["maximum_attempts_per_worker"] == 2_000_000
    assert execution["maximum_active_seconds_per_worker"] == 345_600
    assert observed["shard_count"] == 128
    assert observed["pairs_per_shard"] == 1


def test_implementation_gate_keeps_all_scientific_execution_closed() -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = recovery._authorization(config)
    assert authorization["authorization_status"] == "implementation_only"
    assert authorization["authorization"]["tail_microshard_recovery_authorized"] is False
    for key in (
        "scientific_contract_change_authorized",
        "training_authorized",
        "architecture_selection_authorized",
        "calibration_authorized",
        "sbc_authorized",
        "final_evaluation_authorized",
        "extension_above_131072_authorized",
        "real_noise_authorized",
        "gwosc_gwtc_access_authorized",
    ):
        assert authorization["authorization"][key] is False


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("worker_processes", 64),
        ("shards_per_namespace", 32),
        ("accepted_pairs_per_shard", 4),
        ("total_accepted_pairs", 511),
    ],
)
def test_microshard_authorization_rejects_layout_drift(
    field: str, invalid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = deepcopy(load_yaml(ROOT / recovery.AUTHORIZATION))
    authorization["microshard_execution_contract"][field] = invalid
    original = recovery.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original(path)

    monkeypatch.setattr(recovery, "load_yaml", _load)
    with pytest.raises(ValueError, match="count or scheduler contract"):
        recovery._authorization(config)


def test_recovery_wheel_requires_byte_identical_generator_core(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.whl"
    matching = tmp_path / "matching.whl"
    changed = tmp_path / "changed.whl"
    with ZipFile(original, "w") as archive:
        archive.writestr("gwlens_mm/production/generator.py", b"frozen")
    with ZipFile(matching, "w") as archive:
        archive.writestr("gwlens_mm/production/generator.py", b"frozen")
    with ZipFile(changed, "w") as archive:
        archive.writestr("gwlens_mm/production/generator.py", b"changed")
    assert len(recovery._verify_generator_core_unchanged(original, matching)) == 64
    with pytest.raises(ValueError, match="changed frozen generator code"):
        recovery._verify_generator_core_unchanged(original, changed)
