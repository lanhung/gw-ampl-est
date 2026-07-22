from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.terminal131 import load_terminal_131k_contract
from gwlens_mm.provenance import configuration_hash
from scripts.phase4 import closeout_terminal_tail_microshard_recovery as closeout
from scripts.phase4 import run_terminal_tail_microshard_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]


def _authorized() -> dict[str, object]:
    value = deepcopy(load_yaml(ROOT / recovery.AUTHORIZATION))
    value["authorization_status"] = (
        "authorized_engineering_microshard_recovery_only"
    )
    value["authorization"]["tail_microshard_recovery_authorized"] = True
    value["immutable_orchestration"]["git_commit"] = "1" * 40
    return value


def _result(authorization: dict[str, object]) -> dict[str, object]:
    config = load_terminal_131k_contract(ROOT)
    identities = recovery._recovery_identities(config, "1" * 40, authorization)
    return {
        "status": "passed",
        "phase": "4-terminal-tail-microshard-recovery",
        "generator_commit": recovery.GENERATOR_COMMIT,
        "orchestration_commit": "1" * 40,
        "configuration_hash": configuration_hash(config),
        "train_parent_id": recovery.TRAIN_PARENT_ID,
        "train_dataset_id": recovery.TRAIN_DATASET_ID,
        **identities,
        "new_train_accepted_count": 65536,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "development_tail_shard_count": 512,
        "development_tail_shards_per_namespace": 128,
        "development_tail_pairs_per_shard": 1,
        "terminal_train_accepted_count": 131072,
        "remaining_free_bytes": 100000000000,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "failed_parallel32_evidence_reused": False,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def test_closeout_accepts_exact_microshard_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _authorized()
    original = closeout.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original(path)

    monkeypatch.setattr(closeout, "load_yaml", _load)
    identities, observed = closeout.validate_microshard_execution_result(
        ROOT, load_terminal_131k_contract(ROOT), _result(authorization)
    )
    assert identities["tail_parent_id"].startswith("phase4-terminal-tail-micro128-")
    assert observed["microshard_execution_contract"]["total_shard_count"] == 512


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("development_tail_accepted_count", 511),
        ("development_tail_shard_count", 128),
        ("development_tail_shards_per_namespace", 32),
        ("development_tail_pairs_per_shard", 4),
        ("failed_parallel32_evidence_reused", True),
        ("train_131k_probe_authorized", True),
    ],
)
def test_closeout_rejects_count_layout_or_safety_drift(
    field: str,
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _authorized()
    original = closeout.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original(path)

    monkeypatch.setattr(closeout, "load_yaml", _load)
    result = _result(authorization)
    result[field] = invalid
    with pytest.raises(ValueError, match="count or safety contract"):
        closeout.validate_microshard_execution_result(
            ROOT, load_terminal_131k_contract(ROOT), result
        )
