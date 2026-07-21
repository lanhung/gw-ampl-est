from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.terminal131 import load_terminal_131k_contract
from gwlens_mm.provenance import configuration_hash
from scripts.phase4 import closeout_terminal_tail_recovery as closeout
from scripts.phase4 import run_terminal_tail_parallel_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]


def _result_fixture() -> dict[str, object]:
    config = load_terminal_131k_contract(ROOT)
    authorization = load_yaml(ROOT / recovery.AUTHORIZATION)
    orchestration = str(authorization["immutable_orchestration"]["git_commit"])
    identities = recovery._recovery_identities(config, orchestration)
    return {
        "status": "passed",
        "phase": "4-terminal-tail-parallel-recovery",
        "generator_commit": recovery.GENERATOR_COMMIT,
        "orchestration_commit": orchestration,
        "configuration_hash": configuration_hash(config),
        "preregistration_hash": config["preregistration"]["canonical_hash"],
        "train_parent_id": recovery.TRAIN_PARENT_ID,
        "train_dataset_id": recovery.TRAIN_DATASET_ID,
        **identities,
        "new_train_accepted_count": 65536,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "development_tail_shard_count": 128,
        "terminal_train_accepted_count": 131072,
        "remaining_free_bytes": 100000000000,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "stopped_tail_evidence_reused": False,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def test_recovery_closeout_accepts_exact_result() -> None:
    config = load_terminal_131k_contract(ROOT)
    identities, authorization = closeout.validate_tail_recovery_execution_result(
        ROOT, config, _result_fixture()
    )
    assert identities["tail_parent_id"].startswith(
        "phase4-terminal-tail-parallel32-"
    )
    assert identities["combined_train_id"].startswith("phase4-train-131k-")
    assert authorization["parallel_execution_contract"]["worker_processes"] == 32


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("development_tail_accepted_count", 511),
        ("development_tail_shard_count", 127),
        ("terminal_train_accepted_count", 131071),
        ("all_importance_weights_one", False),
        ("stopped_tail_evidence_reused", True),
        ("train_131k_probe_authorized", True),
        ("extension_above_131072_authorized", True),
    ],
)
def test_recovery_closeout_rejects_count_or_safety_drift(
    field: str, invalid: object
) -> None:
    config = load_terminal_131k_contract(ROOT)
    result = _result_fixture()
    result[field] = invalid
    with pytest.raises(ValueError, match="count or safety contract"):
        closeout.validate_tail_recovery_execution_result(ROOT, config, result)


def test_recovery_closeout_rejects_identity_drift() -> None:
    config = load_terminal_131k_contract(ROOT)
    result = _result_fixture()
    result["combined_train_id"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        closeout.validate_tail_recovery_execution_result(ROOT, config, result)


def test_recovery_closeout_rejects_worker64_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_terminal_131k_contract(ROOT)
    authorization = deepcopy(load_yaml(ROOT / recovery.AUTHORIZATION))
    authorization["authorization"]["worker_64_authorized"] = True
    original = closeout.load_yaml

    def _load(path: Path) -> dict[str, object]:
        if Path(path) == ROOT / recovery.AUTHORIZATION:
            return authorization
        return original(path)

    monkeypatch.setattr(closeout, "load_yaml", _load)
    with pytest.raises(ValueError, match="authorization changed"):
        closeout.validate_tail_recovery_execution_result(
            ROOT, config, _result_fixture()
        )
