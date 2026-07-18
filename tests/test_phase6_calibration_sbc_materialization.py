from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.production.calibration_sbc import (
    build_calibration_sbc_namespace_config,
    calibration_sbc_namespaces,
    derive_calibration_sbc_identities,
    dry_run_plan,
    load_calibration_sbc_contract,
    validate_future_materialization_authorization,
)
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.schema import SplitName

ROOT = Path(__file__).resolve().parents[1]
CONFIG_HASH = "c55dd46d1afefe60753e2b112363261015ea914d55e80c4a5108721cb0b6a17e"


def test_calibration_sbc_data_plan_is_exact_and_execution_closed() -> None:
    config, authorization = load_calibration_sbc_contract(ROOT)
    assert configuration_hash(config) == CONFIG_HASH
    assert authorization["authorization_status"] == "authorized_implementation_only"
    assert (
        authorization["authorization"][
            "checkpoint_score_inference_runner_implementation_authorized"
        ]
        is True
    )
    assert authorization["authorization"]["model_checkpoint_access_authorized"] is False
    assert config["execution"]["enabled"] is False
    assert config["execution"]["materialization_authorized"] is False
    namespaces = calibration_sbc_namespaces(config)
    assert [(item.split, item.accepted_count, item.shard_count) for item in namespaces] == [
        (SplitName.CALIBRATION_FIT, 4096, 32),
        (SplitName.SBC_DIAGNOSTIC, 2048, 16),
    ]
    assert sum(item.accepted_count for item in namespaces) == 6144
    assert sum(item.shard_count for item in namespaces) == 48
    assert len({item.root_seed for item in namespaces}) == 2


def test_built_namespaces_are_direct_target_and_disjoint() -> None:
    config, _ = load_calibration_sbc_contract(ROOT)
    namespaces = calibration_sbc_namespaces(config)
    built = [
        build_calibration_sbc_namespace_config(ROOT, config, namespace)
        for namespace in namespaces
    ]
    assert {item["production_context"]["split"] for item in built} == {
        "calibration_fit",
        "sbc_diagnostic",
    }
    assert {item["production_context"]["proposal_mode"] for item in built} == {
        "evaluation_target_direct"
    }
    assert len({item["production_context"]["id_prefix"] for item in built}) == 2
    assert all(item["accepted_pair_count"] == item["shard_count"] * 128 for item in built)


def test_seed_and_count_drift_fail_closed() -> None:
    config, _ = load_calibration_sbc_contract(ROOT)
    changed = deepcopy(config)
    changed["splits"]["calibration_fit"]["root_seed"] += 1
    with pytest.raises(ValueError, match="root seed"):
        calibration_sbc_namespaces(changed)
    changed = deepcopy(config)
    changed["splits"]["sbc_diagnostic"]["accepted_pair_count"] = 1024
    with pytest.raises(ValueError, match="shard arithmetic"):
        calibration_sbc_namespaces(changed)


def test_dry_run_cannot_resolve_identity_or_execute_science() -> None:
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_execution_closed"
    assert plan["accepted_pair_count"] == 6144
    assert plan["shard_count"] == 48
    assert plan["official_identities"] is None
    assert plan["pair_generated"] is False
    assert plan["calibration_fitted"] is False
    assert plan["sbc_executed"] is False
    assert plan["final_evaluation_accessed"] is False


def test_future_identity_and_materialization_gate_are_exact() -> None:
    config, _ = load_calibration_sbc_contract(ROOT)
    commit = "a" * 40
    identities = derive_calibration_sbc_identities(config, commit)
    assert identities.parent_run_id.startswith("phase6-stage-c-")
    assert identities.calibration_dataset_id != identities.sbc_dataset_id
    authorization = {
        "authorization_status": "authorized_exact_calibration_sbc_materialization_only",
        "immutable_generator": {"git_commit": commit},
        "frozen_contract": {
            "configuration_hash": CONFIG_HASH,
            "calibration_sbc_preregistration_hash": (
                "033b996930c93e7e4a9881fc3de49bb85cf4be96fcbd890bf2543b46368c9d8e"
            ),
        },
        "entry_gate": {
            "training_size_locked": True,
            "architecture_locked": True,
            "three_model_seeds_retained": True,
        },
        "materialization_contract": {
            "calibration_fit_accepted_count": 4096,
            "sbc_diagnostic_accepted_count": 2048,
            "total_accepted_count": 6144,
            "total_shard_count": 48,
        },
        "authorization": {
            "calibration_sbc_materialization_authorized": True,
            "accepted_pair_generator_authorized_within_stage_c_only": True,
            "calibration_fit_statistics_authorized": False,
            "sbc_statistics_authorized": False,
            "checkpoint_access_authorized": False,
            "final_evaluation_authorized": False,
            "model_retraining_or_tuning_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    }
    validate_future_materialization_authorization(
        authorization, config=config, generator_commit=commit
    )
    changed = deepcopy(authorization)
    changed["authorization"]["sbc_statistics_authorized"] = True
    with pytest.raises(PermissionError, match="materialization-only"):
        validate_future_materialization_authorization(
            changed, config=config, generator_commit=commit
        )


def test_real_release_gate_is_blocked_before_future_authorization(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked.json"
    authorization = ROOT / (
        "configs/execution/"
        "phase6_calibration_sbc_materialization_stack_authorization.yaml"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase6/run_calibration_sbc_materialization.py"),
            "preflight",
            "--authorization",
            str(authorization),
            "--orchestration-commit",
            "a" * 40,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "blocked_preexecution"
    assert evidence["official_identities"] is None
