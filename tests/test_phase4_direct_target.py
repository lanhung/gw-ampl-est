from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import gwlens_mm.release_gate as release_gate
from gwlens_mm.config import load_yaml
from gwlens_mm.production.proposal_adapter import sample_production_proposal
from gwlens_mm.production.stage_a import (
    DIRECT_TARGET_ID,
    build_namespace_config,
    derive_canary_identity,
    load_phase4_contract,
    validate_canary_manifest,
    validate_direct_target_record,
)
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.release_gate import evaluate_phase4_release_gate
from gwlens_mm.schema import SplitName, V2Record
from scripts.phase4.run_direct_target_canary import _verify_canary_authorization
from scripts.phase4.run_stage_a import _verify_execution_authorization

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/phase4_direct_target_stage_a.yaml"
PREREGISTRATION = ROOT / "configs/statistics/direct_target_stage_a_preregistration.yaml"
PARENT = ROOT / "configs/statistics/adaptive_scientific_production_preregistration.yaml"
EXAMPLE = ROOT / "examples/v2_metadata_example.json"
RC3_HASH = "6082475631539d3069edacc52f41b37fb8fe725ccd7c6bc9980cc3008795a927"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"


def _direct_record(split: SplitName, dataset: str) -> V2Record:
    data = deepcopy(json.loads(EXAMPLE.read_text()))
    data["pair"]["split"] = split.value
    data["pair"]["dataset_version"] = dataset
    return V2Record.from_dict(data)


def test_rc4_is_a_new_delta_without_modifying_rc3() -> None:
    parent = load_yaml(PARENT)
    preregistration = load_yaml(PREREGISTRATION)
    assert configuration_hash(parent) == RC3_HASH
    assert configuration_hash(preregistration) == RC4_HASH
    assert preregistration["preregistration_version"] == "1.1.0-rc.4"
    assert preregistration["parent_adaptive_preregistration"]["canonical_hash"] == RC3_HASH


def test_phase4_contract_loads_but_all_execution_flags_remain_false() -> None:
    config, preregistration, design = load_phase4_contract(ROOT)
    assert config["preregistration"]["canonical_hash"] == RC4_HASH
    assert design["authorization_status"] == "authorized_design_and_implementation_only"
    authorization = preregistration["authorization"]
    assert authorization.pop("design_and_implementation_authorized") is True
    assert all(value is False for value in authorization.values())
    assert config["execution"]["scientific_execution_enabled"] is False
    assert config["execution"]["canary_execution_enabled"] is False
    assert config["authorization"]["future_execution_path"] == (
        "configs/execution/phase4_direct_target_stage_a_authorization.yaml"
    )
    assert config["authorization"]["canary_execution_path"] is None


def test_stage_a_authorization_is_exact_count_and_training_closed() -> None:
    config, _, _ = load_phase4_contract(ROOT)
    authorization = load_yaml(
        ROOT / "configs/execution/phase4_direct_target_stage_a_authorization.yaml"
    )
    assert authorization["authorization_status"] == (
        "authorized_scientific_materialization_only"
    )
    flags = authorization["authorization"]
    assert flags["scientific_data_generation_authorized"] is True
    assert flags["stage_a_materialization_authorized"] is True
    assert flags["model_training_authorized"] is False
    assert flags["calibration_authorized"] is False
    assert flags["sbc_authorized"] is False
    assert flags["iid_ood_mismatch_evaluation_authorized"] is False
    assert flags["gwosc_gwtc_access_authorized"] is False
    contract = authorization["stage_a_contract"]
    assert (
        contract["train_accepted_count"],
        contract["validation_accepted_count"],
        contract["total_accepted_count"],
        contract["total_shard_count"],
    ) == (32768, 6144, 38912, 304)
    verified = _verify_execution_authorization(
        config, authorization["immutable_generator"]["git_commit"]
    )
    assert verified["authorizing_commit"] == authorization["authorizing_commit"]


def test_direct_target_contract_has_exact_unit_weights() -> None:
    preregistration = load_yaml(PREREGISTRATION)
    direct = preregistration["direct_target_training_contract"]
    assert direct["training_proposal_id"] == DIRECT_TARGET_ID
    assert direct["evaluation_target_id"] == DIRECT_TARGET_ID
    assert direct["q_train_equals_p_eval"] is True
    assert direct["log_importance_weight"] == 0.0
    assert direct["importance_weight"] == 1.0
    assert direct["rc5_weighted_scientific_training_authorized"] is False
    assert direct["proposal_v3_scientific_use_authorized"] is False


def test_stage_a_count_and_shard_arithmetic_is_exact() -> None:
    config = load_yaml(CONFIG)
    stage = config["stage_a"]
    assert stage["train"]["accepted_pair_count"] == 32768
    assert stage["train"]["shard_count"] * stage["train"]["pairs_per_shard"] == 32768
    assert stage["validation"]["accepted_pair_count"] == 6144
    assert (
        stage["validation"]["shard_count"]
        * stage["validation"]["pairs_per_shard"]
        == 6144
    )
    assert stage["total_accepted_pair_count"] == 38912
    assert stage["total_shard_count"] == 304


def test_direct_target_adapter_samples_q_equal_p_deterministically() -> None:
    proposal = load_yaml(ROOT / "configs/proposals/proposal_v3_target_anchored_mixture.yaml")
    left = sample_production_proposal(
        np.random.default_rng(41),
        mode="evaluation_target_direct",
        proposal_config=proposal,
    )
    right = sample_production_proposal(
        np.random.default_rng(41),
        mode="evaluation_target_direct",
        proposal_config=proposal,
    )
    assert left == right
    assert left.component == "evaluation_target"
    assert left.population.proposal_log_probability == left.population.evaluation_log_probability
    assert left.component_log_densities == {
        "evaluation_target": left.population.evaluation_log_probability
    }


@pytest.mark.parametrize(
    ("namespace", "canary", "split", "count", "prefix"),
    [
        ("train", False, "train", 32768, "phase4-train"),
        ("validation", False, "validation", 6144, "phase4-validation"),
        ("train_namespace", True, "generator_qualification", 8, "phase4-canary-train"),
        (
            "validation_namespace",
            True,
            "generator_qualification",
            8,
            "phase4-canary-validation",
        ),
    ],
)
def test_namespace_configs_use_the_same_direct_target_path(
    namespace: str, canary: bool, split: str, count: int, prefix: str
) -> None:
    config = load_yaml(CONFIG)
    generated = build_namespace_config(ROOT, config, namespace, canary=canary)
    context = generated["production_context"]
    assert context["proposal_mode"] == "evaluation_target_direct"
    assert context["proposal_distribution_id"] == DIRECT_TARGET_ID
    assert context["evaluation_distribution_id"] == DIRECT_TARGET_ID
    assert context["split"] == split
    assert context["id_prefix"] == prefix
    assert generated["accepted_pair_count"] == count


@pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.VALIDATION])
def test_typed_direct_target_record_validation(split: SplitName) -> None:
    dataset = f"phase4-{split.value}-dataset"
    record = _direct_record(split, dataset)
    validate_direct_target_record(record, expected_split=split, expected_dataset=dataset)
    data = record.to_dict()
    data["provenance"]["distribution"]["importance_weight"] = 0.9
    with pytest.raises(ValueError, match="unit weight"):
        validate_direct_target_record(
            V2Record.from_dict(data), expected_split=split, expected_dataset=dataset
        )


def test_canary_is_exactly_8_plus_8_and_has_distinct_nonofficial_identities() -> None:
    config = load_yaml(CONFIG)
    canary = config["disposable_canary"]
    assert canary["train_namespace"]["accepted_pair_count"] == 8
    assert canary["validation_namespace"]["accepted_pair_count"] == 8
    assert canary["total_accepted_pair_count"] == 16
    identity = derive_canary_identity(config, "1" * 40)
    assert identity.train_dataset_id != identity.validation_dataset_id
    assert "phase3ca" not in identity.parent_run_id
    assert "stage-a" not in identity.parent_run_id


def test_canary_manifest_requires_resume_and_permanent_use_denials() -> None:
    manifest = {
        "status": "passed",
        "generator_commit": "1" * 40,
        "accepted_pair_count": 16,
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "throughput_or_ess_inspected": False,
        "resume_first_namespace_byte_identical": True,
    }
    validate_canary_manifest(manifest, "1" * 40)
    manifest["throughput_or_ess_inspected"] = True
    with pytest.raises(ValueError, match="efficiency endpoint"):
        validate_canary_manifest(manifest, "1" * 40)


def test_canary_authorization_is_separate_from_frozen_rc4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "gwlens_mm.whl"
    wheel.write_bytes(b"frozen-wheel")
    lock = ROOT / "configs/environment/phase4-autodl-requirements.lock.txt"
    authorization = {
        "authorization_status": "authorized_disposable_canary_only",
        "preregistration": {"canonical_hash": RC4_HASH},
        "immutable_generator": {
            "git_commit": "1" * 40,
            "wheel_path": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "editable_install_authorized": False,
        },
        "canary_contract": {
            "train_namespace_accepted_pairs": 8,
            "validation_namespace_accepted_pairs": 8,
            "total_accepted_pairs": 16,
            "throughput_inspection_authorized": False,
            "ess_inspection_authorized": False,
        },
        "authorization": {
            "disposable_canary_execution_authorized": True,
            "accepted_pair_generator_authorized_within_canary_only": True,
            "scientific_data_generation_authorized": False,
            "stage_a_materialization_authorized": False,
            "model_training_authorized": False,
            "calibration_authorized": False,
            "sbc_authorized": False,
            "scientific_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "use_policy": {
            "scientific_use_authorized": False,
            "training_use_authorized": False,
            "calibration_use_authorized": False,
            "test_use_authorized": False,
            "permanent_exclusion_from_all_scientific_splits": True,
        },
    }
    path = tmp_path / "authorization.yaml"
    path.write_text(yaml.safe_dump(authorization))
    monkeypatch.chdir(ROOT)
    loaded = _verify_canary_authorization(path, "1" * 40, RC4_HASH)
    assert loaded["canary_contract"]["total_accepted_pairs"] == 16


def test_release_gate_is_fail_closed_and_creates_no_official_identity() -> None:
    result = evaluate_phase4_release_gate(ROOT, generator_commit="1" * 40)
    assert result["status"] == "blocked_preexecution"
    assert result["official_identities"] is None
    assert result["scientific_data_generation_authorized"] is False
    assert result["model_training_authorized"] is False
    assert result["gwosc_gwtc_access_authorized"] is False


def test_release_gate_creates_official_identities_only_after_every_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, preregistration, design = load_phase4_contract(ROOT)
    config = deepcopy(config)
    generator_commit = "1" * 40
    canary = {
        "status": "passed",
        "generator_commit": generator_commit,
        "accepted_pair_count": 16,
        "scientific_use_authorized": False,
        "training_use_authorized": False,
        "throughput_or_ess_inspected": False,
        "resume_first_namespace_byte_identical": True,
    }
    canary_path = tmp_path / "canary.json"
    canary_path.write_text(json.dumps(canary, sort_keys=True) + "\n")
    wheel_path = tmp_path / "gwlens_mm.whl"
    wheel_path.write_bytes(b"frozen-generator-wheel")
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    dependency_lock = ROOT / config["environment"]["dependency_lock_path"]
    dependency_hash = hashlib.sha256(dependency_lock.read_bytes()).hexdigest()
    authorization = {
        "authorization_status": "authorized_scientific_materialization_only",
        "authorizing_commit": config["authorization"]["authorizing_git_commit"],
        "immutable_generator": {
            "git_commit": generator_commit,
            "wheel_path": str(wheel_path),
            "wheel_sha256": wheel_hash,
            "environment_lock_sha256": dependency_hash,
        },
        "authorization": {
            "disposable_canary_accepted": True,
            "scientific_data_generation_authorized": True,
            "stage_a_materialization_authorized": True,
            "model_training_authorized": False,
            "calibration_authorized": False,
            "sbc_authorized": False,
            "iid_ood_mismatch_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "stage_a_contract": {
            "train_accepted_count": 32768,
            "validation_accepted_count": 6144,
            "total_accepted_count": 38912,
        },
    }
    authorization_path = tmp_path / "authorization.yaml"
    authorization_path.write_text(yaml.safe_dump(authorization))
    staging = tmp_path / "staging"
    publication = tmp_path / "publication"
    staging.mkdir()
    publication.mkdir()
    config["authorization"]["future_execution_path"] = str(authorization_path)
    config["release"].update(
        {
            "final_generator_commit": generator_commit,
            "generator_wheel_sha256": wheel_hash,
            "canary_manifest_path": str(canary_path),
            "canary_manifest_sha256": hashlib.sha256(canary_path.read_bytes()).hexdigest(),
        }
    )
    config["paths"]["stage_a_staging_root"] = str(staging)
    config["paths"]["stage_a_publication_root"] = str(publication)
    monkeypatch.setattr(
        release_gate, "load_phase4_contract", lambda root, path: (config, preregistration, design)
    )
    monkeypatch.setattr(release_gate, "verify_generator_commit", lambda root, commit: None)
    monkeypatch.setattr(release_gate, "_git_branch", lambda root: "phase4/direct-target-stage-a")
    monkeypatch.setattr(release_gate, "_git_clean", lambda root: True)
    monkeypatch.setattr(release_gate, "verify_psd_files", lambda value: {"status": "passed"})
    monkeypatch.setattr(
        release_gate.shutil,
        "disk_usage",
        lambda path: release_gate.shutil._ntuple_diskusage(500_000_000_000, 0, 500_000_000_000),
    )
    result = release_gate.evaluate_phase4_release_gate(
        ROOT, generator_commit=generator_commit
    )
    assert result["status"] == "ready_for_official_execution"
    assert result["blockers"] == []
    assert result["official_identities"] is not None
    assert result["official_identities"]["train_dataset_id"] != result[
        "official_identities"
    ]["validation_dataset_id"]


def test_environment_and_commitment_hashes_are_reproducible() -> None:
    environment = load_yaml(ROOT / "configs/environment/phase4-autodl-environment.yaml")
    lock = ROOT / environment["dependency_lock_path"]
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == environment["dependency_lock_sha256"]
    commitment = ROOT / "results/phase4/final_evaluation_commitment.json"
    expected = (ROOT / "results/phase4/final_evaluation_commitment.sha256").read_text().split()[0]
    assert hashlib.sha256(commitment.read_bytes()).hexdigest() == expected
    finalized = json.loads(commitment.read_text())
    assert finalized["commitment_status"] == "finalized_before_training"
    assert finalized["future_scientific_generator_commit"] == (
        "bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac"
    )


def test_scripts_do_not_access_distribution_schema_fields_directly() -> None:
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text()
        assert ".evaluation_prior_log_probability" not in text
        assert ".proposal_log_probability" not in text
        assert ".importance_weight" not in text
