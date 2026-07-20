from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.diagnostic_context import classify_balanced_tail
from gwlens_mm.production.terminal131 import (
    TAIL_STRATA,
    build_terminal_namespace_config,
    derive_terminal_identities,
    load_terminal_131k_contract,
    terminal_namespaces,
)
from gwlens_mm.provenance import configuration_hash
from scripts.phase4.closeout_terminal_131k import validate_terminal_execution_result
from scripts.phase4.run_terminal_131k import (
    _validate_scheduler_authorization,
    evaluate_release_gate,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/phase4_terminal_131k.yaml"


class _Image:
    def __init__(self, mu_signed: float) -> None:
        self.mu_signed = mu_signed


def test_terminal_131k_execution_design_is_exact_and_closed() -> None:
    config = load_terminal_131k_contract(ROOT)
    train = config["train_increment"]
    tail = config["development_tail"]
    assert train["accepted_pair_count"] == 65536
    assert train["shard_count"] == 512
    assert train["pairs_per_shard"] == 128
    assert tail["accepted_pair_count"] == 512
    assert tail["namespace_count"] == 4
    assert tail["accepted_pairs_per_namespace"] == 128
    assert config["terminal_reference"]["accepted_train_count"] == 131072
    assert config["execution"]["enabled"] is False
    assert all(value is False for value in config["authorization_boundaries"].values())


def test_terminal_namespaces_are_deterministic_disjoint_and_complete() -> None:
    config = load_terminal_131k_contract(ROOT)
    first = terminal_namespaces(config)
    second = terminal_namespaces(deepcopy(config))
    assert first == second
    assert len(first) == 5
    assert first[0].namespace_id == "train_increment"
    assert first[0].accepted_count == 65536
    assert len({item.root_seed for item in first}) == 5
    assert tuple(item.balanced_tail_stratum for item in first[1:]) == TAIL_STRATA
    assert sum(item.accepted_count for item in first[1:]) == 512


def test_terminal_generator_configs_use_unit_weight_and_numerical_filter() -> None:
    config = load_terminal_131k_contract(ROOT)
    for namespace in terminal_namespaces(config):
        generated = build_terminal_namespace_config(ROOT, config, namespace)
        context = generated["production_context"]
        assert context["proposal_mode"] == "evaluation_target_direct"
        assert context["split"] == namespace.split.value
        assert context["id_prefix"] == namespace.id_prefix
        assert generated["gw"]["source_polarization_numerical_validity"] == {
            "enabled": True,
            "minimum_frequency_hz": 20.0,
            "positive_amplitude_quantile": 0.999,
            "maximum_peak_to_quantile_ratio": 10.0,
        }


def test_terminal_identities_are_deterministic_and_all_distinct() -> None:
    config = load_terminal_131k_contract(ROOT)
    first = derive_terminal_identities(ROOT, config, "1" * 40)
    second = derive_terminal_identities(ROOT, deepcopy(config), "1" * 40)
    assert first == second
    values = {
        first.parent_run_id,
        first.train_dataset_id,
        first.development_tail_parent_id,
        first.combined_train_id,
        *first.development_tail_dataset_ids.values(),
    }
    assert len(values) == 8


def test_development_tail_snr_interval_is_half_open() -> None:
    images = (_Image(9.0), _Image(1.0))
    assert classify_balanced_tail(
        images,
        secondary_network_snr=11.999999,
        external_convergence=0.0,
        density_slope=2.0,
    ).value == "second_image_near_threshold"
    assert (
        classify_balanced_tail(
            images,
            secondary_network_snr=12.0,
            external_convergence=0.0,
            density_slope=2.0,
        )
        is None
    )


def test_terminal_release_gate_fails_closed_without_future_authorization() -> None:
    result = evaluate_release_gate(
        config_path="configs/data/phase4_terminal_131k.yaml",
        generator_commit="1" * 40,
    )
    assert result["status"] == "blocked_preexecution"
    assert result["official_identities"] is None
    assert "generator commit mismatch" in result["blockers"][0]


def test_terminal_config_hash_is_stable() -> None:
    assert configuration_hash(load_yaml(CONFIG_PATH)) == configuration_hash(
        load_yaml(CONFIG_PATH)
    )


def _terminal_result_fixture() -> dict[str, object]:
    config = load_terminal_131k_contract(ROOT)
    authorization = load_yaml(
        ROOT / str(config["future_execution_authorization_path"])
    )
    generator = str(authorization["implementation_commit"])
    identities = derive_terminal_identities(ROOT, config, generator)
    orchestration = "2" * 40
    return {
        "status": "passed",
        "parent_run_id": identities.parent_run_id,
        "train_dataset_id": identities.train_dataset_id,
        "development_tail_parent_id": identities.development_tail_parent_id,
        "development_tail_dataset_ids": dict(
            identities.development_tail_dataset_ids
        ),
        "combined_train_id": identities.combined_train_id,
        "configuration_hash": identities.configuration_hash,
        "generator_commit": generator,
        "orchestration_commit": orchestration,
        "scheduler": {
            "configured_worker_processes": 16,
            "scheduler_worker_processes": 32,
            "orchestration_commit": orchestration,
            "worker_64_authorized": False,
        },
        "new_train_accepted_count": 65536,
        "new_train_shard_count": 512,
        "development_tail_accepted_count": 512,
        "development_tail_namespace_count": 4,
        "terminal_train_accepted_count": 131072,
        "proposal_equals_evaluation": True,
        "all_importance_weights_one": True,
        "remaining_free_bytes": 100000000000,
        "train_131k_probe_authorized": False,
        "architecture_selection_authorized": False,
        "calibration_authorized": False,
        "sbc_authorized": False,
        "final_evaluation_authorized": False,
        "extension_above_131072_authorized": False,
        "gwosc_gwtc_accessed": False,
    }


def test_terminal_closeout_accepts_only_exact_worker32_result() -> None:
    config = load_terminal_131k_contract(ROOT)
    result = _terminal_result_fixture()
    observed = validate_terminal_execution_result(ROOT, config, result)
    assert observed["configuration_hash"] == configuration_hash(config)
    assert observed["development_tail_dataset_ids"] == result[
        "development_tail_dataset_ids"
    ]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("new_train_accepted_count", 65535),
        ("development_tail_accepted_count", 511),
        ("terminal_train_accepted_count", 131071),
        ("all_importance_weights_one", False),
        ("train_131k_probe_authorized", True),
        ("extension_above_131072_authorized", True),
    ],
)
def test_terminal_closeout_rejects_count_or_safety_drift(
    field: str, invalid: object
) -> None:
    config = load_terminal_131k_contract(ROOT)
    result = _terminal_result_fixture()
    result[field] = invalid
    with pytest.raises(ValueError, match="count or safety contract"):
        validate_terminal_execution_result(ROOT, config, result)


def test_worker32_scheduler_is_engineering_only_and_preserves_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_terminal_131k_contract(ROOT)
    generator = "1" * 40
    orchestration = "2" * 40
    identities = derive_terminal_identities(ROOT, config, generator)
    identity_mapping = {
        "parent_run_id": identities.parent_run_id,
        "train_dataset_id": identities.train_dataset_id,
    }
    evidence = tmp_path / "interrupted"
    evidence.mkdir()
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k.APPROVED_REMOTE_ROOT", tmp_path
    )
    authorization = {
        "authorization_status": "authorized_worker32_orchestration_restart_only",
        "frozen_execution": {
            "generator_commit": generator,
            "orchestration_commit": orchestration,
            "configuration_hash": configuration_hash(config),
            "preregistration_hash": config["preregistration"]["canonical_hash"],
            **identity_mapping,
        },
        "scheduler_contract": {
            "configured_worker_processes": 16,
            "authorized_scheduler_workers": 32,
            "maximum_scheduler_workers": 32,
        },
        "interrupted_worker16_evidence": {
            "path": str(evidence),
            "complete_shard_count": 0,
            "partial_shard_count": 16,
            "partial_evidence_reuse_authorized": False,
        },
        "authorization_boundaries": {
            "scientific_contract_change_authorized": False,
            "new_dataset_identity_authorized": False,
            "worker_64_authorized": False,
            "training_authorized": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    }
    selection = _validate_scheduler_authorization(
        authorization=authorization,
        config=config,
        generator_commit=generator,
        orchestration_commit=orchestration,
        identities=identity_mapping,
        requested_workers=32,
    )
    assert selection["scheduler_worker_processes"] == 32
    assert selection["worker_64_authorized"] is False
    assert configuration_hash(config) == identities.configuration_hash


def test_worker64_scheduler_is_rejected_before_authorization_lookup() -> None:
    from scripts.phase4.run_terminal_131k import _resolve_scheduler

    config = load_terminal_131k_contract(ROOT)
    identities = derive_terminal_identities(ROOT, config, "1" * 40)
    with pytest.raises(PermissionError, match="only the reviewed 32-worker"):
        _resolve_scheduler(
            config=config,
            generator_commit="1" * 40,
            identities={
                "parent_run_id": identities.parent_run_id,
                "train_dataset_id": identities.train_dataset_id,
            },
            requested_workers=64,
            orchestration_commit="2" * 40,
            scheduler_authorization_path=None,
        )


def test_terminal_execute_atomically_publishes_both_pools_and_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = deepcopy(load_terminal_131k_contract(ROOT))
    paths = {
        "train_staging_root": tmp_path / "train-staging",
        "train_publication_root": tmp_path / "train-published",
        "tail_staging_root": tmp_path / "tail-staging",
        "tail_publication_root": tmp_path / "tail-published",
        "combined_staging_root": tmp_path / "combined-staging",
        "combined_publication_root": tmp_path / "combined-published",
        "manifest_root": tmp_path / "manifests",
        "log_root": tmp_path / "logs",
    }
    config["paths"] = {key: str(value) for key, value in paths.items()}
    config["resource_gates"]["minimum_post_peak_free_bytes"] = 0
    identities = derive_terminal_identities(ROOT, config, "1" * 40)
    identity_mapping = {
        "parent_run_id": identities.parent_run_id,
        "train_dataset_id": identities.train_dataset_id,
        "development_tail_parent_id": identities.development_tail_parent_id,
        "development_tail_dataset_ids": dict(identities.development_tail_dataset_ids),
        "combined_train_id": identities.combined_train_id,
        "configuration_hash": identities.configuration_hash,
    }
    certificate = {
        "status": "ready_for_official_execution",
        "generator_commit": "1" * 40,
        "configuration_hash": configuration_hash(config),
        "official_identities": identity_mapping,
        "scheduler": {
            "source": "frozen_configuration",
            "configured_worker_processes": 16,
            "scheduler_worker_processes": 16,
            "orchestration_commit": "1" * 40,
            "authorization_path": None,
            "worker_64_authorized": False,
        },
    }
    certificate_path = tmp_path / "release.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k.load_terminal_131k_contract",
        lambda root, path: deepcopy(config),
    )
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k._authorization", lambda loaded: {}
    )
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k.verify_generator_commit",
        lambda *args, **kwargs: None,
    )
    corrected = SimpleNamespace(
        corrected_combined_train_manifest_sha256=(
            config["corrected_65k_reference"][
                "corrected_combined_train_manifest_sha256"
            ]
        )
    )
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k._resolve_corrected", lambda loaded: corrected
    )
    identifiers = {
        "pair": {"pair"},
        "source": {"source"},
        "lens": {"lens"},
        "system": {"system"},
        "noise": {f"noise-{index}" for index in range(6)},
        "attempt_system": {"attempt"},
    }

    def fake_publish(**kwargs: object) -> tuple[dict[str, object], dict[str, set[str]]]:
        namespace = kwargs["namespace"]
        assert hasattr(namespace, "namespace_id")
        return {"status": "passed"}, deepcopy(identifiers)

    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k._publish_namespace", fake_publish
    )
    cross = {
        "all_train_validation_tail_groups_disjoint": True,
        "corrected_65k_membership_preserved": True,
    }
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k._cross_validation",
        lambda *args, **kwargs: cross,
    )
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k.tree_checksum",
        lambda path: ("a" * 64, 1000),
    )
    monkeypatch.setattr(
        "scripts.phase4.run_terminal_131k.shutil.disk_usage",
        lambda path: __import__("shutil")._ntuple_diskusage(10000, 1000, 9000),
    )
    output = tmp_path / "result.json"
    result = execute(
        config_path="unused.yaml",
        generator_commit="1" * 40,
        certificate_path=certificate_path,
        output=output,
    )
    assert result["status"] == "passed"
    assert result["new_train_accepted_count"] == 65536
    assert result["development_tail_accepted_count"] == 512
    assert result["terminal_train_accepted_count"] == 131072
    assert result["train_131k_probe_authorized"] is False
    assert result["extension_above_131072_authorized"] is False
    assert (
        paths["combined_publication_root"]
        / identities.combined_train_id
        / "dataset_manifest.json"
    ).is_file()
