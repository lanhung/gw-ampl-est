from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from scripts.phase4.run_stage_b import (
    _build_namespace_config,
    _cross_component_validation,
    _identities,
    _load_contract,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/phase4_direct_target_stage_b.yaml"


def test_stage_b_is_the_single_exact_nested_increment() -> None:
    config = _load_contract()
    stage = config["stage_b"]
    cumulative = config["cumulative_train_contract"]
    assert stage["accepted_pair_count"] == 32768
    assert stage["shard_count"] == 256
    assert stage["pairs_per_shard"] == 128
    assert stage["shard_count"] * stage["pairs_per_shard"] == 32768
    assert cumulative["accepted_physical_system_count"] == 65536
    assert cumulative["validation_count_unchanged"] == 6144
    assert cumulative["no_new_validation_materialization"] is True
    assert cumulative["calibration_sbc_final_evaluation_materialized"] is False


def test_stage_b_requires_the_committed_continuation_evidence() -> None:
    config = load_yaml(CONFIG_PATH)
    evidence = ROOT / config["learning_curve_evidence"]["path"]
    decision = json.loads(evidence.read_text())
    assert decision["decision"] == "continue_to_train_65k"
    assert decision["final_evaluation_accessed"] is False
    assert decision["calibration_accessed"] is False


def test_stage_b_namespace_is_direct_target_and_independent() -> None:
    config = _load_contract()
    namespace = _build_namespace_config(config)
    context = namespace["production_context"]
    assert namespace["root_seed"] == 2026071403
    assert namespace["accepted_pair_count"] == 32768
    assert context["proposal_mode"] == "evaluation_target_direct"
    assert context["proposal_distribution_id"] == context["evaluation_distribution_id"]
    assert context["id_prefix"] == "phase4-stage-b-train"
    assert context["split"] == "train"
    assert config["stage_a_reference"]["train_dataset_id"] != context["id_prefix"]


def test_stage_b_all_later_execution_boundaries_are_closed() -> None:
    config = _load_contract()
    assert all(value is False for value in config["authorization_boundaries"].values())
    assert config["execution"]["scientific_execution_enabled"] is False


def test_stage_b_identities_are_deterministic_and_distinct() -> None:
    config = _load_contract()
    commit = "1" * 40
    first = _identities(config, commit)
    second = _identities(deepcopy(config), commit)
    assert first == second
    assert first["train_dataset_id"] != config["stage_a_reference"]["train_dataset_id"]
    assert first["parent_run_id"] != config["stage_a_reference"]["parent_run_id"]
    assert first["combined_train_id"] not in {
        first["parent_run_id"],
        first["train_dataset_id"],
    }


def test_stage_b_resource_projection_is_conservative_and_bounded() -> None:
    config = _load_contract()
    resources = config["resource_gates"]
    observed_rate = (
        resources["stage_a_measured_accepted_count"]
        / resources["stage_a_measured_active_hours"]
    )
    projected_hours = 32768 / observed_rate
    assert resources["projected_stage_b_active_hours"] == pytest.approx(
        projected_hours, rel=1e-5
    )
    assert resources["maximum_active_hours"] >= projected_hours
    assert resources["maximum_stage_b_output_bytes"] > resources[
        "projected_stage_b_publication_bytes"
    ]
    assert resources["minimum_prelaunch_free_bytes"] == (
        resources["minimum_post_peak_free_bytes"]
        + resources["projected_stage_b_peak_bytes"]
    )


def test_cross_component_validation_rejects_any_stage_a_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_contract()
    empty = {key: set() for key in ("pair", "source", "lens", "system", "noise")}
    stage_a = deepcopy(empty)
    stage_a["pair"].add("duplicate")

    def fake_ids(path: Path, count: int) -> dict[str, set[str]]:
        del path, count
        return deepcopy(stage_a)

    monkeypatch.setattr("scripts.phase4.run_stage_b._published_group_ids", fake_ids)
    candidate = deepcopy(empty)
    candidate["pair"].add("duplicate")
    with pytest.raises(ValueError, match="pair leakage"):
        _cross_component_validation(config, candidate)


def test_stage_b_config_hash_is_stable_under_reload() -> None:
    first = load_yaml(CONFIG_PATH)
    second = load_yaml(CONFIG_PATH)
    assert configuration_hash(first) == configuration_hash(second)


def test_stage_b_execute_atomically_publishes_extension_and_combined_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_contract()
    config = deepcopy(config)
    paths = {
        "staging_root": tmp_path / "stage-b-staging",
        "publication_root": tmp_path / "stage-b-published",
        "combined_staging_root": tmp_path / "combined-staging",
        "combined_publication_root": tmp_path / "combined-published",
        "manifest_root": tmp_path / "manifests",
        "log_root": tmp_path / "logs",
    }
    config["paths"] = {key: str(value) for key, value in paths.items()}
    config["resource_gates"]["minimum_post_peak_free_bytes"] = 0
    identities = _identities(config, "1" * 40)
    certificate = {
        "status": "ready_for_official_execution",
        "orchestration_commit": "1" * 40,
        "configuration_hash": configuration_hash(config),
        "official_identities": identities,
    }
    certificate_path = tmp_path / "release.json"
    certificate_path.write_text(json.dumps(certificate))
    output = tmp_path / "result.json"

    monkeypatch.setattr(
        "scripts.phase4.run_stage_b._load_contract", lambda path: deepcopy(config)
    )
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b._load_authorization",
        lambda loaded: {"implementation_commit": "0" * 40},
    )
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b._git_clean_commit", lambda root, commit: None
    )
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b._verify_release_lineage",
        lambda loaded, authorization, commit: None,
    )

    def fake_generate(**kwargs: object) -> None:
        stage = Path(str(kwargs["stage"]))
        (stage / "shards" / "shard-00000").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("scripts.phase4.run_stage_b._generate_pending", fake_generate)
    identifiers = {
        "pair": {f"pair-{index}" for index in range(32768)},
        "source": {f"source-{index}" for index in range(32768)},
        "lens": {f"lens-{index}" for index in range(32768)},
        "system": {f"system-{index}" for index in range(32768)},
        "noise": {f"noise-{index}" for index in range(32768 * 6)},
        "attempt_system": set(),
    }
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b.validate_stage_a_namespace",
        lambda *args, **kwargs: ({"status": "passed"}, identifiers),
    )
    combined = {
        "pair_count": 65536,
        "source_count": 65536,
        "lens_count": 65536,
        "system_count": 65536,
        "noise_count": 65536 * 6,
        "stage_a_stage_b_group_disjoint": True,
        "stage_b_validation_group_disjoint": True,
    }
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b._cross_component_validation",
        lambda loaded, values: combined,
    )
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b.tree_checksum",
        lambda path: ("a" * 64, 1000),
    )
    monkeypatch.setattr(
        "scripts.phase4.run_stage_b.shutil.disk_usage",
        lambda path: __import__("shutil")._ntuple_diskusage(10_000, 1000, 9000),
    )

    result = execute(
        config_path="unused.yaml",
        orchestration_commit="1" * 40,
        certificate_path=certificate_path,
        output=output,
    )
    assert result["status"] == "passed"
    assert result["accepted_pair_count"] == 32768
    assert result["cumulative_train_accepted_pair_count"] == 65536
    assert result["train_65k_optimizer_authorized"] is False
    assert result["final_evaluation_authorized"] is False
    assert not (paths["staging_root"] / identities["parent_run_id"]).exists()
    assert (paths["publication_root"] / identities["parent_run_id"]).is_dir()
    combined_manifest = (
        paths["combined_publication_root"]
        / identities["combined_train_id"]
        / "dataset_manifest.json"
    )
    assert combined_manifest.is_file()
    assert json.loads(combined_manifest.read_text())["training_authorized"] is False
