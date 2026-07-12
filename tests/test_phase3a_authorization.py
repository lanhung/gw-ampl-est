from pathlib import Path

import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.production.authorization import load_qualification_authorization
from gwlens_mm.provenance import ArtifactChecksum, DatasetManifest

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "configs/execution/phase3a_qualification_authorization.yaml"
AUTH_COMMIT = "bba0cdd6a750ff367674a85b8722432e613586d8"
CONFIG = ROOT / "configs/data/phase3a_qualification.yaml"


def test_phase3a_authorization_matches_frozen_preregistration() -> None:
    authorization = load_qualification_authorization(
        AUTH, repository_root=ROOT, authorizing_git_commit=AUTH_COMMIT
    )
    assert authorization.preregistration_hash == (
        "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"
    )
    assert authorization.accepted_pair_count == 4096
    assert authorization.shard_pair_count == 128
    assert authorization.shard_count == 32


def test_phase3a_execution_config_matches_rc5_and_bounded_parallel_plan() -> None:
    config = load_yaml(CONFIG)
    assert config["preregistration"] == {
        "path": "configs/statistics/phase2_preregistration.yaml",
        "version": "1.0.0-rc.5",
        "canonical_hash": (
            "4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb"
        ),
        "base_main_commit": "80167ea690914bb18be1fd1994b4dc626490e146",
    }
    assert config["authorization"]["authorizing_git_commit"] == AUTH_COMMIT
    assert config["accepted_pair_count"] == 4096
    assert config["shard_count"] == 32
    assert config["pairs_per_shard"] == 128
    assert config["microbenchmark"]["accepted_pair_count"] == 32
    assert config["microbenchmark"]["worker_processes"] == 8
    assert config["execution"]["qualification_worker_processes"] == 16
    assert config["execution"]["attempt_id_stride"] == 32
    assert config["interruption_test"]["stop_after_complete_shards"] == 3
    assert config["resource_gates"]["maximum_output_bytes"] == 10_000_000_000
    assert config["resource_gates"]["maximum_projected_walltime_hours"] == 24


def test_phase3a_authorization_fails_closed_on_training(tmp_path: Path) -> None:
    data = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    data["authorization"]["model_training_authorized"] = True
    path = tmp_path / "authorization.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="model_training_authorized=false"):
        load_qualification_authorization(
            path, repository_root=ROOT, authorizing_git_commit=AUTH_COMMIT
        )


def test_qualification_manifest_denies_every_downstream_use() -> None:
    manifest = DatasetManifest(
        dataset_id="qualification-test",
        schema_version="2.0.0-alpha.3",
        generator_git_commit="a" * 40,
        configuration_hash="b" * 64,
        root_seed=1,
        planned_pair_count=1,
        accepted_pair_count=1,
        attempted_pair_count=1,
        pair_ids=("pair",),
        source_ids=("source",),
        lens_ids=("lens",),
        noise_segment_ids=("noise",),
        artifacts=(ArtifactChecksum("manifest.json", "c" * 64, 1, "complete"),),
        generation_status="complete",
        dataset_purpose="generator_qualification",
        scientific_use_authorized=False,
        authorizing_git_commit="d" * 40,
    )
    manifest.validate()
    with pytest.raises(ValueError, match="downstream use"):
        DatasetManifest(
            **{**manifest.to_dict(), "training_use_authorized": True}
        ).validate()
