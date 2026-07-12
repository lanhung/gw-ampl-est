import json
from pathlib import Path

import pytest

from gwlens_mm.config import load_yaml, validate_smoke_configuration
from gwlens_mm.provenance import (
    ArtifactChecksum,
    DatasetManifest,
    configuration_hash,
    dataset_id,
    derive_seed,
    reject_duplicate_dataset_ids,
)
from gwlens_mm.schema import SCHEMA_VERSION
from gwlens_mm.storage import strain_storage_estimate

ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_and_seed_derivation_are_deterministic():
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})
    seed = derive_seed(1234, "detector_noise", "pair-1", "H1")
    assert seed == derive_seed(1234, "detector_noise", "pair-1", "H1")
    assert seed != derive_seed(1234, "detector_noise", "pair-1", "L1")


def example_manifest(identifier="gwlens-v2-example"):
    return DatasetManifest(
        dataset_id=identifier,
        schema_version="2.0.0-alpha.2",
        generator_git_commit="a" * 40,
        configuration_hash="b" * 64,
        root_seed=1,
        planned_pair_count=1,
        accepted_pair_count=1,
        attempted_pair_count=2,
        pair_ids=("pair-1",),
        source_ids=("source-1",),
        lens_ids=("lens-1",),
        noise_segment_ids=("noise-1", "noise-2"),
        artifacts=(ArtifactChecksum("arrays/strain.zarr", None, None),),
        generation_status="planned",
        dataset_purpose="engineering_smoke",
        scientific_use_authorized=False,
        authorizing_git_commit="c" * 40,
    )


def test_manifest_round_trip_and_duplicate_detection():
    manifest = example_manifest()
    manifest.validate()
    assert DatasetManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(ValueError, match="duplicate dataset IDs"):
        reject_duplicate_dataset_ids([manifest, example_manifest()])


def test_dataset_id_is_deterministic():
    identifier = dataset_id("2.0.0-alpha.2", "a" * 40, "b" * 64, 10)
    assert identifier == dataset_id("2.0.0-alpha.2", "a" * 40, "b" * 64, 10)


def test_complete_manifest_requires_complete_artifacts_and_target_count():
    pending_artifact = example_manifest()
    data = pending_artifact.to_dict()
    data["generation_status"] = "complete"
    with pytest.raises(ValueError, match="artifact"):
        DatasetManifest.from_json(json.dumps(data))

    data["artifacts"] = [
        {
            "relative_path": "arrays/strain.zarr",
            "sha256": "c" * 64,
            "bytes": 123,
            "status": "complete",
        }
    ]
    data["planned_pair_count"] = 2
    with pytest.raises(ValueError, match="accepted count"):
        DatasetManifest.from_json(json.dumps(data))

    data["planned_pair_count"] = 1
    complete = DatasetManifest.from_json(json.dumps(data))
    assert complete.generation_status == "complete"


@pytest.mark.parametrize(
    ("pairs", "expected_bytes"),
    [
        (48, 48 * 2 * 3 * 4096 * 4 * 3),
        (10_000, 10_000 * 2 * 3 * 4096 * 4 * 3),
        (100_000, 100_000 * 2 * 3 * 4096 * 4 * 3),
    ],
)
def test_storage_size_calculation(pairs, expected_bytes):
    assert strain_storage_estimate(pairs).bytes == expected_bytes


def test_phase1b_smoke_configuration_is_explicitly_authorized():
    config = load_yaml(ROOT / "configs/data/v2_smoke.yaml")
    validate_smoke_configuration(config, expected_execution_authorized=True)
    assert config["execution_authorized"] is True
    assert sum(config["accepted_pairs"].values()) == 48
    assert config["schema_version"] == SCHEMA_VERSION == "2.0.0-alpha.2"


def test_engineering_manifest_rejects_scientific_authorization():
    data = example_manifest().to_dict()
    data["scientific_use_authorized"] = True
    with pytest.raises(ValueError, match="cannot be authorized"):
        DatasetManifest.from_json(json.dumps(data))


def phase2_config():
    return load_yaml(ROOT / "configs/statistics/phase2_preregistration.yaml")


def test_phase2_preregistration_is_fail_closed_and_hash_frozen():
    config = phase2_config()
    assert config["status"] == "awaiting_human_review"
    for field in (
        "execution_enabled",
        "scientific_data_generation_authorized",
        "training_authorized",
        "gwtc_gwosc_download_authorized",
        "engineering_smoke_scientific_use_authorized",
    ):
        assert config[field] is False
    assert config["splits"]["real_noise_test"] == 0
    assert configuration_hash(config) == (
        "4ae2899a054342fcc1100554f72cd826969afb7030885edbcaacb251efd603aa"
    )


def test_phase2_counts_and_storage_arithmetic_are_exact():
    config = phase2_config()
    counts = {
        key: value
        for key, value in config["splits"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    storage = config["storage_plan"]
    assert sum(counts.values()) == storage["planned_pairs_including_qualification"]
    assert (
        storage["bytes_per_pair_uncompressed"]
        == storage["products_per_pair"] * 2 * 3 * 16384 * 4
    )
    assert (
        storage["raw_array_bytes"]
        == storage["planned_pairs_including_qualification"]
        * storage["bytes_per_pair_uncompressed"]
    )
    assert storage["qualification_raw_bytes"] < 10_000_000_000
    assert storage["projected_remaining_bytes"] > 100_000_000_000


def test_phase2_evaluation_support_is_contained_in_proposal():
    config = phase2_config()
    proposal = config["proposal_distribution"]
    evaluation = config["evaluation_distribution"]
    for field in (
        "lens_redshift",
        "source_redshift",
        "einstein_radius_arcsec",
        "axis_ratio",
        "shear_amplitude",
        "epl_density_slope",
        "external_convergence",
    ):
        assert proposal[field]["minimum"] <= evaluation[field]["minimum"]
        assert proposal[field]["maximum"] >= evaluation[field]["maximum"]
