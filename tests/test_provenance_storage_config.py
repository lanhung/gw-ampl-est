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


def test_smoke_configuration_validates_but_is_not_authorized():
    config = load_yaml(ROOT / "configs/data/v2_smoke.yaml")
    validate_smoke_configuration(config)
    assert config["execution_authorized"] is False
    assert sum(config["accepted_pairs"].values()) == 48
    assert config["schema_version"] == SCHEMA_VERSION == "2.0.0-alpha.2"
