from pathlib import Path

import pytest
import yaml

from gwlens_mm.production.authorization import load_qualification_authorization
from gwlens_mm.provenance import ArtifactChecksum, DatasetManifest

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "configs/execution/phase3a_qualification_authorization.yaml"
AUTH_COMMIT = "46df8bb23ad99ce153e3fc9dc3b4fe46962f41f9"


def test_phase3a_authorization_matches_frozen_preregistration() -> None:
    authorization = load_qualification_authorization(
        AUTH, repository_root=ROOT, authorizing_git_commit=AUTH_COMMIT
    )
    assert authorization.preregistration_hash == (
        "16a75327df5aacafa1fb4459e19429cc08d3350cd3056986356ef3c57864c1e8"
    )
    assert authorization.accepted_pair_count == 4096
    assert authorization.shard_pair_count == 128
    assert authorization.shard_count == 32


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
