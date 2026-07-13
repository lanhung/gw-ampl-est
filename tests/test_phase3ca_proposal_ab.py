from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.ab_qualification import (
    arm_config,
    bootstrap_throughput,
    load_and_verify_contract,
    postselection_diagnostics,
    validate_distribution_metadata_finite,
    validate_first_block_health,
)
from gwlens_mm.production.proposal_adapter import sample_production_proposal
from gwlens_mm.production.storage import ShardWriter, sha256_file
from gwlens_mm.schema import DistributionMetadata, V2Record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/phase3ca_proposal_v3_ab.yaml"
RETRY_CONFIG = ROOT / "configs/data/phase3ca1_proposal_v3_ab_retry.yaml"
EXAMPLE = ROOT / "examples/v2_metadata_example.json"


def _health_fixture(tmp_path: Path, *, nonfinite: bool = False) -> tuple[Path, str]:
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    dataset_id = "phase3ca1-health-regression"
    stage = tmp_path / "stage"
    (stage / "shards").mkdir(parents=True)
    writer = ShardWriter(stage / "shards", 0, expected_pairs=32, sample_count=4096)
    template = json.loads(EXAMPLE.read_text())
    arrays = np.zeros((2, 3, 4096), dtype=np.float32)
    for index in range(32):
        data = deepcopy(template)
        data["pair"].update(
            pair_id=f"health-pair-{index:06d}",
            source_id=f"health-source-{index:06d}",
            lens_id=f"health-lens-{index:06d}",
            physical_system_id=f"health-system-{index:06d}",
            split="generator_qualification",
            dataset_version=dataset_id,
        )
        for reference in data["provenance"]["detector_noise_references"]:
            if reference["segment_id"] is not None:
                reference["segment_id"] += f"-{index:06d}"
        record = V2Record.from_dict(data)
        assert (
            V2Record.from_json(record.to_json())
            .provenance.distribution.evaluation_prior_log_probability
            == -12.0
        )
        writer.append(
            record,
            arrays,
            arrays,
            arrays,
            attempt_id=index,
            partition_metadata={"em_cell": "health_fixture"},
        )
    writer.finalize()
    if nonfinite:
        import pandas as pd

        parquet = stage / "shards/shard-00000/records.parquet"
        frame = pd.read_parquet(parquet)
        data = json.loads(frame.loc[0, "record_json"])
        data["provenance"]["distribution"]["proposal_log_probability"] = float("nan")
        frame.loc[0, "record_json"] = json.dumps(data, allow_nan=True)
        frame.to_parquet(parquet, index=False)
        manifest_path = stage / "shards/shard-00000/shard_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            if artifact["relative_path"] == "records.parquet":
                artifact["sha256"] = sha256_file(parquet)
                artifact["bytes"] = parquet.stat().st_size
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        marker = stage / "shards/shard-00000/COMPLETE.json"
        marker.write_text(
            json.dumps({"shard_manifest_sha256": sha256_file(manifest_path)}) + "\n"
        )
    return stage, dataset_id


def test_phase3ca_contract_is_bounded_and_non_scientific() -> None:
    config = load_yaml(CONFIG)
    assert config["arm_count"] == 2
    assert config["accepted_pairs_per_arm"] == 512
    assert config["total_accepted_pairs"] == 1024
    assert config["blocks_per_arm"] * config["accepted_pairs_per_block"] == 512
    for key in (
        "scientific_use_authorized",
        "training_use_authorized",
        "calibration_use_authorized",
        "test_use_authorized",
    ):
        assert config["use_policy"][key] is False
    assert config["use_policy"]["permanent_exclusion_from_all_scientific_splits"] is True
    authorization = load_yaml(ROOT / config["authorization"]["path"])
    assert authorization["authorization"]["proposal_v3_ab_qualification_authorized"] is True
    for key in (
        "scientific_data_generation_authorized",
        "model_training_authorized",
        "calibration_authorized",
        "gwosc_gwtc_access_authorized",
        "stage_a_authorized",
    ):
        assert authorization["authorization"][key] is False


def test_contract_derives_distinct_arm_identities() -> None:
    marker = ROOT / "SYNCED_COMMIT"
    head = (
        marker.read_text().strip()
        if marker.is_file()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    config, _, _, identity = load_and_verify_contract(ROOT, head)
    assert identity.control_dataset_id != identity.candidate_dataset_id
    control = arm_config(ROOT, config, "rc5_control")
    candidate = arm_config(ROOT, config, "proposal_v3_candidate")
    assert control["root_seed"] != candidate["root_seed"]
    assert control["engineering_ab"]["id_prefix"] != candidate["engineering_ab"]["id_prefix"]
    assert control["accepted_pair_count"] == candidate["accepted_pair_count"] == 512


def test_retry_contract_uses_new_seeds_namespaces_and_identities() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config, _, _, identity = load_and_verify_contract(
        ROOT, head, str(RETRY_CONFIG.relative_to(ROOT))
    )
    failed = config["failed_run_exclusion"]
    assert config["arms"]["rc5_control"]["root_seed"] == 2026071213
    assert config["arms"]["proposal_v3_candidate"]["root_seed"] == 2026071214
    assert config["bootstrap"]["seed"] == 2026071212
    assert identity.parent_run_id != failed["parent_run_id"]
    assert identity.control_dataset_id != failed["control_dataset_id"]
    assert identity.candidate_dataset_id != failed["candidate_dataset_id"]
    assert config["accepted_pairs_per_arm"] == 512
    assert config["total_accepted_pairs"] == 1024
    assert failed["old_pair_count_toward_retry"] == 0


def test_production_adapter_records_exact_finite_v3_provenance() -> None:
    proposal = load_yaml(ROOT / "configs/proposals/proposal_v3_target_anchored_mixture.yaml")
    result = sample_production_proposal(
        np.random.default_rng(2026071211),
        mode="proposal_v3_candidate",
        proposal_config=proposal,
    )
    assert result.component in {"rc5_broad", "evaluation_target", "central"}
    assert set(result.component_log_densities) == {
        "rc5_broad",
        "evaluation_target",
        "central",
    }
    values = (
        *result.component_log_densities.values(),
        result.population.proposal_log_probability,
        result.population.evaluation_log_probability,
        result.population.importance_weight,
    )
    assert np.all(np.isfinite(values))


def test_paired_bootstrap_uses_all_matched_blocks() -> None:
    config = load_yaml(CONFIG)
    rows = []
    for index in range(16):
        rows.extend(
            (
                {"arm": "rc5_control", "block_index": index, "active_wall_seconds": 96.0},
                {
                    "arm": "proposal_v3_candidate",
                    "block_index": index,
                    "active_wall_seconds": 32.0,
                },
            )
        )
    result = bootstrap_throughput(rows, config)
    assert result["point_estimate"] == 3.0
    assert result["lower_95"] == pytest.approx(3.0)
    assert result["passed"] is True


def test_postselection_gate_fails_missing_tail_support() -> None:
    config = load_yaml(CONFIG)
    candidate = {
        "weights": [1.0] * 512,
        "families": ["sie_external_shear", "epl_external_shear"] * 256,
        "em_cells": [f"cell_{index % 8}" for index in range(512)],
        "accepted_components": ["central"] * 512,
        "source_radii": [0.1] * 512,
        "einstein_radii": [1.0] * 512,
        "lens_redshifts": [0.5] * 512,
        "multiplicity_counts": {2: 500, 4: 12},
    }
    result = postselection_diagnostics(candidate, config)
    assert result["status"] == "failed"
    assert result["checks"]["tail_support"] is False


def test_real_first_block_health_path_accepts_alpha3_round_trip(tmp_path: Path) -> None:
    stage, dataset_id = _health_fixture(tmp_path)
    result = validate_first_block_health(stage, dataset_id)
    assert result["status"] == "passed"
    assert result["accepted_pair_count"] == 32
    assert result["throughput_inspected"] is False


def test_real_first_block_health_path_rejects_nonfinite(tmp_path: Path) -> None:
    stage, dataset_id = _health_fixture(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match="finite|proposal log probability"):
        validate_first_block_health(stage, dataset_id)


def test_typed_distribution_helper_uses_alpha3_field() -> None:
    valid = DistributionMetadata(-12.0, -11.0, np.e, True, False, None)
    validate_distribution_metadata_finite(valid)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_distribution_metadata_finite(
            DistributionMetadata(float("nan"), -11.0, 1.0, True, False, None)
        )
    maintained = (
        ROOT / "src/gwlens_mm/production/ab_qualification.py"
    ).read_text() + (ROOT / "scripts/phase3ca/run_proposal_ab.py").read_text()
    assert ".evaluation_log_probability" not in maintained
