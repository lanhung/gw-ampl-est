from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.production.ab_qualification import (
    arm_config,
    bootstrap_throughput,
    load_and_verify_contract,
    postselection_diagnostics,
)
from gwlens_mm.production.proposal_adapter import sample_production_proposal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/phase3ca_proposal_v3_ab.yaml"


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
