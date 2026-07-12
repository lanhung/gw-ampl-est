from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.physics.quantities import LensFamily
from gwlens_mm.proposals.ab_skeleton import build_dry_run_plan
from gwlens_mm.proposals.exact_mixture import (
    COMPONENT_ORDER,
    Component,
    ProposalSpecification,
    evaluate_latent_preflight,
    log_component_density,
    log_evaluation_density,
    log_importance_weight,
    log_mixture_density,
    low_z_log_density,
    sample_latent,
    truncated_normal_log_density,
)
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/proposals/proposal_v2_exact_mixture.yaml"
AUTHORIZATION_PATH = ROOT / "configs/execution/phase3c0_proposal_v2_design_authorization.yaml"
RESULT_PATH = ROOT / "results/phase3c0/proposal_v2_latent_validation.json"
EXPECTED_CONFIG_HASH = "e4e249da3f177202960e8a6f6c0c347a25aa572abb818b6d0e172469a75e45b5"
EXPECTED_REPLAY_HASH = "02c1d1dae179703478b7c2250385912cf5c09fdb262b405f75440c0bac46703d"


@pytest.fixture(scope="module")
def config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture(scope="module")
def specification(config):
    return ProposalSpecification.from_mapping(config)


def test_mixture_and_authorization_are_fail_closed(config, specification) -> None:
    assert config["proposal_version"] == "proposal-v2-exact-mixture-1.0.0-rc.1"
    assert specification.component_weights == (0.2, 0.6, 0.1, 0.1)
    assert math.isclose(sum(specification.component_weights), 1.0, abs_tol=1.0e-15)
    assert configuration_hash(config) == EXPECTED_CONFIG_HASH
    assert config["authorization"]["waveform_pair_generation_authorized"] is False
    assert config["authorization"]["proposal_ab_qualification_authorized"] is False
    authorization = load_yaml(AUTHORIZATION_PATH)["authorization"]
    for key, value in authorization.items():
        if key.endswith("_authorized") and key not in {
            "proposal_design_authorized",
            "proposal_sampler_implementation_authorized",
            "proposal_log_density_implementation_authorized",
            "latent_only_normalization_tests_authorized",
            "ab_runner_skeleton_authorized",
        }:
            assert value is False, key


@pytest.mark.parametrize("sigma", [0.8, 1.5])
def test_truncated_normal_integrates_to_one(sigma: float) -> None:
    grid = np.linspace(-2.5, 2.5, 200_001, endpoint=False)
    density = np.exp(
        [truncated_normal_log_density(float(x), 0.0, sigma, -2.5, 2.5) for x in grid]
    )
    integral = float(np.trapz(density, grid))
    assert math.isclose(integral, 1.0, abs_tol=2.0e-5)
    assert math.isclose(integral * integral, 1.0, abs_tol=4.0e-5)
    assert math.isfinite(truncated_normal_log_density(-2.5, 0.0, sigma, -2.5, 2.5))
    assert truncated_normal_log_density(2.5, 0.0, sigma, -2.5, 2.5) == -math.inf


@pytest.mark.parametrize("z_lens", [0.1, 0.4, 0.9, 1.0])
def test_low_z_conditional_density_integrates_to_one(z_lens: float) -> None:
    lower = max(0.5, z_lens + 0.1)
    grid = np.linspace(lower, 3.0, 200_001, endpoint=False)
    density = np.exp([low_z_log_density(float(z), z_lens) for z in grid])
    assert math.isclose(float(np.trapz(density, grid)), 1.0, abs_tol=2.0e-5)
    assert math.isfinite(low_z_log_density(lower, z_lens))
    assert low_z_log_density(3.0, z_lens) == -math.inf


def test_sampler_is_deterministic_and_never_leaves_support(specification) -> None:
    first_rng = np.random.default_rng(17)
    second_rng = np.random.default_rng(17)
    first = [sample_latent(first_rng, specification) for _ in range(2000)]
    second = [sample_latent(second_rng, specification) for _ in range(2000)]
    assert first == second
    for draw in first:
        assert -2.5 <= draw.source_u_x < 2.5
        assert -2.5 <= draw.source_u_y < 2.5
        assert 0.1 <= draw.z_lens < 1.0
        assert max(0.5, draw.z_lens + 0.1) <= draw.z_source < 3.0
        assert draw.mass_2_source >= 10.0
        assert math.isfinite(log_mixture_density(draw, specification))
        assert math.isfinite(log_evaluation_density(draw, specification))
        assert math.isfinite(log_importance_weight(draw, specification))


def test_safety_component_makes_mixture_positive_on_rc5_support(specification) -> None:
    rng = np.random.default_rng(21)
    for _ in range(1000):
        draw = sample_latent(rng, specification)
        rc5_log = log_component_density(draw, Component.RC5_BROAD, specification)
        mixture_log = log_mixture_density(draw, specification)
        assert math.isfinite(rc5_log)
        assert mixture_log >= math.log(0.2) + rc5_log - 1.0e-12


def test_logsumexp_matches_direct_component_sum(specification) -> None:
    draw = sample_latent(np.random.default_rng(99), specification)
    terms = np.array(
        [
            math.log(weight) + log_component_density(draw, component, specification)
            for component, weight in zip(COMPONENT_ORDER, specification.component_weights)
        ]
    )
    scale = float(np.max(terms))
    direct = scale + math.log(float(np.sum(np.exp(terms - scale))))
    assert math.isclose(log_mixture_density(draw, specification), direct, abs_tol=1.0e-12)


def test_latent_preflight_code_never_calls_pair_generation(specification) -> None:
    small = replace(specification, preflight_draw_count=2000, preflight_seed=1234)
    result = evaluate_latent_preflight(small)
    assert result["draw_count"] == 2000
    assert result["waveform_pair_count"] == 0
    assert result["accepted_pair_generator_called"] is False
    assert result["proposal_ab_qualification_run"] is False
    assert result["replay_byte_identical"] is True


def test_ab_runner_is_dry_run_only_and_counted_exactly(config) -> None:
    plan = build_dry_run_plan(config, "phase3c0-dry-run-fixture")
    assert plan.dry_run_only is True
    assert plan.control_dataset_id != plan.candidate_dataset_id
    assert plan.control_manifest != plan.candidate_manifest
    assert plan.control_checksum_manifest != plan.candidate_checksum_manifest
    assert plan.parent_comparison_manifest.endswith("comparison.json")
    assert len(plan.blocks) == 16
    assert sum(block.accepted_pairs_per_arm for block in plan.blocks) == 512
    assert plan.maximum_accepted_pairs_per_arm == 512
    assert plan.maximum_accepted_pairs_total == 1024
    assert plan.maximum_attempts_per_arm == 1_000_000
    assert plan.maximum_active_hours_per_arm == 6.0
    assert plan.blocks[0].first_arm == "rc5_control"
    assert plan.blocks[1].first_arm == "proposal_v2_candidate"


def test_ab_runner_rejects_execution_enabled_or_inconsistent_config(config) -> None:
    enabled = {
        **config,
        "future_ab_skeleton": {
            **config["future_ab_skeleton"],
            "dry_run_only": False,
        },
    }
    with pytest.raises(ValueError, match="dry-run-only"):
        build_dry_run_plan(enabled, "invalid-enabled")
    inconsistent = {
        **config,
        "future_ab_skeleton": {
            **config["future_ab_skeleton"],
            "total_accepted_pairs": 512,
        },
    }
    with pytest.raises(ValueError, match="arithmetic"):
        build_dry_run_plan(inconsistent, "invalid-count")


def test_both_lens_families_have_finite_target_density(specification) -> None:
    rng = np.random.default_rng(90210)
    seen = set()
    for _ in range(100):
        draw = sample_latent(rng, specification)
        seen.add(draw.lens_family)
        assert math.isfinite(log_evaluation_density(draw, specification))
    assert seen == {LensFamily.SIE_EXTERNAL_SHEAR, LensFamily.EPL_EXTERNAL_SHEAR}


def test_recorded_latent_hard_stop_is_authentic_and_pair_free() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["proposal_configuration_hash"] == EXPECTED_CONFIG_HASH
    assert result["deterministic_replay_sha256"] == EXPECTED_REPLAY_HASH
    assert result["status"] == "failed_hard_stop"
    assert result["draw_count"] == 200_000
    assert result["overall_relative_ess"] < 0.50
    assert all(value < 0.40 for value in result["relative_ess_by_lens_family"].values())
    assert result["waveform_pair_count"] == 0
    assert result["accepted_pair_generator_called"] is False
    assert result["proposal_ab_qualification_run"] is False
    replay_line = (ROOT / "results/phase3c0/proposal_v2_replay.sha256").read_text()
    assert replay_line.startswith(EXPECTED_REPLAY_HASH)
    assert len(EXPECTED_REPLAY_HASH) == 64
