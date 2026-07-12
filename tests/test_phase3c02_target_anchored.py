from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.proposals.target_anchored import (
    TargetAnchoredSpecification,
    build_v3_dry_run_plan,
    ess_certificate,
    log_target_density,
    log_v3_density,
    log_v3_weight,
    sample_evaluation_target,
    sample_v3,
)
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "configs/proposals/proposal_v3_target_anchored_mixture.yaml"
V2_PATH = ROOT / "configs/proposals/proposal_v2_exact_mixture.yaml"
AUTH_PATH = ROOT / "configs/execution/phase3c02_target_anchored_proposal_authorization.yaml"
EXPECTED_V3_HASH = "2d7998ca099c1ecddbb5d9cb1d824f37d3d398826a88831b2bccddbda814cbf4"
IMMUTABLE_V2_FILES = {
    "configs/proposals/proposal_v2_exact_mixture.yaml": (
        "f3d2b08cd5c7625960bb3a343da2a447174f6597de9657d6875ebfc8de8a2b27"
    ),
    "results/phase3c0/proposal_v2_latent_validation.json": (
        "a6eca279cc29d1c1f8075e45084afc1440022bc28539ed7c95327613a74b37f2"
    ),
    "results/phase3c0/proposal_v2_replay.sha256": (
        "65d230985665acf79c84af0d2edfcf9b24dfa0735f51584cec276ddaa7648b88"
    ),
    "docs/reports/PHASE3C0_PROPOSAL_V2_IMPLEMENTATION_REPORT.md": (
        "2284c5c24e3dd22438b6669b50525b17a4b40a02a35374768b73ac27b9f92a51"
    ),
}


def _config():
    return load_yaml(V3_PATH)


def _specification():
    return TargetAnchoredSpecification.from_mapping(_config())


def test_v3_identity_weights_hash_and_authorization_are_frozen() -> None:
    config = _config()
    specification = _specification()
    assert config["proposal_version"] == "proposal-v3-target-anchored-mixture-1.0.0-rc.1"
    assert specification.weights == (0.20, 0.55, 0.25)
    assert configuration_hash(config) == EXPECTED_V3_HASH
    authorization = load_yaml(AUTH_PATH)["authorization"]
    for key in (
        "waveform_pair_generation_authorized",
        "accepted_pair_generator_authorized",
        "proposal_ab_qualification_authorized",
        "scientific_data_generation_authorized",
        "model_training_authorized",
        "gwosc_gwtc_access_authorized",
        "phase3c_a_or_later_authorized",
    ):
        assert authorization[key] is False


def test_rejected_v2_evidence_is_byte_immutable() -> None:
    for relative, expected in IMMUTABLE_V2_FILES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_exact_evaluation_sampler_is_deterministic_and_in_support() -> None:
    specification = _specification()
    first = np.random.default_rng(123)
    second = np.random.default_rng(123)
    draws = [sample_evaluation_target(first, specification) for _ in range(2000)]
    assert draws == [sample_evaluation_target(second, specification) for _ in range(2000)]
    for draw in draws:
        assert -2.5 <= draw.source_u_x < 2.5
        assert -2.5 <= draw.source_u_y < 2.5
        assert max(0.5, draw.z_lens + 0.1) <= draw.z_source < 3.0
        assert math.isfinite(log_target_density(draw, specification))


def test_target_anchor_and_weight_bound_hold_for_both_families() -> None:
    specification = _specification()
    rng = np.random.default_rng(999)
    seen = set()
    for _ in range(10_000):
        _, draw = sample_v3(rng, specification)
        seen.add(draw.lens_family.value)
        log_p = log_target_density(draw, specification)
        log_q = log_v3_density(draw, specification)
        assert log_q + 1.0e-12 >= log_p + math.log(0.55)
        assert math.exp(log_v3_weight(draw, specification)) <= 1.0 / 0.55 + 1.0e-12
    assert seen == {"sie_external_shear", "epl_external_shear"}


def test_theoretical_ess_certificate_is_exact() -> None:
    certificate = ess_certificate(_config())
    assert certificate["status"] == "passed"
    assert certificate["population_relative_ess_lower_bound"] == 0.55
    assert certificate["per_family_bound"] == 0.55
    assert math.isclose(certificate["maximum_importance_weight"], 1.0 / 0.55)


def test_recorded_v3_preflight_passes_and_rc5_is_diagnostic_only() -> None:
    v3 = json.loads(
        (ROOT / "results/phase3c02/proposal_v3_latent_validation.json").read_text()
    )
    rc5 = json.loads((ROOT / "results/phase3c02/rc5_latent_baseline.json").read_text())
    assert v3["status"] == "passed"
    assert v3["draw_count"] == 200_000
    assert v3["overall_relative_ess"] >= 0.50
    assert all(value >= 0.40 for value in v3["relative_ess_by_lens_family"].values())
    assert v3["maximum_importance_weight"] <= 1.0 / 0.55
    assert v3["target_anchor_failure_count"] == 0
    assert v3["replay_byte_identical"] is True
    assert v3["waveform_pair_count"] == 0
    assert rc5["interpretation"] == "diagnostic_only_no_retrospective_gate"
    assert rc5["overall_relative_ess"] == 0.11776258208979214


def test_factorwise_diagnostics_are_explicitly_marginal() -> None:
    result = json.loads(
        (ROOT / "results/phase3c02/factorwise_weight_diagnostics.json").read_text()
    )
    assert result["interpretation"] == (
        "diagnostic_marginals_not_unique_sequential_decomposition"
    )
    assert result["rc5"]["lens_structural"]["relative_ess"] < 0.20
    assert result["proposal_v3"]["lens_structural"]["relative_ess"] > 0.90


def test_v3_ab_plan_is_still_dry_run_only() -> None:
    plan = build_v3_dry_run_plan(load_yaml(V2_PATH), _config(), "phase3c02-fixture")
    assert plan.dry_run_only is True
    assert plan.maximum_accepted_pairs_per_arm == 512
    assert plan.maximum_accepted_pairs_total == 1024
    assert "proposal-v3-target-anchored-rc1" in plan.candidate_dataset_id
    assert plan.control_dataset_id != plan.candidate_dataset_id
    assert len(plan.blocks) == 16
