from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.features import PreparedExample
from gwlens_mm.training.final_evaluation import (
    FINAL_ANALYSIS_HASH,
    FinalEvaluationGateError,
    aggregate_seed_metrics,
    condition_on_lens_family,
    coverage_gate,
    cross_family_analysis_context,
    dry_run_plan,
    em_only_example,
    equal_family_draw_mixture,
    equal_family_log_mixture,
    gw_only_example,
    load_final_evaluation_analysis_contract,
    validate_final_split_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def _example() -> PreparedExample:
    return PreparedExample(
        gw_strain=np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        detector_mask=np.ones((2, 3), dtype=np.float32),
        astrometry_items=np.ones((5, 9), dtype=np.float32),
        astrometry_mask=np.ones(5, dtype=np.float32),
        scalar_features=np.arange(1, 23, dtype=np.float32),
        scalar_mask=np.ones(22, dtype=np.float32),
        modality_mask=np.ones(8, dtype=np.float32),
        lens_family_condition=np.asarray((1.0, 0.0), dtype=np.float32),
        target=np.asarray((2.0, 3.0), dtype=np.float32),
        physical_system_id="physical-system-a",
        lens_family="sie_external_shear",
        em_cell_signature="astrometry,timing",
    )


def test_rc6_contract_is_hash_bound_and_every_execution_gate_is_closed() -> None:
    config, authorization = load_final_evaluation_analysis_contract(ROOT)
    assert configuration_hash(config) == FINAL_ANALYSIS_HASH
    assert config["preregistration_version"] == "1.1.0-rc.6"
    assert config["final_pool"]["total_accepted_physical_systems"] == 20480
    assert config["posterior_inference"]["seeds"] == [0, 1, 2]
    assert all(value is False for value in config["execution"].values())
    forbidden = {
        name: value
        for name, value in authorization["authorization"].items()
        if name not in {
            "downstream_preregistration_authorized",
            "pure_metric_implementation_authorized",
            "synthetic_fixture_tests_authorized",
        }
    }
    assert forbidden and all(value is False for value in forbidden.values())


def test_frozen_generator_configuration_and_commitment_are_byte_unchanged() -> None:
    config = load_yaml(ROOT / "configs/data/phase4_final_evaluation.yaml")
    assert configuration_hash(config) == (
        "11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66"
    )
    commitment = ROOT / "results/phase4/final_evaluation_commitment.json"
    assert hashlib.sha256(commitment.read_bytes()).hexdigest() == (
        "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
    )
    payload = json.loads(commitment.read_text())
    assert payload["future_scientific_generator_commit"] == (
        "bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac"
    )
    assert payload["training_or_materialization_authorized"] is False


def test_cross_family_contexts_replace_only_the_non_executable_analysis_semantics() -> None:
    epl = cross_family_analysis_context("sie_truth_epl_assumed")
    assert epl.analysis_cell_id == "sie_truth_epl_prior_marginalized_assumed"
    assert epl.inference_mode.endswith("training_slope_prior_marginalized")
    assert "2.08" not in epl.inference_mode
    mixture = cross_family_analysis_context("epl_truth_family_marginalized")
    assert mixture.inference_mode == "equal_density_mixture_of_sie_and_epl_family_conditions"
    with pytest.raises(FinalEvaluationGateError, match="unknown cross-family"):
        cross_family_analysis_context("invented_context")
    maintained = (ROOT / "src/gwlens_mm/training/final_evaluation.py").read_text()
    assert "fixed_slope_2.08" not in maintained


def test_family_condition_and_equal_mixture_have_exact_semantics() -> None:
    original = _example()
    changed = condition_on_lens_family(original, "epl_external_shear")
    assert changed.lens_family_condition.tolist() == [0.0, 1.0]
    assert changed.physical_system_id == original.physical_system_id
    assert np.array_equal(changed.gw_strain, original.gw_strain)
    assert np.array_equal(changed.target, original.target)
    assert original.lens_family_condition.tolist() == [1.0, 0.0]
    with pytest.raises(FinalEvaluationGateError, match="unsupported"):
        condition_on_lens_family(original, "sis")

    sie_log = np.log(np.asarray((0.2, 0.8)))
    epl_log = np.log(np.asarray((0.6, 0.1)))
    observed = np.exp(equal_family_log_mixture(sie_log, epl_log))
    assert observed.tolist() == pytest.approx([0.4, 0.45])
    sie_draws = np.zeros((2, 2048, 2), dtype=np.float32)
    epl_draws = np.ones((2, 2048, 2), dtype=np.float32)
    mixture_draws = equal_family_draw_mixture(sie_draws, epl_draws)
    assert mixture_draws.shape == (2, 4096, 2)
    assert np.all(mixture_draws[:, 0::2] == 0.0)
    assert np.all(mixture_draws[:, 1::2] == 1.0)


def test_ablation_views_remove_only_the_frozen_inputs() -> None:
    original = _example()
    gw = gw_only_example(original)
    assert np.array_equal(gw.gw_strain, original.gw_strain)
    assert np.array_equal(gw.detector_mask, original.detector_mask)
    assert np.array_equal(gw.scalar_features[:2], original.scalar_features[:2])
    assert np.array_equal(gw.scalar_mask[:2], original.scalar_mask[:2])
    assert np.all(gw.scalar_features[2:] == 0.0)
    assert np.all(gw.scalar_mask[2:] == 0.0)
    assert np.all(gw.astrometry_items == 0.0)
    assert np.all(gw.astrometry_mask == 0.0)
    assert np.all(gw.modality_mask == 0.0)

    em = em_only_example(original)
    assert np.all(em.gw_strain == 0.0)
    assert np.all(em.detector_mask == 0.0)
    assert np.all(em.scalar_features[:2] == 0.0)
    assert np.all(em.scalar_mask[:2] == 0.0)
    assert np.array_equal(em.scalar_features[2:], original.scalar_features[2:])
    assert np.array_equal(em.astrometry_items, original.astrometry_items)
    for view in (gw, em):
        assert np.array_equal(view.lens_family_condition, original.lens_family_condition)
        assert np.array_equal(view.target, original.target)
        assert view.physical_system_id == original.physical_system_id
    assert np.any(original.gw_strain != 0.0)
    assert np.any(original.astrometry_items != 0.0)


def test_coverage_gate_uses_frozen_binomial_floor_and_raw_counts() -> None:
    iid = coverage_gate(7373, 8192, 0.90, 0.01)
    assert iid["three_binomial_standard_errors"] == pytest.approx(0.0099437, rel=1e-4)
    assert iid["tolerance"] == pytest.approx(0.01)
    assert iid["passed"] is True
    tail = coverage_gate(900, 1024, 0.90, 0.06)
    assert tail["tolerance"] == pytest.approx(0.06)
    assert tail["successes"] == 900
    failed = coverage_gate(7000, 8192, 0.90, 0.01)
    assert failed["passed"] is False


def test_seed_aggregation_requires_all_seeds_and_never_selects_one() -> None:
    result = aggregate_seed_metrics(
        {
            0: {"nlp": 1.0, "crps": 4.0},
            1: {"nlp": 2.0, "crps": 5.0},
            2: {"nlp": 3.0, "crps": 6.0},
        }
    )
    assert result["nlp"] == pytest.approx({"mean": 2.0, "sample_standard_deviation": 1.0})
    with pytest.raises(FinalEvaluationGateError, match="seeds 0, 1, and 2"):
        aggregate_seed_metrics({0: {"nlp": 1.0}, 1: {"nlp": 2.0}})


def test_final_split_identity_validator_requires_exact_disjoint_pool() -> None:
    counts = {
        "iid_test": 8192,
        "balanced_tail_diagnostic": 4096,
        "cross_family_misspecification_test": 2048,
        "parameter_region_ood": 2048,
        "waveform_mismatch_test": 2048,
        "psd_mismatch_test": 2048,
    }
    ids = {
        split: tuple(f"{split}-{index}" for index in range(count))
        for split, count in counts.items()
    }
    observed = validate_final_split_ids(ids)
    assert observed["total"] == 20480
    changed = dict(ids)
    changed["psd_mismatch_test"] = (
        ids["iid_test"][0],
        *ids["psd_mismatch_test"][1:],
    )
    with pytest.raises(FinalEvaluationGateError, match="overlap"):
        validate_final_split_ids(changed)


def test_dry_plan_and_cli_cannot_resolve_or_open_final_data(tmp_path: Path) -> None:
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_execution_closed"
    assert plan["final_case_count"] == 20480
    assert plan["official_final_data_identity"] is None
    assert plan["selected_checkpoint_identities"] is None
    assert plan["final_data_accessed"] is False
    assert plan["metric_computed"] is False
    output = tmp_path / "plan.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase7/prepare_final_evaluation_analysis.py"),
            "--root",
            str(ROOT),
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )
    assert json.loads(output.read_text()) == plan
