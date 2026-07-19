from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.training.features import InputStandardizer, PreparedExample
from gwlens_mm.training.reference_baseline import (
    NEIGHBOR_COUNT,
    POSTERIOR_DRAW_COUNT,
    REFERENCE_CONFIG_HASH,
    ReferenceBankIndex,
    ReferenceBaselineGateError,
    build_reference_posterior,
    build_standardized_reference_bank,
    dry_run_plan,
    load_reference_baseline_contract,
    reference_feature_vector,
    score_standardized_reference_queries,
)

ROOT = Path(__file__).resolve().parents[1]


def _example(
    index: int,
    *,
    family: str = "sie_external_shear",
    em_cell: str = "full_precise_spectroscopic",
) -> PreparedExample:
    family_condition = (
        np.asarray((1.0, 0.0), dtype=np.float32)
        if family == "sie_external_shear"
        else np.asarray((0.0, 1.0), dtype=np.float32)
    )
    scalar = np.linspace(0.0, 1.0, 22, dtype=np.float32) + index / 1000.0
    astrometry = np.zeros((5, 9), dtype=np.float32)
    astrometry[:2, :5] = index / 2000.0
    astrometry[:2, 5:8] = np.eye(3, dtype=np.float32)[:2]
    return PreparedExample(
        gw_strain=np.full((2, 3, 8), index, dtype=np.float32),
        detector_mask=np.ones((2, 3), dtype=np.float32),
        astrometry_items=astrometry,
        astrometry_mask=np.asarray((1, 1, 0, 0, 0), dtype=np.float32),
        scalar_features=scalar,
        scalar_mask=np.ones(22, dtype=np.float32),
        modality_mask=np.ones(7, dtype=np.float32),
        lens_family_condition=family_condition,
        target=np.asarray(
            (0.2 + index / 500.0, 0.7 + (index % 17) / 100.0),
            dtype=np.float32,
        ),
        physical_system_id=f"system-{index:05d}",
        lens_family=family,
        em_cell_signature="all_modalities",
        em_cell=em_cell,
    )


def _bank() -> list[PreparedExample]:
    return [_example(index) for index in range(300)]


def test_rc7_supersedes_only_the_impossible_gold_claim_and_stays_closed() -> None:
    config, authorization = load_reference_baseline_contract(ROOT)
    assert configuration_hash(config) == REFERENCE_CONFIG_HASH
    assert config["preregistration_version"] == "1.1.0-rc.7"
    superseded = config["superseded_non_executable_obligation"]
    assert superseded["full_latent_proposal_implemented"] is False
    assert superseded["importance_sampling_efficiency_claim_authorized"] is False
    assert superseded["exact_likelihood_correction_claim_authorized"] is False
    assert all(value is False for value in config["execution"].values())
    assert authorization["authorization"]["scientific_reference_bank_access_authorized"] is False
    assert authorization["authorization"]["final_evaluation_unsealing_authorized"] is False


def test_reference_features_exclude_gw_detector_and_target_truth() -> None:
    example = _example(1)
    baseline = reference_feature_vector(example)
    changed = replace(
        example,
        gw_strain=np.full_like(example.gw_strain, 999.0),
        detector_mask=np.zeros_like(example.detector_mask),
        target=np.asarray((99.0, -99.0), dtype=np.float32),
    )
    assert np.array_equal(reference_feature_vector(changed), baseline)
    changed_em = replace(
        example,
        scalar_features=example.scalar_features + 1.0,
    )
    assert not np.array_equal(reference_feature_vector(changed_em), baseline)


def test_reference_posterior_uses_exact_stable_256_neighbor_contract() -> None:
    query = _example(1000)
    bank = _bank()
    posterior = build_reference_posterior(query, bank)
    assert len(posterior.neighbor_physical_system_ids) == NEIGHBOR_COUNT
    assert posterior.query_physical_system_id not in posterior.neighbor_physical_system_ids
    assert posterior.normalized_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(posterior.normalized_weights > 0.0)
    assert posterior.effective_neighbor_count > 200.0
    assert np.linalg.det(posterior.kernel_covariance) > 0.0
    repeated = build_reference_posterior(query, tuple(reversed(bank)))
    assert repeated.neighbor_physical_system_ids == posterior.neighbor_physical_system_ids
    assert np.array_equal(repeated.normalized_weights, posterior.normalized_weights)
    assert np.array_equal(repeated.kernel_covariance, posterior.kernel_covariance)


def test_reference_bank_index_is_order_stable_and_group_disjoint() -> None:
    bank = _bank()
    first = ReferenceBankIndex.build(bank)
    second = ReferenceBankIndex.build(tuple(reversed(bank)))
    assert first.identity_sha256 == second.identity_sha256
    assert first.feature_dimension == len(reference_feature_vector(bank[0]))
    group = next(iter(first.groups.values()))
    assert group.features.flags.writeable is False
    with pytest.raises(TypeError):
        first.groups[("extra", "extra")] = group  # type: ignore[index]
    query = _example(1000)
    posterior = first.posterior(query)
    repeated = second.posterior(query)
    assert posterior.neighbor_physical_system_ids == repeated.neighbor_physical_system_ids
    assert np.array_equal(posterior.normalized_weights, repeated.normalized_weights)
    with pytest.raises(ReferenceBaselineGateError, match="overlaps"):
        first.posterior(bank[0])
    permitted_self_exclusion = first.posterior(
        bank[0], require_group_disjoint=False
    )
    assert bank[0].physical_system_id not in (
        permitted_self_exclusion.neighbor_physical_system_ids
    )


def test_reference_bank_index_rejects_missing_cell_duplicate_and_sparse_stratum() -> None:
    missing_cell = replace(_example(0), em_cell=None)
    with pytest.raises(ReferenceBaselineGateError, match="EM-cell"):
        ReferenceBankIndex.build([missing_cell])
    with pytest.raises(ReferenceBaselineGateError, match="duplicated"):
        ReferenceBankIndex.build([_example(0), _example(0)])
    sparse = ReferenceBankIndex.build(_bank()[:255])
    with pytest.raises(ReferenceBaselineGateError, match="fewer than 256"):
        sparse.posterior(_example(1000))


def test_reference_index_scores_all_frozen_metrics_without_persisting_draws() -> None:
    index = ReferenceBankIndex.build(_bank())
    score = index.score(_example(1000))
    value = score.as_mapping()
    assert value["physical_system_id"] == "system-01000"
    assert np.isfinite(value["log_probability"])
    assert np.isfinite(value["negative_log_probability_per_target_dimension"])
    assert len(value["crps"]) == 2
    assert set(value["marginal_coverage"]) == {"0.50", "0.80", "0.90", "0.95"}
    assert set(value["joint_central_coverage"]) == set(value["marginal_coverage"])
    assert set(value["interval_width"]) == set(value["marginal_coverage"])
    assert value["posterior_draws_persisted"] is False
    assert value["reference_is_exact_likelihood_or_gold"] is False
    assert len(value["neighbor_identity_sha256"]) == 64
    assert index.score(_example(1000)).as_mapping() == value
    with pytest.raises(ReferenceBaselineGateError, match="duplicated"):
        index.score_queries((_example(1000), _example(1000)))


def test_reference_execution_surface_is_metadata_only_and_uses_frozen_scales() -> None:
    class Dataset:
        def __init__(self, examples: list[PreparedExample]) -> None:
            self.examples = examples
            self.metadata_calls = 0

        def __len__(self) -> int:
            return len(self.examples)

        def metadata_example(self, index: int) -> PreparedExample:
            self.metadata_calls += 1
            return replace(self.examples[index], gw_strain=np.empty((0,), dtype=np.float32))

    training = Dataset(_bank())
    standardizer = InputStandardizer.fit(
        [replace(item, gw_strain=np.empty((0,), dtype=np.float32)) for item in _bank()]
    )
    index = build_standardized_reference_bank(training, standardizer)
    assert training.metadata_calls == len(training)
    queries = Dataset([_example(1000), _example(1001)])
    scores = score_standardized_reference_queries(index, queries, standardizer)
    assert len(scores) == 2
    assert queries.metadata_calls == 2
    assert all(score.physical_system_id.startswith("system-") for score in scores)


def test_query_truth_and_gw_cannot_change_neighbors_weights_or_draws() -> None:
    query = _example(1000)
    changed = replace(
        query,
        target=np.asarray((1000.0, -1000.0), dtype=np.float32),
        gw_strain=np.full_like(query.gw_strain, -123.0),
        detector_mask=np.zeros_like(query.detector_mask),
    )
    first = build_reference_posterior(query, _bank())
    second = build_reference_posterior(changed, _bank())
    assert first.neighbor_physical_system_ids == second.neighbor_physical_system_ids
    assert np.array_equal(first.normalized_weights, second.normalized_weights)
    assert np.array_equal(first.kernel_covariance, second.kernel_covariance)
    assert np.array_equal(first.sample(), second.sample())


def test_family_cell_and_self_exclusion_fail_closed() -> None:
    query = _example(0)
    bank = _bank()
    posterior = build_reference_posterior(query, bank)
    assert query.physical_system_id not in posterior.neighbor_physical_system_ids
    wrong_family = [
        _example(index, family="epl_external_shear") for index in range(300, 600)
    ]
    wrong_cell = [
        _example(index, em_cell="full_photometric_redshifts")
        for index in range(600, 900)
    ]
    mixed = bank[:200] + wrong_family + wrong_cell
    with pytest.raises(ReferenceBaselineGateError, match="fewer than 256"):
        build_reference_posterior(query, mixed)


def test_kde_density_is_normalized_component_mixture_and_sampling_replays() -> None:
    posterior = build_reference_posterior(_example(1000), _bank())
    point = np.asarray((0.5, 0.8))
    covariance = posterior.kernel_covariance
    inverse = np.linalg.inv(covariance)
    normalization = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))
    delta = point - posterior.neighbor_targets
    direct = np.sum(
        posterior.normalized_weights
        * normalization
        * np.exp(-0.5 * np.einsum("ni,ij,nj->n", delta, inverse, delta))
    )
    assert np.exp(posterior.log_probability(point)) == pytest.approx(direct, rel=1e-12)
    batch = posterior.log_probability(np.stack((point, point + 0.1)))
    assert batch.shape == (2,)
    first = posterior.sample()
    second = posterior.sample()
    assert first.shape == (POSTERIOR_DRAW_COUNT, 2)
    assert np.array_equal(first, second)
    with pytest.raises(ReferenceBaselineGateError, match="4096"):
        posterior.sample(32)


def test_dry_plan_and_cli_cannot_open_scientific_data(tmp_path: Path) -> None:
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_execution_closed"
    assert plan["scientific_reference_bank_identity"] is None
    assert plan["query_publication_identity"] is None
    assert plan["reference_bank_accessed"] is False
    assert plan["final_evaluation_accessed"] is False
    assert plan["likelihood_gold_claimed"] is False
    assert plan["importance_sampling_efficiency_computed"] is False
    output = tmp_path / "reference-plan.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase7/prepare_reference_baseline.py"),
            "--root",
            str(ROOT),
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )
    assert json.loads(output.read_text()) == plan


def test_parent_rc6_hash_remains_frozen() -> None:
    parent = load_yaml(ROOT / "configs/statistics/final_evaluation_analysis_preregistration.yaml")
    assert configuration_hash(parent) == (
        "7e0e252f0a972e0b0ad2fe8f93f74f1f0172639a6fb258fc7a953be5fb7973e1"
    )
