from __future__ import annotations

import json
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.schema import V2Record
from gwlens_mm.training.contracts import (
    TrainingGateError,
    deterministic_probe_subset,
    load_training_stack_contract,
    model_configuration_hash,
    validate_direct_target_unit_weights,
    validate_scientific_training_gate,
)
from gwlens_mm.training.data import index_complete_shards
from gwlens_mm.training.engine import (
    DeterministicEpochSampler,
    TargetStandardizer,
    TrainingRunIdentity,
    standardizer_hash,
    validate_engineering_smoke_limits,
)
from gwlens_mm.training.features import InputStandardizer, load_input_policy, prepare_example
from gwlens_mm.training.metrics import (
    central_coverage,
    empirical_crps,
    negative_log_probability_per_dimension,
)
from gwlens_mm.training.model import MissingTrainingDependency, build_probe_model
from gwlens_mm.training.whitening import ASDCurve, bilby_psd_whiten, whiten_detector_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/v2_metadata_example.json"


def _example_record() -> V2Record:
    return V2Record.from_dict(json.loads(EXAMPLE.read_text()))


def test_implementation_authorization_keeps_scientific_training_closed() -> None:
    authorization, model = load_training_stack_contract(ROOT)
    assert authorization["authorization"]["training_stack_implementation_authorized"] is True
    assert authorization["authorization"]["scientific_probe_training_authorized"] is False
    assert authorization["authorization"]["stage_a_data_access_authorized"] is False
    assert model["execution"] == {
        "scientific_training_enabled": False,
        "model_selection_enabled": False,
        "stage_a_access_enabled": False,
        "engineering_smoke_only": True,
    }
    assert len(model_configuration_hash(model)) == 64


def test_scientific_training_gate_fails_closed_before_publication_and_commitment() -> None:
    with pytest.raises(TrainingGateError, match="authorization is absent"):
        validate_scientific_training_gate(
            ROOT,
            authorization_path=(
                ROOT / "configs/execution/phase4_probe_training_stack_authorization.yaml"
            ),
            train_publication_root=Path("/not/published/train"),
            validation_publication_root=Path("/not/published/validation"),
        )


def test_probe_subset_requires_complete_32k_and_is_order_invariant() -> None:
    ids = tuple(f"system-{index:05d}" for index in range(32768))
    selected = deterministic_probe_subset(ids, root_seed=2026071401)
    reversed_selected = deterministic_probe_subset(tuple(reversed(ids)), root_seed=2026071401)
    assert len(selected) == 16384
    assert len(set(selected)) == 16384
    assert selected == reversed_selected
    with pytest.raises(TrainingGateError, match="exactly 32768"):
        deterministic_probe_subset(ids[:16384], root_seed=2026071401)


def test_direct_target_density_provenance_is_validated_but_not_an_input() -> None:
    record = json.loads(EXAMPLE.read_text())
    assert validate_direct_target_unit_weights([record]) == 1
    record["provenance"]["distribution"]["importance_weight"] = 0.9
    with pytest.raises(TrainingGateError, match="unit weights"):
        validate_direct_target_unit_weights([record])
    policy = load_input_policy(ROOT)
    with pytest.raises(ValueError, match="forbidden model inputs"):
        policy.validate_model_inputs(["importance_weight"])


def test_bilby_whitening_matches_frozen_formula_without_event_std() -> None:
    rng = np.random.default_rng(37)
    series = rng.normal(size=(2, 1024)).astype(np.float32)
    curve = ASDCurve(
        np.asarray((0.0, 512.0)), np.asarray((2.0, 2.0)), "constant-test-asd"
    )
    observed = bilby_psd_whiten(
        series,
        sampling_frequency_hz=1024.0,
        asd_curve=curve,
        minimum_frequency_hz=20.0,
        maximum_frequency_hz=512.0,
    )
    frequency = np.fft.rfftfreq(1024, d=1 / 1024.0)
    mask = (frequency >= 20.0) & (frequency <= 512.0)
    fd = np.fft.rfft(series, axis=-1) / 1024.0
    whitened_fd = np.zeros_like(fd)
    whitened_fd[..., mask] = fd[..., mask] / (2.0 * np.sqrt(1.0 / 4.0))
    expected = np.fft.irfft(whitened_fd, n=1024, axis=-1)
    expected *= np.sqrt(mask.sum()) / (mask.sum() / len(frequency))
    np.testing.assert_allclose(observed, expected.astype(np.float32), rtol=2e-6, atol=2e-6)


def test_detector_whitening_preserves_unavailable_zero_slots() -> None:
    curve = ASDCurve(np.asarray((0.0, 512.0)), np.asarray((1.0, 1.0)), "flat")
    strain = np.ones((2, 3, 1024), dtype=np.float32)
    mask = np.asarray(((True, False, True), (True, True, False)))
    strain[~mask] = 0.0
    output = whiten_detector_grid(
        strain,
        mask,
        sampling_frequency_hz=1024,
        detector_curves=(curve, curve, curve),
        minimum_frequency_hz=20,
    )
    assert output.shape == strain.shape
    assert output.dtype == np.float32
    assert np.all(output[~mask] == 0.0)


def test_feature_extraction_supports_five_observed_images_without_truth_ids() -> None:
    data = deepcopy(json.loads(EXAMPLE.read_text()))
    base_image = data["lens_truth"]["physical_images"][2]
    for index in (4, 5):
        image = deepcopy(base_image)
        image["image_id"] = f"image_{index}"
        image["position_arcsec"] = [0.01 * index, -0.01 * index]
        image["arrival_time_seconds"] += index
        data["lens_truth"]["physical_images"].append(image)
        data["lens_truth"]["censored_image_ids"].append(image["image_id"])
        astrometry = deepcopy(data["em_observation"]["observed_image_astrometry"][-1])
        astrometry["image_id"] = image["image_id"]
        astrometry["position_arcsec"] = image["position_arcsec"]
        data["em_observation"]["observed_image_astrometry"].append(astrometry)
        data["em_observation"]["censoring_flags"][image["image_id"]] = True
    record = V2Record.from_dict(data)
    strain = np.zeros((2, 3, record.gw_observation.sample_count), dtype=np.float32)
    example = prepare_example(record, strain)
    assert example.astrometry_items.shape == (5, 9)
    assert example.astrometry_mask.sum() == 5
    assert example.scalar_features.shape == (22,)
    assert example.modality_mask.shape == (7,)
    np.testing.assert_array_equal(example.lens_family_condition, (1.0, 0.0))
    np.testing.assert_allclose(example.target, np.log((3.0, 2.0)), rtol=1e-6)
    assert not hasattr(example, "importance_weight")
    assert not hasattr(example, "source_id")


def test_manifest_only_shard_index_refuses_partial_and_duplicates(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shards = dataset / "shards"
    for shard_index in range(2):
        shard = shards / f"shard-{shard_index:05d}"
        shard.mkdir(parents=True)
        (shard / "COMPLETE.json").write_text("{}\n")
        ids = [f"system-{shard_index}-{row}" for row in range(2)]
        (shard / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "accepted_pair_count": 2,
                    "physical_system_ids": ids,
                }
            )
        )
    entries = index_complete_shards(
        dataset,
        expected_pairs_per_shard=2,
        expected_total_pairs=4,
        require_published=False,
    )
    assert [entry.physical_system_id for entry in entries] == [
        "system-0-0",
        "system-0-1",
        "system-1-0",
        "system-1-1",
    ]
    (shards / "shard-00002.partial").mkdir()
    with pytest.raises(TrainingGateError, match="partial"):
        index_complete_shards(dataset, expected_pairs_per_shard=2, require_published=False)


def test_metrics_and_target_standardization_are_finite() -> None:
    target = np.asarray(((0.0, 1.0), (1.0, 2.0)))
    samples = np.asarray(
        [
            [[-0.5 + i / 10, 0.5 + i / 10] for i in range(11)],
            [[0.5 + i / 10, 1.5 + i / 10] for i in range(11)],
        ]
    )
    crps = empirical_crps(samples, target)
    assert crps.shape == (2, 2)
    assert np.all(crps >= 0)
    coverage = central_coverage(samples, target, (0.8,))
    assert set(coverage) == {"marginal_0.80", "joint_0.80", "width_0.80"}
    assert negative_log_probability_per_dimension(np.asarray((-2.0, -4.0)), 2) == 1.5
    standardizer = TargetStandardizer.fit(target)
    standardized = standardizer.transform_numpy(target)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-7)


def test_epoch_sampler_is_resume_stable_and_epoch_specific() -> None:
    sampler = DeterministicEpochSampler(64, seed=1)
    sampler.set_epoch(7)
    first = tuple(sampler)
    sampler.set_epoch(7)
    replay = tuple(sampler)
    sampler.set_epoch(8)
    next_epoch = tuple(sampler)
    assert first == replay
    assert first != next_epoch
    assert set(first) == set(range(64))


def test_input_standardizer_uses_only_observed_values_and_preserves_missing_zero() -> None:
    record = _example_record()
    strain = np.zeros((2, 3, record.gw_observation.sample_count), dtype=np.float32)
    first = prepare_example(record, strain)
    second = deepcopy(first)
    second.scalar_features[0] += 2.0
    standardizer = InputStandardizer.fit((first, second))
    transformed = standardizer.transform(first)
    assert np.all(transformed.scalar_features[first.scalar_mask == 0] == 0.0)
    assert np.all(transformed.astrometry_items[first.astrometry_mask == 0] == 0.0)
    assert np.all(np.isfinite(transformed.scalar_features))
    assert np.all(np.isfinite(transformed.astrometry_items))
    target_standardizer = TargetStandardizer.fit(
        np.asarray(((0.0, 1.0), (1.0, 2.0)))
    )
    identity = TrainingRunIdentity(
        model_configuration_hash="0" * 64,
        training_code_commit="1" * 40,
        training_environment_sha256="2" * 64,
        train_manifest_sha256="3" * 64,
        validation_manifest_sha256="4" * 64,
        final_evaluation_commitment_sha256="5" * 64,
        membership_sha256="6" * 64,
        input_standardizer_sha256=standardizer_hash(standardizer),
        target_standardizer_sha256=standardizer_hash(target_standardizer),
        training_rung_count=16384,
        seed=0,
    )
    identity.validate()


def test_engineering_smoke_caps_and_optional_model_dependency() -> None:
    authorization = load_yaml(
        ROOT / "configs/execution/phase4_probe_training_stack_authorization.yaml"
    )
    validate_engineering_smoke_limits(examples=48, optimizer_steps=4, authorization=authorization)
    with pytest.raises(TrainingGateError, match="example cap"):
        validate_engineering_smoke_limits(
            examples=49, optimizer_steps=4, authorization=authorization
        )
    if find_spec("torch") is None or find_spec("nflows") is None:
        model = load_yaml(ROOT / "configs/models/phase4_probe_nsf.yaml")
        with pytest.raises(MissingTrainingDependency):
            build_probe_model(model, seed=0)
