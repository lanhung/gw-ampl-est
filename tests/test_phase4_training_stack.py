from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from gwlens_mm.config import load_yaml
from gwlens_mm.schema import SplitName, V2Record
from gwlens_mm.training.contracts import (
    TrainingGateError,
    deterministic_probe_subset,
    load_training_stack_contract,
    model_configuration_hash,
    validate_direct_target_unit_weights,
    validate_scientific_training_gate,
)
from gwlens_mm.training.data import (
    PublishedStageADataset,
    index_complete_shards,
    resolve_stage_a_publication,
)
from gwlens_mm.training.engine import (
    DeterministicEpochSampler,
    DeterministicShardEpochSampler,
    TargetStandardizer,
    TrainingRunIdentity,
    _restore_rng_state,
    optimization_batch_geometry,
    standardizer_hash,
    validate_engineering_smoke_limits,
)
from gwlens_mm.training.features import InputStandardizer, load_input_policy, prepare_example
from gwlens_mm.training.learning_curve import compare_16k_to_32k
from gwlens_mm.training.metrics import (
    central_coverage,
    empirical_crps,
    negative_log_probability_per_dimension,
)
from gwlens_mm.training.model import MissingTrainingDependency, build_probe_model
from gwlens_mm.training.whitening import ASDCurve, bilby_psd_whiten, whiten_detector_grid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/v2_metadata_example.json"
STAGE_A_GENERATOR = "2be777e727ef9d8e1a85f89c68966df5d37932b0"
RC4_HASH = "5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98"


def _example_record() -> V2Record:
    return V2Record.from_dict(json.loads(EXAMPLE.read_text()))


def _write_parent_publication(
    tmp_path: Path, *, train_count: int, validation_count: int, pairs_per_shard: int
) -> Path:
    parent = tmp_path / "published" / "stage-a-parent"
    parent.mkdir(parents=True)
    identities = {"train": "scientific-train", "validation": "scientific-validation"}
    validations = {}
    for split, accepted_count in (
        ("train", train_count),
        ("validation", validation_count),
    ):
        dataset_id = identities[split]
        child = parent / dataset_id
        (child / "shards").mkdir(parents=True)
        (child / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "generating_or_resuming",
                    "split": split,
                    "dataset_id": dataset_id,
                    "generator_commit": STAGE_A_GENERATOR,
                    "accepted_target": accepted_count,
                    "all_importance_weights_one": True,
                }
            )
        )
        validations[split] = {
            "status": "passed",
            "split": split,
            "dataset_id": dataset_id,
            "accepted_pair_count": accepted_count,
            "complete_shard_count": accepted_count // pairs_per_shard,
            "pairs_per_shard": pairs_per_shard,
            "generator_commit": STAGE_A_GENERATOR,
            "configuration_hash": split * 8,
            "proposal_equals_evaluation": True,
            "all_importance_weights_one": True,
        }
    (parent / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "generator_commit": STAGE_A_GENERATOR,
                "preregistration_hash": RC4_HASH,
                "accepted_pair_count": train_count + validation_count,
                "train_accepted_pair_count": train_count,
                "validation_accepted_pair_count": validation_count,
                "validations": validations,
                "train_validation_group_disjoint": True,
                "proposal_equals_evaluation": True,
                "all_importance_weights_one": True,
                "model_training_authorized": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return parent


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
            stage_a_publication_root=Path("/not/published/stage-a"),
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


def test_atomic_parent_publication_resolves_child_datasets(tmp_path: Path) -> None:
    parent = _write_parent_publication(
        tmp_path, train_count=4, validation_count=2, pairs_per_shard=2
    )
    publication = resolve_stage_a_publication(
        parent,
        expected_generator_commit=STAGE_A_GENERATOR,
        expected_preregistration_hash=RC4_HASH,
        expected_train_count=4,
        expected_validation_count=2,
        expected_pairs_per_shard=2,
    )
    assert publication.train_root == parent / "scientific-train"
    assert publication.validation_root == parent / "scientific-validation"
    assert len(publication.manifest_sha256) == 64
    assert set(publication.namespace_manifest_sha256) == {"train", "validation"}
    for shard_index in range(2):
        shard = publication.train_root / "shards" / f"shard-{shard_index:05d}"
        shard.mkdir()
        (shard / "COMPLETE.json").write_text("{}\n")
        (shard / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "accepted_pair_count": 2,
                    "physical_system_ids": [
                        f"train-system-{shard_index}-0",
                        f"train-system-{shard_index}-1",
                    ],
                }
            )
            + "\n"
        )
    entries = index_complete_shards(
        publication.train_root,
        expected_pairs_per_shard=2,
        expected_total_pairs=4,
    )
    assert len(entries) == 4
    manifest = json.loads((parent / "dataset_manifest.json").read_text())
    manifest["validations"]["train"]["dataset_id"] = "wrong-train-id"
    (parent / "dataset_manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(TrainingGateError, match="directory is absent"):
        resolve_stage_a_publication(
            parent,
            expected_train_count=4,
            expected_validation_count=2,
            expected_pairs_per_shard=2,
        )


def test_future_probe_gate_accepts_only_exact_parent_manifest(tmp_path: Path) -> None:
    parent = _write_parent_publication(
        tmp_path,
        train_count=32768,
        validation_count=6144,
        pairs_per_shard=128,
    )
    commitment = ROOT / "results/phase4/final_evaluation_commitment.json"
    authorization = {
        "authorization_status": "authorized_probe_training_only",
        "authorization": {
            "stage_a_data_access_authorized": True,
            "scientific_probe_training_authorized": True,
            "probe_optimizer_execution_authorized": True,
            "learning_curve_decision_authorized": True,
            "model_tuning_authorized": False,
            "calibration_authorized": False,
            "sbc_authorized": False,
            "final_evaluation_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
        "authorized_training_rungs": [16384, 32768],
        "authorized_training_seeds": [0, 1, 2],
        "final_evaluation_commitment_sha256": hashlib.sha256(
            commitment.read_bytes()
        ).hexdigest(),
        "stage_a_generator_commit": STAGE_A_GENERATOR,
        "stage_a_parent_manifest_sha256": hashlib.sha256(
            (parent / "dataset_manifest.json").read_bytes()
        ).hexdigest(),
    }
    authorization_path = tmp_path / "probe_authorization.yaml"
    authorization_path.write_text(yaml.safe_dump(authorization, sort_keys=False))
    evidence = validate_scientific_training_gate(
        ROOT,
        authorization_path=authorization_path,
        stage_a_publication_root=parent,
    )
    assert evidence["publication"].train_dataset_id == "scientific-train"
    authorization["stage_a_parent_manifest_sha256"] = "0" * 64
    authorization_path.write_text(yaml.safe_dump(authorization, sort_keys=False))
    with pytest.raises(TrainingGateError, match="parent manifest hash"):
        validate_scientific_training_gate(
            ROOT,
            authorization_path=authorization_path,
            stage_a_publication_root=parent,
        )


def test_published_reader_traverses_parent_parquet_zarr_path(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    zarr = pytest.importorskip("zarr")
    parent = _write_parent_publication(
        tmp_path, train_count=2, validation_count=2, pairs_per_shard=2
    )
    publication = resolve_stage_a_publication(
        parent,
        expected_train_count=2,
        expected_validation_count=2,
        expected_pairs_per_shard=2,
    )
    shard = publication.train_root / "shards" / "shard-00000"
    shard.mkdir()
    serialized = []
    physical_ids = []
    for index in range(2):
        record = deepcopy(json.loads(EXAMPLE.read_text()))
        record["pair"].update(
            {
                "dataset_version": publication.train_dataset_id,
                "pair_id": f"reader-pair-{index}",
                "source_id": f"reader-source-{index}",
                "lens_id": f"reader-lens-{index}",
                "physical_system_id": f"reader-system-{index}",
                "split": "train",
            }
        )
        physical_ids.append(record["pair"]["physical_system_id"])
        serialized.append(json.dumps(record, sort_keys=True))
    pandas.DataFrame({"record_json": serialized}).to_parquet(shard / "records.parquet")
    noisy = zarr.open_array(
        str(shard / "noisy.zarr"),
        mode="w",
        shape=(2, 2, 3, 4096),
        chunks=(1, 2, 3, 4096),
        dtype="f4",
    )
    noisy[:] = 0.0
    (shard / "COMPLETE.json").write_text("{}\n")
    (shard / "shard_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "accepted_pair_count": 2,
                "physical_system_ids": physical_ids,
            }
        )
        + "\n"
    )
    curve = ASDCurve(
        np.asarray((0.0, 1024.0)), np.asarray((1.0, 1.0)), "reader-test"
    )
    dataset = PublishedStageADataset(
        publication.train_root,
        expected_split=SplitName.TRAIN,
        detector_curves={"H1": curve, "L1": curve, "V1": curve},
        expected_pairs_per_shard=2,
        expected_total_pairs=2,
        minimum_frequency_hz=20.0,
        maximum_frequency_hz=1024.0,
    )
    metadata = dataset.metadata_example(0)
    assert metadata.gw_strain.size == 0
    example = dataset[0]
    assert example.gw_strain.shape == (2, 3, 4096)
    assert example.physical_system_id == "reader-system-0"
    assert np.all(np.isfinite(example.gw_strain))


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


def test_rng_restore_moves_loaded_states_back_to_cpu() -> None:
    calls: list[tuple[str, Any]] = []

    class LoadedState:
        def __init__(self, name: str) -> None:
            self.name = name

        def cpu(self) -> str:
            calls.append((self.name, "cpu"))
            return f"cpu-{self.name}"

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_rng_state_all(values: list[object]) -> None:
            calls.append(("cuda-set", tuple(values)))

    class Torch:
        cuda = Cuda()

        @staticmethod
        def set_rng_state(value: object) -> None:
            calls.append(("cpu-set", value))

    _restore_rng_state(
        Torch(),
        {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": LoadedState("torch-cpu"),
            "torch_cuda": [LoadedState("cuda-0"), LoadedState("cuda-1")],
        },
    )
    assert calls[-2:] == [
        ("cuda-1", "cpu"),
        ("cuda-set", ("cpu-cuda-0", "cpu-cuda-1")),
    ]
    assert ("cpu-set", "cpu-torch-cpu") in calls


def test_shard_sampler_is_resume_stable_and_keeps_shards_local() -> None:
    keys = tuple(f"shard-{index // 4}" for index in range(24))
    sampler = DeterministicShardEpochSampler(keys, seed=2)
    sampler.set_epoch(11)
    first = tuple(sampler)
    sampler.set_epoch(11)
    assert tuple(sampler) == first
    sampler.set_epoch(12)
    assert tuple(sampler) != first
    assert set(first) == set(range(24))
    transitions = sum(keys[left] != keys[right] for left, right in zip(first, first[1:]))
    assert transitions == 5


def test_physical_microbatch_preserves_frozen_effective_batch() -> None:
    effective, microbatch, steps = optimization_batch_geometry(
        {
            "batch_size": 256,
            "physical_microbatch_size": 64,
            "gradient_accumulation_steps": 4,
        }
    )
    assert (effective, microbatch, steps) == (256, 64, 4)
    with pytest.raises(TrainingGateError, match="must equal frozen batch size"):
        optimization_batch_geometry(
            {
                "batch_size": 256,
                "physical_microbatch_size": 64,
                "gradient_accumulation_steps": 2,
            }
        )


def test_input_standardizer_uses_only_observed_values_and_preserves_missing_zero() -> None:
    record = _example_record()
    strain = np.zeros((2, 3, record.gw_observation.sample_count), dtype=np.float32)
    first = prepare_example(record, strain)
    second = deepcopy(first)
    second.scalar_features[0] += 2.0
    standardizer = InputStandardizer.fit((first, second))
    streaming_standardizer = InputStandardizer.fit_iterable(iter((first, second)))
    assert streaming_standardizer == standardizer
    transformed = standardizer.transform(first)
    assert np.all(transformed.scalar_features[first.scalar_mask == 0] == 0.0)
    assert np.all(transformed.astrometry_items[first.astrometry_mask == 0] == 0.0)
    assert np.all(np.isfinite(transformed.scalar_features))
    assert np.all(np.isfinite(transformed.astrometry_items))
    target_standardizer = TargetStandardizer.fit(
        np.asarray(((0.0, 1.0), (1.0, 2.0)))
    )
    streaming_target_standardizer = TargetStandardizer.fit_iterable(
        iter((np.asarray((0.0, 1.0)), np.asarray((1.0, 2.0))))
    )
    assert streaming_target_standardizer == target_standardizer
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


def _write_learning_curve_cases(path: Path, *, nlp: float, crps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tail_groups = (
        "high_absolute_magnification",
        "extreme_relative_magnification",
        "second_image_near_threshold",
        "extreme_profile_or_environment",
    )
    case_count = 6144
    for index in range(case_count):
        tail_view = tail_groups[index // 128] if index < 512 else "none"
        row = {
            "physical_system_id": f"validation-system-{index:04d}",
            "lens_family": "sie_external_shear" if index % 2 == 0 else "epl_external_shear",
            "em_cell_signature": "all_modalities",
            "tail_view": tail_view,
            "nlp_nat_per_target_dimension": nlp,
            "crps_log_mu_primary": crps,
            "crps_log_mu_secondary": crps,
            "crps_mean": crps,
        }
        for level in (0.50, 0.80, 0.90, 0.95):
            key = f"{level:.2f}"
            covered = index < round(level * case_count)
            row[f"covered_primary_{key}"] = covered
            row[f"covered_secondary_{key}"] = covered
            row[f"covered_joint_{key}"] = covered
            row[f"width_primary_{key}"] = 1.0
            row[f"width_secondary_{key}"] = 1.0
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_learning_curve_decision_is_paired_and_has_a_forward_exit(tmp_path: Path) -> None:
    for rung, nlp, crps in ((16384, 1.0, 1.0), (32768, 0.995, 0.995)):
        for seed in (0, 1, 2):
            _write_learning_curve_cases(
                tmp_path / f"rung-{rung}" / f"seed-{seed}" / "development_cases.csv",
                nlp=nlp,
                crps=crps,
            )
    saturated = compare_16k_to_32k(tmp_path, bootstrap_replicates=100)
    assert saturated["decision"] == "lock_train_32k"
    assert saturated["all_saturation_conditions_passed"] is True
    _write_learning_curve_cases(
        tmp_path / "rung-32768" / "seed-2" / "development_cases.csv",
        nlp=0.95,
        crps=0.95,
    )
    improving = compare_16k_to_32k(tmp_path, bootstrap_replicates=100)
    assert improving["decision"] == "continue_to_train_65k"
    assert improving["all_saturation_conditions_passed"] is False


def test_committed_probe_decision_preserves_development_only_boundary() -> None:
    result = json.loads(
        (ROOT / "results/phase4/probe/learning_curve_decision.json").read_text()
    )
    summary = json.loads(
        (ROOT / "results/phase4/probe/probe_training_summary.json").read_text()
    )
    assert result["status"] == "learning_curve_decision_complete"
    assert result["decision"] == "continue_to_train_65k"
    assert result["all_saturation_conditions_passed"] is False
    assert result["validation_case_count"] == 6144
    assert result["paired_nlp_bootstrap"]["replicates"] == 10000
    assert result["paired_nlp_bootstrap"]["lower_95"] > 0.01
    assert result["calibration_accessed"] is False
    assert result["final_evaluation_accessed"] is False
    assert summary["stage_b_authorized_by_decision"] is False
    assert summary["calibration_accessed"] is False
    assert summary["final_evaluation_accessed"] is False
