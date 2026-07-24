from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.physics.quantities import LensFamily
from gwlens_mm.production.diagnostic_context import (
    BalancedTailStratum,
    classify_balanced_tail,
)
from gwlens_mm.production.final_evaluation import (
    FINAL_EVALUATION_COMMITMENT_HASH,
    NUMERICAL_VALIDITY_ADDENDUM_HASH,
    ORIGINAL_COMMITTED_GENERATOR,
    build_final_evaluation_namespace_config,
    collect_published_group_identifiers,
    derive_final_evaluation_identities,
    dry_run_plan,
    final_evaluation_namespaces,
    load_final_evaluation_contract,
    resolve_bound_published_reference_dataset,
    validate_final_evaluation_record,
    validate_future_final_evaluation_authorization,
)
from gwlens_mm.production.gw import psd_file_keyword
from gwlens_mm.production.proposal_adapter import sample_production_proposal
from gwlens_mm.proposals.diagnostic import (
    ParameterOODStratum,
    log_parameter_ood_density,
    sample_parameter_ood,
)
from gwlens_mm.proposals.target_anchored import TargetAnchoredSpecification
from gwlens_mm.provenance import configuration_hash
from gwlens_mm.schema import V2Record

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Image:
    mu_signed: float


def _proposal() -> tuple[dict, TargetAnchoredSpecification]:
    config = load_yaml(ROOT / "configs/proposals/proposal_v3_target_anchored_mixture.yaml")
    return config, TargetAnchoredSpecification.from_mapping(config)


def test_final_evaluation_gate_and_dry_plan_remain_execution_disabled() -> None:
    config, authorization = load_final_evaluation_contract(ROOT)
    assert authorization["authorization"]["waveform_pair_generation_authorized"] is False
    assert authorization["authorization"]["scientific_probe_training_authorized"] is False
    assert config["execution"]["enabled"] is False
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_execution_blocked"
    assert plan["accepted_count"] == 20480
    assert plan["shard_count"] == 160
    assert plan["namespace_count"] == 15
    assert plan["waveform_pairs_generated"] == 0


def test_namespace_expansion_is_exact_disjoint_and_generator_ready() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    first = final_evaluation_namespaces(config)
    second = final_evaluation_namespaces(config)
    assert first == second
    assert len({item.namespace_id for item in first}) == 15
    assert len({item.root_seed for item in first}) == 15
    assert sum(item.accepted_count for item in first) == 20480
    assert sum(item.shard_count for item in first) == 160
    for namespace in first:
        generated = build_final_evaluation_namespace_config(ROOT, config, namespace)
        assert generated["accepted_pair_count"] == namespace.accepted_count
        assert generated["shard_count"] == namespace.shard_count
        assert generated["production_context"]["split"] == namespace.split.value
        assert generated["production_context"]["diagnostic_context_id"]
        assert generated["execution"]["attempt_id_stride"] == 512
    waveform = next(item for item in first if item.truth_waveform is not None)
    waveform_config = build_final_evaluation_namespace_config(ROOT, config, waveform)
    assert waveform_config["gw"]["waveform"] == "SEOBNRv4PHM"
    assert "source_polarization_numerical_validity" not in waveform_config["gw"]
    assert waveform_config["production_context"][
        "source_polarization_numerical_validity"
    ] == "not_applicable_alternate_waveform"
    assert waveform_config["production_context"][
        "alternate_waveform_finite_array_validation_required"
    ] is True
    psd = next(item for item in first if item.truth_psd_curves is not None)
    psd_config = build_final_evaluation_namespace_config(ROOT, config, psd)
    assert psd_config["gw"]["psd_curves"]["H1"]["file"] == (
        "aLIGO_ZERO_DET_high_P_psd.txt"
    )
    for namespace in first:
        if namespace.truth_waveform is not None:
            continue
        generated = build_final_evaluation_namespace_config(ROOT, config, namespace)
        assert generated["gw"]["source_polarization_numerical_validity"] == {
            "enabled": True,
            "minimum_frequency_hz": 20.0,
            "positive_amplitude_quantile": 0.999,
            "maximum_peak_to_quantile_ratio": 10.0,
        }
        assert generated["production_context"][
            "source_polarization_numerical_validity"
        ] == "applied_before_selection"


def test_real_alpha3_record_uses_typed_final_evaluation_health_path() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    namespace = final_evaluation_namespaces(config)[0]
    data = json.loads((ROOT / "examples/v2_metadata_example.json").read_text())
    data["pair"]["split"] = namespace.split.value
    data["pair"]["dataset_version"] = "sealed-iid-fixture"
    data["pair"]["proposal_distribution_id"] = namespace.proposal_distribution_id
    data["pair"]["evaluation_prior_id"] = namespace.evaluation_distribution_id
    record = V2Record.from_dict(data)
    validate_final_evaluation_record(
        record, namespace, expected_dataset="sealed-iid-fixture"
    )


@pytest.mark.parametrize(
    ("mode", "family"),
    [
        ("evaluation_target_family_sie", LensFamily.SIE_EXTERNAL_SHEAR),
        ("evaluation_target_family_epl", LensFamily.EPL_EXTERNAL_SHEAR),
    ],
)
def test_conditioned_family_production_draws_are_exact_unit_weight(
    mode: str, family: LensFamily
) -> None:
    proposal, _ = _proposal()
    rng = np.random.default_rng(20260714)
    for _ in range(64):
        result = sample_production_proposal(rng, mode=mode, proposal_config=proposal)
        assert result.population.lens_family is family
        assert result.population.proposal_log_probability == (
            result.population.evaluation_log_probability
        )
        assert result.population.importance_weight == 1.0


@pytest.mark.parametrize("stratum", tuple(ParameterOODStratum))
def test_parameter_ood_sampler_density_and_production_adapter_agree(
    stratum: ParameterOODStratum,
) -> None:
    proposal, specification = _proposal()
    rng = np.random.default_rng(44)
    for _ in range(64):
        draw, log_density = sample_parameter_ood(rng, specification, stratum)
        assert math.isfinite(log_density)
        if stratum is ParameterOODStratum.SLOPE_OUTSIDE_TRAINING:
            assert draw.lens_family is LensFamily.EPL_EXTERNAL_SHEAR
            assert 1.4 <= draw.density_slope < 1.6 or 2.5 < draw.density_slope <= 2.7
        elif stratum is ParameterOODStratum.EXTREME_FLATTENING:
            assert 0.25 <= draw.axis_ratio < 0.4
        elif stratum is ParameterOODStratum.HIGH_EXTERNAL_SHEAR:
            assert 0.15 < draw.shear_amplitude <= 0.25
        else:
            assert 0.15 < abs(draw.external_convergence) <= 0.25
    adapted = sample_production_proposal(
        np.random.default_rng(55),
        mode=f"parameter_ood_{stratum.value}",
        proposal_config=proposal,
    )
    assert adapted.population.proposal_log_probability == (
        adapted.population.evaluation_log_probability
    )
    assert adapted.population.importance_weight == 1.0


def test_parameter_ood_endpoint_semantics_match_frozen_parent() -> None:
    _, specification = _proposal()
    draw, _ = sample_parameter_ood(
        np.random.default_rng(9),
        specification,
        ParameterOODStratum.HIGH_EXTERNAL_SHEAR,
    )
    assert math.isinf(
        log_parameter_ood_density(
            replace(draw, shear_amplitude=0.15),
            specification,
            ParameterOODStratum.HIGH_EXTERNAL_SHEAR,
        )
    )
    assert math.isfinite(
        log_parameter_ood_density(
            replace(draw, shear_amplitude=0.25),
            specification,
            ParameterOODStratum.HIGH_EXTERNAL_SHEAR,
        )
    )


def test_balanced_tail_priority_is_frozen() -> None:
    assert classify_balanced_tail(
        (_Image(25.0), _Image(1.0)),
        secondary_network_snr=10.5,
        external_convergence=0.2,
        density_slope=2.6,
    ) is BalancedTailStratum.HIGH_ABSOLUTE_MAGNIFICATION
    assert classify_balanced_tail(
        (_Image(9.0), _Image(0.9)),
        secondary_network_snr=10.5,
        external_convergence=0.2,
        density_slope=2.6,
    ) is BalancedTailStratum.EXTREME_RELATIVE_MAGNIFICATION
    assert classify_balanced_tail(
        (_Image(9.0), _Image(1.0)),
        secondary_network_snr=10.5,
        external_convergence=0.2,
        density_slope=2.6,
    ) is BalancedTailStratum.SECOND_IMAGE_NEAR_THRESHOLD
    assert classify_balanced_tail(
        (_Image(9.0), _Image(1.0)),
        secondary_network_snr=13.0,
        external_convergence=0.2,
        density_slope=2.0,
    ) is BalancedTailStratum.EXTREME_PROFILE_OR_ENVIRONMENT
    assert classify_balanced_tail(
        (_Image(9.0), _Image(1.0)),
        secondary_network_snr=12.0,
        external_convergence=0.0,
        density_slope=2.0,
    ) is None


def test_psd_constructor_semantics_and_commitment_are_fail_closed() -> None:
    assert psd_file_keyword("aLIGO_O4_high_asd.txt") == "asd_file"
    assert psd_file_keyword("AdV_psd.txt") == "psd_file"
    with pytest.raises(ValueError, match="ASD or PSD"):
        psd_file_keyword("unknown.txt")
    commitment = json.loads(
        (ROOT / "results/phase4/final_evaluation_commitment.json").read_text()
    )
    assert commitment["commitment_status"] == "finalized_before_training"
    assert commitment["future_scientific_generator_commit"] == (
        "bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac"
    )


def test_waveform_engine_loads_hashes_and_assigns_declared_curve_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwlens_mm.production import gw as gw_module

    curve_root = tmp_path / "bilby" / "gw" / "detector" / "noise_curves"
    curve_root.mkdir(parents=True)
    asd = curve_root / "test_asd.txt"
    psd = curve_root / "test_psd.txt"
    asd.write_bytes(b"asd-curve")
    psd.write_bytes(b"psd-curve")
    constructed: list[dict[str, str]] = []

    class _PSD:
        def __init__(self, **kwargs: str) -> None:
            constructed.append(kwargs)

    class _IFO:
        power_spectral_density: object | None = None

    detector = SimpleNamespace(
        PowerSpectralDensity=_PSD,
        get_empty_interferometer=lambda name: _IFO(),
    )
    fake_bilby = SimpleNamespace(
        __file__=str(tmp_path / "bilby" / "__init__.py"),
        gw=SimpleNamespace(
            detector=detector,
            source=SimpleNamespace(lal_binary_black_hole=object()),
            WaveformGenerator=lambda **kwargs: object(),
        ),
    )
    monkeypatch.setattr(
        gw_module.importlib,
        "import_module",
        lambda name: fake_bilby if name == "bilby" else __import__(name),
    )
    config = deepcopy(load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")["gw"])
    config["psd_curves"] = {
        "H1": {"file": asd.name, "sha256": hashlib.sha256(asd.read_bytes()).hexdigest()},
        "L1": {"file": asd.name, "sha256": hashlib.sha256(asd.read_bytes()).hexdigest()},
        "V1": {"file": psd.name, "sha256": hashlib.sha256(psd.read_bytes()).hexdigest()},
    }
    engine = gw_module.ProductionWaveformEngine(config, 7)
    assert constructed == [
        {"asd_file": str(asd)},
        {"asd_file": str(asd)},
        {"psd_file": str(psd)},
    ]
    assert engine._interferometer("H1").power_spectral_density is not None


def test_final_evaluation_config_hash_and_resource_caps_are_frozen() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    assert configuration_hash(config) == (
        "11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66"
    )
    assert config["resource_gates"] == {
        "minimum_prelaunch_free_bytes": 145000000000,
        "minimum_post_publication_free_bytes": 100000000000,
        "maximum_published_bytes": 30000000000,
    }
    assert config["execution"]["maximum_attempts_per_worker"] == 20000000


def test_finalized_commitment_matches_every_deterministic_namespace() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    namespaces = final_evaluation_namespaces(config)
    commitment = json.loads(
        (ROOT / "results/phase4/final_evaluation_commitment.json").read_text()
    )
    committed = commitment["namespace_commitments"]
    assert set(committed) == {item.namespace_id for item in namespaces}
    for namespace in namespaces:
        item = committed[namespace.namespace_id]
        assert item["root_seed"] == namespace.root_seed
        assert item["accepted_count"] == namespace.accepted_count
        assert item["shard_count"] == namespace.shard_count
    assert commitment["attempt_stream_allocation"]["attempt_id_stride"] == 512
    assert commitment["accepted_rank_allocation_rule"]["namespace_scoped"] is True
    assert all(value is False for value in commitment["use_policy"].values())


def test_future_official_identities_are_namespace_specific() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    identities = derive_final_evaluation_identities(ROOT, config, "a" * 40)
    assert identities.parent_run_id.startswith("phase7-final-evaluation-")
    assert len(identities.namespace_dataset_ids) == 15
    assert len(set(identities.namespace_dataset_ids.values())) == 15


def test_atomic_parent_manifest_binds_child_reference_dataset(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "project"
    parent = approved / "published" / "parent"
    child = parent / "train-dataset"
    shard = child / "shards" / "shard-00000"
    shard.mkdir(parents=True)
    (shard / "records.parquet").write_bytes(b"fixture")
    manifest = parent / "dataset_manifest.json"
    manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    specification = {
        "dataset_id": child.name,
        "dataset_root": str(child),
        "parent_root": str(parent),
        "parent_manifest_sha256": digest,
    }
    resolved = resolve_bound_published_reference_dataset(
        specification,
        approved_root=approved,
    )
    assert resolved.dataset_root == child
    assert resolved.parent_manifest_path == manifest

    changed = deepcopy(specification)
    changed["parent_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="atomic parent"):
        resolve_bound_published_reference_dataset(
            changed,
            approved_root=approved,
        )
    changed = deepcopy(specification)
    changed["dataset_root"] = str(parent)
    with pytest.raises(ValueError, match="atomic parent"):
        resolve_bound_published_reference_dataset(
            changed,
            approved_root=approved,
        )


def test_bound_child_reference_streams_records_without_fake_child_manifest(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    approved = tmp_path / "project"
    parent = approved / "published" / "parent"
    child = parent / "train-dataset"
    shard = child / "shards" / "shard-00000"
    shard.mkdir(parents=True)
    record_json = (ROOT / "examples/v2_metadata_example.json").read_text(
        encoding="utf-8"
    )
    pandas.DataFrame({"record_json": [record_json]}).to_parquet(
        shard / "records.parquet",
        index=False,
    )
    manifest = parent / "dataset_manifest.json"
    manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
    specification = {
        "dataset_id": child.name,
        "dataset_root": str(child),
        "parent_root": str(parent),
        "parent_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    bound = resolve_bound_published_reference_dataset(
        specification,
        approved_root=approved,
    )
    identifiers = collect_published_group_identifiers(
        (bound.dataset_root,),
        require_root_manifest=False,
    )
    assert len(identifiers["system"]) == 1
    with pytest.raises(ValueError, match="manifest is absent"):
        collect_published_group_identifiers((bound.dataset_root,))


def test_numerical_validity_addendum_preserves_original_commitment() -> None:
    commitment_path = ROOT / "results/phase4/final_evaluation_commitment.json"
    addendum_path = ROOT / (
        "results/phase4/final_evaluation_numerical_validity_addendum.json"
    )
    assert hashlib.sha256(commitment_path.read_bytes()).hexdigest() == (
        "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"
    )
    assert hashlib.sha256(addendum_path.read_bytes()).hexdigest() == (
        "431c09f2c279e1c745bd118fb1b0c06643de7dc42f605af78a49ca99b5b0019b"
    )
    addendum = json.loads(addendum_path.read_text())
    assert addendum["original_commitment_mutated"] is False
    assert addendum["baseline_waveform_namespace_count"] == 14
    assert addendum["alternate_waveform_namespace"]["namespace_id"] == (
        "waveform_mismatch_test/seobnrv4phm_truth"
    )
    assert all(value is False for value in addendum["use_policy"].values())


def test_future_materialization_binds_narrow_generator_revision() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    generator_commit = "a" * 40
    authorization = {
        "authorization_status": (
            "authorized_sealed_final_evaluation_materialization_only"
        ),
        "immutable_generator": {"git_commit": generator_commit},
        "frozen_contract": {
            "configuration_hash": configuration_hash(config),
            "commitment_sha256": FINAL_EVALUATION_COMMITMENT_HASH,
            "numerical_validity_addendum_sha256": (
                NUMERICAL_VALIDITY_ADDENDUM_HASH
            ),
            "waveform_numerical_validity_preregistration_hash": (
                "7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69"
            ),
        },
        "prospective_generator_revision": {
            "original_committed_generator": ORIGINAL_COMMITTED_GENERATOR,
            "scope": "waveform_numerical_validity_implementation_only",
            "counts_seeds_distributions_changed": False,
            "original_commitment_mutated": False,
        },
        "materialization_contract": {
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "training_size_and_architecture_locked": True,
        },
        "published_reference_contract": {
            "corrected_combined_train_manifest_sha256": "b" * 64,
            "correction_parent_manifest_sha256": (
                "0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2"
            ),
            "correction_publication_tree_sha256": (
                "a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12"
            ),
            "logical_system_counts": {
                "train": 65536,
                "validation": 6144,
                "calibration_fit": 4096,
                "sbc_diagnostic": 2048,
            },
        },
        "authorization": {
            "sealed_materialization_authorized": True,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
            "model_training_authorized": False,
            "calibration_fit_authorized": False,
            "learning_curve_use_authorized": False,
            "architecture_selection_use_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    }
    validate_future_final_evaluation_authorization(
        authorization,
        config=config,
        generator_commit=generator_commit,
        commitment_sha256=FINAL_EVALUATION_COMMITMENT_HASH,
        numerical_validity_addendum_sha256=NUMERICAL_VALIDITY_ADDENDUM_HASH,
    )
    changed = deepcopy(authorization)
    changed["prospective_generator_revision"][
        "counts_seeds_distributions_changed"
    ] = True
    with pytest.raises(ValueError, match="narrowly bound"):
        validate_future_final_evaluation_authorization(
            changed,
            config=config,
            generator_commit=generator_commit,
            commitment_sha256=FINAL_EVALUATION_COMMITMENT_HASH,
            numerical_validity_addendum_sha256=NUMERICAL_VALIDITY_ADDENDUM_HASH,
        )


def test_future_final_materialization_accepts_only_exact_terminal_reference() -> None:
    config, _ = load_final_evaluation_contract(ROOT)
    generator_commit = "a" * 40
    authorization = {
        "authorization_status": (
            "authorized_sealed_final_evaluation_materialization_only"
        ),
        "immutable_generator": {"git_commit": generator_commit},
        "frozen_contract": {
            "configuration_hash": configuration_hash(config),
            "commitment_sha256": FINAL_EVALUATION_COMMITMENT_HASH,
            "numerical_validity_addendum_sha256": NUMERICAL_VALIDITY_ADDENDUM_HASH,
            "waveform_numerical_validity_preregistration_hash": (
                "7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69"
            ),
        },
        "prospective_generator_revision": {
            "original_committed_generator": ORIGINAL_COMMITTED_GENERATOR,
            "scope": "waveform_numerical_validity_implementation_only",
            "counts_seeds_distributions_changed": False,
            "original_commitment_mutated": False,
        },
        "materialization_contract": {
            "accepted_pair_count": 20480,
            "shard_count": 160,
            "namespace_count": 15,
            "training_size_and_architecture_locked": True,
        },
        "published_reference_contract": {
            "training_reference_mode": "terminal_131k",
            "corrected_combined_train_manifest_sha256": "1" * 64,
            "correction_parent_manifest_sha256": (
                "0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2"
            ),
            "correction_publication_tree_sha256": (
                "a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12"
            ),
            "terminal_preregistration_hash": (
                "77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a"
            ),
            "terminal_combined_train_manifest_sha256": "2" * 64,
            "terminal_train_increment_parent_manifest_sha256": "3" * 64,
            "development_tail_manifest_sha256": "4" * 64,
            "validation_manifest_sha256": "5" * 64,
            "strict_corrected_65k_subset": True,
            "development_tail_excluded_from_final_reference": True,
            "extension_above_131072_authorized": False,
            "terminal_size_decision": "lock_train_131k_saturated",
            "terminal_size_decision_sha256": "6" * 64,
            "selected_architecture_locked_rung": 131072,
            "selected_architecture_decision_sha256": "7" * 64,
            "logical_system_counts": {
                "train": 131072,
                "validation": 6144,
                "calibration_fit": 4096,
                "sbc_diagnostic": 2048,
            },
        },
        "authorization": {
            "sealed_materialization_authorized": True,
            "unsealing_authorized": False,
            "scientific_analysis_authorized": False,
            "model_training_authorized": False,
            "calibration_fit_authorized": False,
            "learning_curve_use_authorized": False,
            "architecture_selection_use_authorized": False,
            "gwosc_gwtc_access_authorized": False,
        },
    }
    validate_future_final_evaluation_authorization(
        authorization,
        config=config,
        generator_commit=generator_commit,
        commitment_sha256=FINAL_EVALUATION_COMMITMENT_HASH,
        numerical_validity_addendum_sha256=NUMERICAL_VALIDITY_ADDENDUM_HASH,
    )
    changed = deepcopy(authorization)
    changed["published_reference_contract"]["terminal_size_decision"] = (
        "lock_train_65k"
    )
    with pytest.raises(ValueError, match="published-reference"):
        validate_future_final_evaluation_authorization(
            changed,
            config=config,
            generator_commit=generator_commit,
            commitment_sha256=FINAL_EVALUATION_COMMITMENT_HASH,
            numerical_validity_addendum_sha256=NUMERICAL_VALIDITY_ADDENDUM_HASH,
        )


def test_real_final_release_gate_stays_blocked_without_exact_authorization(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase4/prepare_final_evaluation.py"),
            "--root",
            str(ROOT),
            "--execute",
            "--authorization",
            str(
                ROOT
                / (
                    "configs/execution/"
                    "phase4_final_evaluation_generator_implementation_authorization.yaml"
                )
            ),
            "--generator-commit",
            "a" * 40,
            "--commitment",
            str(ROOT / "results/phase4/final_evaluation_commitment.json"),
            "--numerical-validity-addendum",
            str(
                ROOT
                / "results/phase4/final_evaluation_numerical_validity_addendum.json"
            ),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    blocked = json.loads(output.read_text())
    assert blocked["status"] == "blocked_preexecution"
    assert blocked["official_identities"] is None


def test_published_reference_group_ids_are_streamed_and_duplicates_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwlens_mm.production import final_evaluation as final_evaluation_module

    data = json.loads((ROOT / "examples/v2_metadata_example.json").read_text())
    record_json = json.dumps(data)

    class _Frame:
        def __getitem__(self, key: str) -> list[str]:
            assert key == "record_json"
            return [record_json]

    fake_pandas = SimpleNamespace(read_parquet=lambda path, columns: _Frame())
    monkeypatch.setattr(
        final_evaluation_module.importlib,
        "import_module",
        lambda name: fake_pandas if name == "pandas" else __import__(name),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "dataset_manifest.json").write_text('{"status":"passed"}\n')
        (root / "records.parquet").write_bytes(b"fixture")
    identifiers = collect_published_group_identifiers((first,))
    assert identifiers["pair"] == {data["pair"]["pair_id"]}
    assert len(identifiers["noise"]) == len(
        V2Record.from_dict(data).provenance.used_noise_segment_ids
    )
    with pytest.raises(ValueError, match="duplicate pair"):
        collect_published_group_identifiers((first, second))
    system_id = data["pair"]["physical_system_id"]
    excluded = collect_published_group_identifiers(
        (first,), excluded_physical_system_ids=(system_id,)
    )
    assert all(not values for values in excluded.values())
    with pytest.raises(ValueError, match="exclusions"):
        collect_published_group_identifiers(
            (first,), excluded_physical_system_ids=("not-present",)
        )
