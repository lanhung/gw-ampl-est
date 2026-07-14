from __future__ import annotations

import hashlib
import json
import math
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
    build_final_evaluation_namespace_config,
    collect_published_group_identifiers,
    dry_run_plan,
    final_evaluation_namespaces,
    load_final_evaluation_contract,
    validate_final_evaluation_record,
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
    psd = next(item for item in first if item.truth_psd_curves is not None)
    psd_config = build_final_evaluation_namespace_config(ROOT, config, psd)
    assert psd_config["gw"]["psd_curves"]["H1"]["file"] == (
        "aLIGO_ZERO_DET_high_P_psd.txt"
    )


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


def test_psd_constructor_semantics_and_commitment_are_fail_closed() -> None:
    assert psd_file_keyword("aLIGO_O4_high_asd.txt") == "asd_file"
    assert psd_file_keyword("AdV_psd.txt") == "psd_file"
    with pytest.raises(ValueError, match="ASD or PSD"):
        psd_file_keyword("unknown.txt")
    commitment = json.loads(
        (ROOT / "results/phase4/final_evaluation_commitment.json").read_text()
    )
    assert commitment["commitment_status"] == "unfinalized_design_template"
    assert commitment["future_scientific_generator_commit"] is None


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
