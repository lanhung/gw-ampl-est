from __future__ import annotations

from pathlib import Path

from gwlens_mm.config import load_yaml
from gwlens_mm.production.waveform_correction import (
    CORRECTION_PREREGISTRATION_HASH,
    build_replacement_namespace_config,
    derive_waveform_correction_identity,
    load_waveform_correction_contract,
)
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/phase4_waveform_numerical_correction.yaml"


def test_waveform_correction_preregistration_and_audit_are_frozen() -> None:
    config = load_waveform_correction_contract(ROOT)
    preregistration = load_yaml(ROOT / config["preregistration"]["path"])
    assert configuration_hash(preregistration) == CORRECTION_PREREGISTRATION_HASH
    assert preregistration["preregistration_version"] == "1.1.1-rc.1"
    assert preregistration["scientific_invariants"]["posterior_target_changed"] is False
    assert preregistration["diagnostic_basis"]["optimizer_steps_observed_before_freeze"] == 0


def test_waveform_correction_is_exactly_two_plus_three_replacements() -> None:
    config = load_waveform_correction_contract(ROOT)
    stage_a = config["replacement_namespaces"]["stage_a_train"]
    stage_b = config["replacement_namespaces"]["stage_b_train"]
    assert (stage_a["accepted_pair_count"], stage_b["accepted_pair_count"]) == (2, 3)
    assert len(stage_a["excluded_physical_system_ids"]) == 2
    assert len(stage_b["excluded_physical_system_ids"]) == 3
    assert config["corrected_counts"]["combined_train"] == 65536
    assert config["corrected_counts"]["stage_a_validation"] == 6144


def test_replacement_namespaces_enable_frozen_numerical_validity() -> None:
    config = load_waveform_correction_contract(ROOT)
    for component, count in (("stage_a_train", 2), ("stage_b_train", 3)):
        namespace = build_replacement_namespace_config(ROOT, config, component)
        assert namespace["accepted_pair_count"] == count
        assert namespace["production_context"]["proposal_mode"] == (
            "evaluation_target_direct"
        )
        validity = namespace["gw"]["source_polarization_numerical_validity"]
        assert validity == {
            "enabled": True,
            "minimum_frequency_hz": 20.0,
            "positive_amplitude_quantile": 0.999,
            "maximum_peak_to_quantile_ratio": 10.0,
        }


def test_correction_identities_are_deterministic_fresh_and_distinct() -> None:
    config = load_waveform_correction_contract(ROOT)
    first = derive_waveform_correction_identity(config, "a" * 40)
    second = derive_waveform_correction_identity(config, "a" * 40)
    assert first == second
    assert first.stage_a_replacement_dataset_id != first.stage_b_replacement_dataset_id
    assert "waveform-correction" in first.parent_run_id


def test_correction_execution_and_every_downstream_gate_are_closed() -> None:
    config = load_waveform_correction_contract(ROOT)
    assert config["execution"]["replacement_materialization_enabled"] is False
    assert config["execution"]["corrected_view_publication_enabled"] is False
    assert all(value is False for value in config["authorization_boundaries"].values())
    authorization = load_yaml(ROOT / config["authorization_path"])
    assert authorization["implementation_commit"] is None
    assert authorization["authorization"]["replacement_materialization_authorized"] is False
    assert authorization["authorization"]["model_training_authorized"] is False
