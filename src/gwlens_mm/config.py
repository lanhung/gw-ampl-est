"""Validation for the metadata-only Phase 1B smoke specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .physics.quantities import LensFamily
from .schema import FROZEN_SMOKE_SCHEMA_VERSION


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def validate_smoke_configuration(
    config: Mapping[str, Any], *, expected_execution_authorized: bool
) -> None:
    if config.get("schema_version") != FROZEN_SMOKE_SCHEMA_VERSION:
        raise ValueError(
            f"smoke specification must use frozen schema {FROZEN_SMOKE_SCHEMA_VERSION}"
        )
    if config.get("execution_authorized") is not expected_execution_authorized:
        raise ValueError(
            "smoke execution authorization does not match the current phase gate"
        )
    accepted = config["accepted_pairs"]
    expected = {
        LensFamily.SIS.value: 24,
        LensFamily.SIE_EXTERNAL_SHEAR.value: 12,
        LensFamily.EPL_EXTERNAL_SHEAR.value: 12,
    }
    if accepted != expected or sum(accepted.values()) != 48:
        raise ValueError("smoke specification must contain the reviewed 24/12/12 composition")
    gw = config["gw"]
    if gw["selected_image_slots"] != 2 or tuple(gw["detector_slots"]) != ("H1", "L1", "V1"):
        raise ValueError("smoke GW axes must be two images by H1/L1/V1")
    if gw["duration_seconds"] != 1 or gw["sample_rate_hz"] != 4096 or gw["dtype"] != "float32":
        raise ValueError("smoke arrays must be one-second 4096-Hz float32")
    required_products = {"noisy_strain", "clean_injected_signal", "noise_realization"}
    if set(gw["separate_products"]) != required_products:
        raise ValueError("smoke arrays must preserve noisy, clean, and noise separately")
    if gw.get("unavailable_detector_fill_value") != 0.0:
        raise ValueError("unavailable detector slots must be zero-filled")
    if gw.get("require_noisy_equals_clean_plus_noise") is not True:
        raise ValueError("smoke validation must require noisy = clean + noise")
    timing = gw.get("timing_observation_model", {})
    if not timing.get("measurement_method"):
        raise ValueError("smoke timing observation requires a measurement method")
    if timing.get("deterministic_control") is not False:
        raise ValueError("smoke timing observation is not a deterministic control")
    if (
        not isinstance(timing.get("standard_deviation_seconds"), (int, float))
        or timing["standard_deviation_seconds"] <= 0
    ):
        raise ValueError("smoke timing observation requires positive uncertainty")
    if config["count_control"] != "accepted_count":
        raise ValueError("smoke generation must use accepted-count control")
    modalities = set(config["em"]["modalities"])
    if "image_astrometry" not in modalities or "image_positions" in modalities:
        raise ValueError("smoke EM astrometry must be image-ID keyed")
    if config["output_root"] != "/root/autodl-tmp/lensing-4/data_v2/smoke":
        raise ValueError("smoke output root is outside the approved AutoDL project")
