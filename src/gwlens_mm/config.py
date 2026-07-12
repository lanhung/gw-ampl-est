"""Validation for the metadata-only Phase 1B smoke specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .physics.quantities import LensFamily


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def validate_smoke_configuration(config: Mapping[str, Any]) -> None:
    if config.get("execution_authorized") is not False:
        raise ValueError("Phase 1A smoke specification must not authorize execution")
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
    if config["count_control"] != "accepted_count":
        raise ValueError("smoke generation must use accepted-count control")
    if config["output_root"] != "/root/autodl-tmp/lensing-4/data_v2/smoke":
        raise ValueError("smoke output root is outside the approved AutoDL project")
