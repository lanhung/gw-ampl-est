#!/usr/bin/env python3
"""Evaluate frozen 8-second waveform boundaries against 32-second references."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from gwlens_mm.config import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def _parameters(case: Mapping[str, Any]) -> Dict[str, float]:
    factor = 1.0 + float(case["source_redshift"])
    return {
        "mass_1": float(case["mass_1_source"]) * factor,
        "mass_2": float(case["mass_2_source"]) * factor,
        "luminosity_distance": 2000.0,
        "a_1": float(case["a_1"]),
        "tilt_1": float(case["tilt_1"]),
        "phi_12": 2.0,
        "a_2": float(case["a_2"]),
        "tilt_2": float(case["tilt_2"]),
        "phi_jl": 1.0,
        "theta_jn": 0.7,
        "phase": 0.2,
        "ra": 1.0,
        "dec": 0.3,
        "psi": 0.5,
    }


def _waveform(
    bilby: Any,
    gw_config: Mapping[str, Any],
    parameters: Mapping[str, float],
    duration: float,
    merger_offset: float,
) -> np.ndarray:
    sample_rate = int(gw_config["sample_rate_hz"])
    generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments={
            "waveform_approximant": gw_config["waveform"],
            "reference_frequency": float(gw_config["reference_frequency_hz"]),
            "minimum_frequency": float(gw_config["minimum_frequency_hz"]),
        },
    )
    polarizations = generator.frequency_domain_strain(parameters)
    interferometer = bilby.gw.detector.get_empty_interferometer("H1")
    geocent_time = 1126259462.0
    interferometer.set_strain_data_from_zero_noise(
        sample_rate,
        duration,
        start_time=geocent_time - merger_offset,
    )
    response = interferometer.get_detector_response(
        polarizations,
        {**parameters, "geocent_time": geocent_time},
        frequencies=generator.frequency_array,
    )
    return np.fft.irfft(response, n=int(duration * sample_rate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    gw = config["gw"]
    boundary = config["waveform_boundary"]
    bilby = importlib.import_module("bilby")
    cases = (
        {
            "id": "minimum_mass_sie",
            "lens_family": "sie_external_shear",
            "mass_1_source": 20.0,
            "mass_2_source": 10.0,
            "source_redshift": 0.5,
            "a_1": 0.0,
            "a_2": 0.0,
            "tilt_1": 0.0,
            "tilt_2": 0.0,
        },
        {
            "id": "minimum_mass_epl_precessing",
            "lens_family": "epl_external_shear",
            "mass_1_source": 20.0,
            "mass_2_source": 10.0,
            "source_redshift": 0.5,
            "a_1": 0.99,
            "a_2": 0.99,
            "tilt_1": 1.2,
            "tilt_2": 2.0,
        },
        {
            "id": "extreme_ratio_high_z",
            "lens_family": "sie_external_shear",
            "mass_1_source": 80.0,
            "mass_2_source": 20.0,
            "source_redshift": 3.0,
            "a_1": 0.9,
            "a_2": 0.8,
            "tilt_1": 0.2,
            "tilt_2": 2.5,
        },
        {
            "id": "maximum_mass_high_z",
            "lens_family": "epl_external_shear",
            "mass_1_source": 80.0,
            "mass_2_source": 80.0,
            "source_redshift": 3.0,
            "a_1": 0.99,
            "a_2": 0.99,
            "tilt_1": 1.0,
            "tilt_2": 2.0,
        },
    )
    rows = []
    sample_rate = int(gw["sample_rate_hz"])
    edge_samples = int(float(boundary["edge_guard_seconds"]) * sample_rate)
    for case in cases:
        parameters = _parameters(case)
        short = _waveform(
            bilby,
            gw,
            parameters,
            float(gw["duration_seconds"]),
            float(gw["merger_offset_seconds"]),
        )
        reference_duration = float(boundary["reference_duration_seconds"])
        reference = _waveform(
            bilby,
            gw,
            parameters,
            reference_duration,
            reference_duration - (
                float(gw["duration_seconds"]) - float(gw["merger_offset_seconds"])
            ),
        )[-len(short) :]
        reference_norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        total_energy = max(float(np.sum(short**2)), np.finfo(float).tiny)
        row = {
            **case,
            "finite": bool(np.all(np.isfinite(short)) and np.all(np.isfinite(reference))),
            "reference_relative_difference": float(np.linalg.norm(short - reference))
            / reference_norm,
            "leading_edge_energy_fraction": float(np.sum(short[:edge_samples] ** 2))
            / total_energy,
            "trailing_edge_energy_fraction": float(np.sum(short[-edge_samples:] ** 2))
            / total_energy,
        }
        row["passed"] = (
            row["finite"]
            and row["reference_relative_difference"]
            <= float(boundary["maximum_reference_relative_difference"])
            and row["leading_edge_energy_fraction"]
            <= float(boundary["maximum_edge_energy_fraction"])
            and row["trailing_edge_energy_fraction"]
            <= float(boundary["maximum_edge_energy_fraction"])
        )
        rows.append(row)
    result = {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "criteria": boundary,
        "cases": rows,
        "failure_count": sum(not row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
