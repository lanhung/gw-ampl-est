#!/usr/bin/env python3
"""Evaluate the frozen RC.5 waveform construction and numerical reference."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.production.gw import (
    detector_frame_newtonian_chirp_time_seconds,
    raised_cosine_guard_window,
)

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


def _response(
    bilby: Any,
    gw_config: Mapping[str, Any],
    parameters: Mapping[str, float],
    case: Mapping[str, Any],
    duration: float,
    merger_offset: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
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
    lens_factor = math.sqrt(abs(float(case["mu_signed"]))) * np.exp(
        -1j * np.pi * float(case["morse_half_index"])
    )
    lensed = {name: values * lens_factor for name, values in polarizations.items()}
    interferometer = bilby.gw.detector.get_empty_interferometer("H1")
    geocent_time = 1126259462.0 + float(case["delay_seconds"])
    interferometer.set_strain_data_from_zero_noise(
        sample_rate,
        duration,
        start_time=geocent_time - merger_offset,
    )
    frequency_domain = np.asarray(
        interferometer.get_detector_response(
            lensed,
            {**parameters, "geocent_time": geocent_time},
            frequencies=generator.frequency_array,
        )
    )
    time_domain = np.asarray(
        bilby.core.utils.infft(frequency_domain, sample_rate), dtype=np.float64
    )
    expected_time_domain = np.fft.irfft(frequency_domain) * sample_rate
    denominator = max(
        float(np.linalg.norm(expected_time_domain)), np.finfo(float).tiny
    )
    normalization_difference = (
        float(np.linalg.norm(time_domain - expected_time_domain)) / denominator
    )
    return frequency_domain, time_domain, normalization_difference


def _condition(
    time_domain: np.ndarray,
    sample_count: int,
    window: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    crop = np.asarray(time_domain[-sample_count:], dtype=np.float64)
    conditioned = np.asarray(crop * window, dtype=np.float32)
    return crop, conditioned


def _selection_snr(
    bilby: Any,
    clean: np.ndarray,
    sample_rate: int,
    duration: float,
    segment_start: float,
) -> float:
    interferometer = bilby.gw.detector.get_empty_interferometer("H1")
    interferometer.set_strain_data_from_zero_noise(
        sample_rate,
        duration,
        start_time=segment_start,
    )
    frequency_domain, frequencies = bilby.core.utils.nfft(
        np.asarray(clean, dtype=np.float64), sample_rate
    )
    if not np.array_equal(frequencies, interferometer.frequency_array):
        raise ValueError("selection-SNR frequency grid mismatch")
    value = complex(interferometer.optimal_snr_squared(frequency_domain))
    return math.sqrt(max(float(value.real), 0.0))


def _cases() -> Tuple[Dict[str, Any], ...]:
    return (
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
            "mu_signed": 2.0,
            "morse_half_index": 0.0,
            "delay_seconds": 0.0,
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
            "mu_signed": -5.0,
            "morse_half_index": 0.5,
            "delay_seconds": 100000.0,
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
            "mu_signed": 25.0,
            "morse_half_index": 0.0,
            "delay_seconds": 500000.0,
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
            "mu_signed": -100.0,
            "morse_half_index": 1.0,
            "delay_seconds": 1000000.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    gw = config["gw"]
    boundary = config["waveform_boundary"]
    bilby = importlib.import_module("bilby")
    sample_rate = int(gw["sample_rate_hz"])
    sample_count = int(gw["sample_count"])
    guard = int(gw["zero_guard_samples_per_edge"])
    window = raised_cosine_guard_window(
        sample_count,
        guard,
        int(gw["raised_cosine_transition_samples_per_edge"]),
    )
    rows = []
    for case in _cases():
        parameters = _parameters(case)
        construction_fd, construction_full, construction_roundtrip = _response(
            bilby,
            gw,
            parameters,
            case,
            float(gw["construction_duration_seconds"]),
            float(gw["construction_merger_offset_seconds"]),
        )
        reference_fd, reference_full, reference_roundtrip = _response(
            bilby,
            gw,
            parameters,
            case,
            float(boundary["reference_duration_seconds"]),
            float(boundary["reference_merger_offset_seconds"]),
        )
        construction_crop, construction = _condition(
            construction_full, sample_count, window
        )
        _, reference = _condition(reference_full, sample_count, window)
        reference_norm = max(
            float(np.linalg.norm(reference.astype(np.float64))), np.finfo(float).tiny
        )
        construction_energy = max(
            float(np.sum(construction_full**2)), np.finfo(float).tiny
        )
        crop_energy = max(float(np.sum(construction_crop**2)), np.finfo(float).tiny)
        relative_difference = float(
            np.linalg.norm(
                construction.astype(np.float64) - reference.astype(np.float64)
            )
        ) / reference_norm
        outside_crop_fraction = (
            float(np.sum(construction_full[:-sample_count] ** 2)) / construction_energy
        )
        retained_fraction = float(np.sum(construction.astype(np.float64) ** 2)) / crop_energy
        leading_guard_energy = float(np.sum(construction[:guard] ** 2))
        trailing_guard_energy = float(np.sum(construction[-guard:] ** 2))
        geocent_time = 1126259462.0 + float(case["delay_seconds"])
        segment_start = geocent_time - float(gw["merger_offset_seconds"])
        selection_snr = _selection_snr(
            bilby,
            construction,
            sample_rate,
            float(gw["duration_seconds"]),
            segment_start,
        )
        stored = construction.astype(np.float32)
        stored_snr = _selection_snr(
            bilby,
            stored,
            sample_rate,
            float(gw["duration_seconds"]),
            segment_start,
        )
        snr_denominator = max(abs(selection_snr), np.finfo(float).tiny)
        snr_relative_difference = abs(selection_snr - stored_snr) / snr_denominator
        chirp_time = detector_frame_newtonian_chirp_time_seconds(
            float(parameters["mass_1"]),
            float(parameters["mass_2"]),
            float(gw["minimum_frequency_hz"]),
        )
        untapered_premerger_seconds = float(gw["merger_offset_seconds"]) - (
            guard + int(gw["raised_cosine_transition_samples_per_edge"])
        ) / sample_rate
        finite = bool(
            np.all(np.isfinite(construction_fd))
            and np.all(np.isfinite(reference_fd))
            and np.all(np.isfinite(construction_full))
            and np.all(np.isfinite(reference_full))
            and np.all(np.isfinite(construction))
            and np.all(np.isfinite(reference))
        )
        row = {
            **case,
            "detector_frame_newtonian_chirp_time_seconds": chirp_time,
            "untapered_premerger_support_seconds": untapered_premerger_seconds,
            "chirp_time_contained": chirp_time <= untapered_premerger_seconds,
            "finite": finite,
            "conditioned_reference_relative_difference": relative_difference,
            "construction_energy_outside_crop_fraction": outside_crop_fraction,
            "conditioned_crop_energy_retained_fraction": retained_fraction,
            "leading_guard_energy": leading_guard_energy,
            "trailing_guard_energy": trailing_guard_energy,
            "construction_infft_normalization_relative_difference": (
                construction_roundtrip
            ),
            "reference_infft_normalization_relative_difference": reference_roundtrip,
            "conditioned_selection_snr": selection_snr,
            "stored_clean_selection_snr": stored_snr,
            "stored_clean_snr_relative_difference": snr_relative_difference,
        }
        row["passed"] = bool(
            finite
            and row["chirp_time_contained"]
            and relative_difference
            <= float(boundary["maximum_conditioned_reference_relative_difference"])
            and outside_crop_fraction
            <= float(boundary["maximum_construction_energy_outside_crop_fraction"])
            and retained_fraction
            >= float(boundary["minimum_conditioned_crop_energy_retained_fraction"])
            and leading_guard_energy == 0.0
            and trailing_guard_energy == 0.0
            and construction_roundtrip
            <= float(boundary["transform_roundtrip_relative_tolerance"])
            and reference_roundtrip
            <= float(boundary["transform_roundtrip_relative_tolerance"])
            and snr_relative_difference
            <= float(boundary["stored_clean_snr_relative_tolerance"])
        )
        rows.append(row)
    result = {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "criteria": boundary,
        "cases": rows,
        "failure_count": sum(not row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
