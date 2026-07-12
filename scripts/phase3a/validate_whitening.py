#!/usr/bin/env python3
"""Validate frozen detector-specific Gaussian-noise whitening behavior."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.production.run_control import verify_psd_files
from gwlens_mm.provenance import derive_seed

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data/phase3a_qualification.yaml")
    gw = config["gw"]
    criteria = config["whitening"]
    bilby = importlib.import_module("bilby")
    verified_psds = verify_psd_files(gw["psd_curves"])
    realizations = int(config["diagnostics"]["whitening_realizations_per_detector"])
    summaries: Dict[str, Dict[str, Any]] = {}
    for detector_index, detector in enumerate(("H1", "L1", "V1")):
        time_pieces = []
        band_powers: Dict[str, list[float]] = {
            f"{low:g}-{high:g}": [] for low, high in criteria["frequency_bands_hz"]
        }
        psd_sources = set()
        for realization in range(realizations):
            interferometer = bilby.gw.detector.get_empty_interferometer(detector)
            psd = interferometer.power_spectral_density
            psd_source = getattr(psd, "asd_file", None) or getattr(psd, "psd_file", None)
            if psd_source is None or Path(str(psd_source)).name != str(
                gw["psd_curves"][detector]["file"]
            ):
                raise ValueError(f"Bilby default PSD does not match config for {detector}")
            psd_sources.add(str(psd_source))
            seed = derive_seed(
                int(config["root_seed"]),
                "detector_noise",
                "whitening-fixture",
                detector,
                str(realization),
            )
            bilby.core.utils.random.seed(seed)
            interferometer.set_strain_data_from_power_spectral_density(
                int(gw["sample_rate_hz"]),
                float(gw["duration_seconds"]),
                start_time=float(config["gps_schedule"]["start_gps"])
                + detector_index * 1000
                + realization * 10,
            )
            whitened_time = np.asarray(
                interferometer.whitened_time_domain_strain, dtype=np.float64
            )
            whitened_frequency = np.asarray(
                interferometer.whitened_frequency_domain_strain,
                dtype=np.complex128,
            )
            frequencies = np.asarray(interferometer.frequency_array, dtype=np.float64)
            time_pieces.append(whitened_time)
            analysis = (frequencies >= float(gw["minimum_frequency_hz"])) & (
                frequencies < int(gw["sample_rate_hz"]) / 2
            )
            overall_power = float(np.mean(np.abs(whitened_frequency[analysis]) ** 2))
            for low, high in criteria["frequency_bands_hz"]:
                key = f"{low:g}-{high:g}"
                selected = (frequencies >= float(low)) & (frequencies < float(high))
                band_power = float(np.mean(np.abs(whitened_frequency[selected]) ** 2))
                band_powers[key].append(band_power / overall_power)
        values = np.concatenate(time_pieces)
        band_ratios = {
            key: float(np.mean(items)) for key, items in band_powers.items()
        }
        summary = {
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "quantiles": np.quantile(
                values, [0.001, 0.01, 0.5, 0.99, 0.999]
            ).tolist(),
            "outlier_count_abs_gt_8": int(np.sum(np.abs(values) > 8.0)),
            "frequency_band_power_ratios": band_ratios,
            "psd_sources": sorted(psd_sources),
            "verified_psd": verified_psds[detector],
            "synthetic_noise_label": "synthetic_gaussian_curve_conditioned",
            "per_event_observed_standard_deviation_normalization": False,
        }
        summary["passed"] = bool(
            summary["finite_fraction"] == float(criteria["finite_fraction_required"])
            and abs(float(summary["mean"]))
            <= float(criteria["aggregate_absolute_mean_maximum"])
            and float(criteria["aggregate_standard_deviation_minimum"])
            <= float(summary["standard_deviation"])
            <= float(criteria["aggregate_standard_deviation_maximum"])
            and all(
                float(criteria["frequency_band_power_ratio_minimum"])
                <= value
                <= float(criteria["frequency_band_power_ratio_maximum"])
                for value in band_ratios.values()
            )
        )
        summaries[detector] = summary
    passed = all(bool(summary["passed"]) for summary in summaries.values())
    result = {
        "status": "passed" if passed else "failed",
        "realizations_per_detector": realizations,
        "criteria": criteria,
        "detectors": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
