#!/usr/bin/env python3
"""Regenerate the five known pathologies and a boundary-valid source waveform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from gwlens_mm.production.gw import (
    ProductionWaveformEngine,
    WaveformNumericalPathology,
    validate_source_polarization_spectrum,
)
from gwlens_mm.production.waveform_correction import (
    CORRECTION_CONFIG,
    build_replacement_namespace_config,
    load_waveform_correction_contract,
)
from gwlens_mm.schema import V2Record

ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _record_parameters(record: V2Record) -> Dict[str, float]:
    intrinsic = record.source_truth.intrinsic_parameters
    extrinsic = record.source_truth.extrinsic_parameters
    return {
        "mass_1": float(extrinsic["mass_1_detector"]),
        "mass_2": float(extrinsic["mass_2_detector"]),
        "luminosity_distance": float(
            record.source_truth.physical_luminosity_distance_mpc
        ),
        "a_1": float(intrinsic["a_1"]),
        "tilt_1": float(intrinsic["tilt_1"]),
        "phi_12": float(intrinsic["phi_12"]),
        "a_2": float(intrinsic["a_2"]),
        "tilt_2": float(intrinsic["tilt_2"]),
        "phi_jl": float(intrinsic["phi_jl"]),
        "theta_jn": float(extrinsic["theta_jn"]),
        "phase": float(extrinsic["phase"]),
        "ra": float(extrinsic["ra"]),
        "dec": float(extrinsic["dec"]),
        "psi": float(extrinsic["psi"]),
    }


def _load_record(dataset: Path, shard_index: int, physical_id: str) -> V2Record:
    pandas = __import__("pandas")
    path = dataset / "shards" / f"shard-{shard_index:05d}" / "records.parquet"
    frame = pandas.read_parquet(path, columns=["record_json"])
    matches = []
    for raw in frame["record_json"]:
        record = V2Record.from_json(str(raw))
        if record.pair.physical_system_id == physical_id:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"expected one {physical_id} record, found {len(matches)}")
    return matches[0]


def execute(config_path: str = CORRECTION_CONFIG) -> Dict[str, Any]:
    config = load_waveform_correction_contract(ROOT, config_path)
    audit = json.loads((ROOT / config["audit"]["path"]).read_text(encoding="utf-8"))
    namespace = build_replacement_namespace_config(ROOT, config, "stage_a_train")
    engine = ProductionWaveformEngine(namespace["gw"], int(namespace["root_seed"]))
    base = config["immutable_base_publications"]
    datasets = {
        "stage_a_train": Path(str(base["stage_a"]["parent_root"]))
        / str(base["stage_a"]["train_dataset_id"]),
        "stage_b_train": Path(str(base["stage_b"]["parent_root"]))
        / str(base["stage_b"]["train_dataset_id"]),
    }
    cases = [*audit["pathologies"], audit["largest_valid_ratio"]]
    results = []
    for case in cases:
        record = _load_record(
            datasets[str(case["split_component"])],
            int(case["shard_index"]),
            str(case["physical_system_id"]),
        )
        parameters = _record_parameters(record)
        raw = engine._waveform_generator.frequency_domain_strain(parameters)
        diagnostic = validate_source_polarization_spectrum(
            raw,
            np.asarray(engine._waveform_generator.frequency_array),
            minimum_frequency_hz=20.0,
            positive_amplitude_quantile=0.999,
            maximum_peak_to_quantile_ratio=1.0e300,
        )
        rejected = False
        try:
            engine.source_polarizations(parameters)
        except WaveformNumericalPathology:
            rejected = True
        expected_rejected = case in audit["pathologies"]
        if rejected != expected_rejected:
            raise ValueError(
                f"waveform-correction classification mismatch for {case['physical_system_id']}"
            )
        results.append(
            {
                "physical_system_id": case["physical_system_id"],
                "split_component": case["split_component"],
                "maximum_peak_to_quantile_ratio": (
                    diagnostic.maximum_peak_to_quantile_ratio
                ),
                "expected_rejected": expected_rejected,
                "rejected": rejected,
            }
        )
    return {
        "status": "passed",
        "pathology_count": 5,
        "valid_boundary_case_count": 1,
        "cases": results,
        "all_known_pathologies_rejected": all(
            item["rejected"] for item in results if item["expected_rejected"]
        ),
        "largest_known_valid_record_accepted": all(
            not item["rejected"] for item in results if not item["expected_rejected"]
        ),
        "accepted_pair_generator_called": False,
        "waveform_pairs_generated": 0,
        "published_data_modified": False,
        "gwosc_gwtc_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CORRECTION_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = execute(args.config)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
