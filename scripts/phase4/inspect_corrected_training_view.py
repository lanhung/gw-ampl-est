#!/usr/bin/env python3
"""Resolve the immutable correction overlay without opening Parquet or Zarr."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.contracts import RC4_HASH, WAVEFORM_CORRECTION_HASH
from gwlens_mm.training.data import resolve_corrected_training_publication

ROOT = Path(__file__).resolve().parents[2]
BASE_GENERATOR = "2be777e727ef9d8e1a85f89c68966df5d37932b0"


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-base-publication", required=True, type=Path)
    parser.add_argument("--correction-publication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    execution = _load(root / "results/phase4/waveform_correction_execution_result.json")
    combined_path = root / "results/phase4/train_65k_combined_manifest.json"
    publication = resolve_corrected_training_publication(
        arguments.correction_publication,
        stage_a_parent_root=arguments.stage_a_publication,
        stage_b_parent_root=arguments.stage_b_publication,
        combined_base_root=arguments.combined_base_publication,
        expected_base_generator_commit=BASE_GENERATOR,
        expected_base_preregistration_hash=RC4_HASH,
        expected_correction_generator_commit=str(execution["generator_commit"]),
        expected_correction_preregistration_hash=WAVEFORM_CORRECTION_HASH,
        expected_correction_manifest_sha256=str(execution["parent_manifest_sha256"]),
        expected_correction_tree_sha256=str(execution["publication_tree_sha256"]),
        expected_combined_base_manifest_sha256=hashlib.sha256(
            combined_path.read_bytes()
        ).hexdigest(),
    )
    result = {
        "status": "corrected_training_view_resolved_metadata_only",
        "correction_manifest_sha256": publication.correction_manifest_sha256,
        "correction_tree_sha256": publication.correction_tree_sha256,
        "corrected_stage_a_train_manifest_sha256": (
            publication.corrected_stage_a_train_manifest_sha256
        ),
        "corrected_combined_train_manifest_sha256": (
            publication.corrected_combined_train_manifest_sha256
        ),
        "stage_a_train_count": 32768,
        "stage_a_validation_count": 6144,
        "combined_train_count": 65536,
        "stage_a_excluded_ids": list(publication.stage_a_excluded_ids),
        "stage_b_excluded_ids": list(publication.stage_b_excluded_ids),
        "stage_a_replacement_ids": list(publication.stage_a_replacement_ids),
        "stage_b_replacement_ids": list(publication.stage_b_replacement_ids),
        "parquet_opened": False,
        "zarr_opened": False,
        "optimizer_started": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
        "gwosc_gwtc_accessed": False,
    }
    _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
