#!/usr/bin/env python3
"""Build deterministic Phase-0 manifests from the curated legacy snapshot."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "vendor" / "legacy_snapshot"
MANIFESTS = ROOT / "manifests"

SOURCE_ROOTS = {
    "candidate_wjx": "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main",
    "candidate_qkzhang": "/root/autodl-tmp/qkzhang",
    "baseline_tmp": "/root/autodl-tmp/tmp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def purpose_for(path: Path) -> str:
    name = path.name.lower()
    if "generate_sis" in name or "sis_gw_events" in name:
        return "SIS catalog generation"
    if "point_mass" in name or "pm_gw_events" in name:
        return "point-mass catalog generation"
    if "unlensed" in name:
        return "unlensed catalog generation"
    if "train_mu0_baseline" in name:
        return "PDF baseline training entry"
    if "train_mu0_from_wave_obs_randomu" in name:
        return "PDF baseline shared wave-plus-observable model"
    if "obsfix" in name:
        return "post-PDF observable-fix training variant"
    if "pi_resnet" in name or "pair_dataset" in name:
        return "legacy pair-verification model"
    if "cqt" in name or path.parent.name == "cqt_deit":
        return "legacy CQT pair-verification baseline"
    if "audit_0222" in name or "build_split" in name:
        return "legacy reproducibility utility"
    if "qc" in name:
        return "legacy smoke quality control"
    return "legacy support code"


def decision_for(path: Path) -> str:
    text = str(path)
    if "baseline_tmp" in text and (
        "SIS_GW_events" in path.name
        or "train_mu0" in path.name
    ):
        return "reuse selected conventions after unit tests; rewrite for v2"
    if "src/generation" in text or "data_gen_check" in text:
        return "smoke-only; extract tested physics conventions"
    if "classifier" in text or "cqt_deit" in text:
        return "legacy comparison only; exclude from v2 posterior implementation"
    return "retain unchanged as provenance"


def build_snapshot_manifests() -> None:
    files: list[dict[str, object]] = []
    code: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for path in sorted(p for p in SNAPSHOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = path.relative_to(SNAPSHOT)
        source_key = rel.parts[0]
        # README.md documents this local curated snapshot; it is not a file
        # copied from any remote source root and therefore is not inventory
        # evidence.
        if source_key not in SOURCE_ROOTS:
            continue
        source_rel = Path(*rel.parts[1:])
        source_path = f"{SOURCE_ROOTS[source_key]}/{source_rel.as_posix()}"
        digest = sha256(path)
        snapshot_rel = path.relative_to(ROOT).as_posix()
        checksum_lines.append(f"{digest}  {snapshot_rel}")
        row = {
            "source_group": source_key,
            "source_path": source_path,
            "snapshot_path": snapshot_rel,
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "sha256": digest,
            "content_role": "source" if path.suffix == ".py" else "metadata_or_documentation",
            "decision": decision_for(path),
        }
        files.append(row)
        if path.suffix == ".py":
            code.append(
                {
                    "source_group": source_key,
                    "source_path": source_path,
                    "snapshot_path": snapshot_rel,
                    "sha256": digest,
                    "purpose": purpose_for(path),
                    "version_evidence": "git:8326ea0" if source_key == "candidate_wjx" else "no_git_history",
                    "decision": decision_for(path),
                }
            )
    write_csv(
        MANIFESTS / "legacy_file_inventory.csv",
        ["source_group", "source_path", "snapshot_path", "bytes", "mtime_utc", "sha256", "content_role", "decision"],
        files,
    )
    write_csv(
        MANIFESTS / "legacy_code_inventory.csv",
        ["source_group", "source_path", "snapshot_path", "sha256", "purpose", "version_evidence", "decision"],
        code,
    )
    (MANIFESTS / "legacy_snapshot_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def build_dataset_inventory() -> None:
    rows = [
        # qkzhang 0222/0228 catalogs audited directly.
        ("qk_sis_0222", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/SIS_data_0222", "SIS", 2500, "(2500,98304)", "float64", "unknown single channel; likely ET legacy", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "seed 6130 lens sequence; waveform generator version incomplete", 33, "legacy benchmark and analytic checks only"),
        ("qk_sis_0228", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/SIS_data_0228", "SIS", 2500, "(2500,98304)", "float64", "unknown single channel; likely ET legacy", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "seed unavailable; no exact generator provenance", 33, "legacy IID holdout only; not final v2 test"),
        ("qk_pm_0222", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/PM_data_0222", "point_mass", 2500, "(2500,98304)", "float64", "unknown single channel; likely ET legacy", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "seed 614 lens sequence; waveform generator version incomplete", 33, "legacy pair-verification benchmark only"),
        ("qk_pm_0228", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/PM_data_0228", "point_mass", 2500, "(2500,98304)", "float64", "unknown single channel; likely ET legacy", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "seed unavailable; no exact generator provenance", 33, "legacy IID holdout only"),
        ("qk_unlensed_0222", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/Unlensed_data_0222", "unlensed", 5000, "(5000,98304)", "float64", "unknown single channel", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "no exact generator provenance", 11, "legacy background only"),
        ("qk_unlensed_0228", "candidate_qkzhang", "/root/autodl-tmp/qkzhang/Unlensed_data_0228", "unlensed", 5000, "(5000,98304)", "float64", "unknown single channel", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "no exact generator provenance", 11, "legacy background only"),
        # Later H1/L1 regenerations in the pair-classification project.
        ("wjx_final_v3", "candidate_wjx", "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/data/final_v3", "SIS+point_mass", 2500, "(2500,2,98304)", "float64", "H1,L1", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "metadata numerically matches qk 0222; generation Git history imported later", 88, "smoke-only; not real noise or final test"),
        ("wjx_ligo_reduced", "candidate_wjx", "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/data/ligo_reduced", "SIS+point_mass+unlensed", 2500, "(2500,2,98304)", "float64", "H1,L1", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "sample-identical copy of final_v3 for checked arrays", 154, "duplicate legacy artifact; do not copy into v2"),
        ("wjx_ligo_full", "candidate_wjx", "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/data/ligo_full", "SIS+point_mass+unlensed", 10000, "(10000,2,98304)", "float64", "H1,L1", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "generator code at wjx commit 8326ea0; data lack per-dataset commit manifest", 580, "legacy scale benchmark only"),
        ("wjx_ligo_bf_sis", "candidate_wjx", "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/data/ligo_bf_sis", "SIS_partial", 2500, "(2500,2,98304)", "float64", "H1,L1", "stationary Gaussian from design PSD", "IMRPhenomXPHM", "partial copy of final_v3 SIS", 7.4, "exclude as incomplete duplicate"),
        # Exact PDF baseline and later obsfix regeneration.
        ("pdf_mu0_before_obsfix", "baseline_tmp", "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0", "SIS", 2500, "(2500,4096)", "float32", "ET single detector", "stationary Gaussian from ET design PSD", "IMRPhenomXPHM", "generator sha256 ae3879fb...; seed 238; no Git", 0.634, "retain as exact PDF point-regression baseline only"),
        ("pdf_mu0_obsfix", "baseline_tmp", "/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0(1)", "SIS", 2500, "(2500,4096)", "float32", "ET single detector", "stationary Gaussian from ET design PSD", "IMRPhenomXPHM", "generator sha256 0d25a260...; seed 238; same waveform sample hashes as before_obsfix", 0.634, "post-PDF diagnostic variant; privileged/smoke only"),
    ]
    fieldnames = [
        "dataset_id", "source_group", "path", "lens_family", "primary_events", "primary_strain_shape",
        "dtype", "detectors", "noise", "waveform", "provenance", "approx_size_gib", "decision",
    ]
    write_csv(
        MANIFESTS / "legacy_dataset_inventory.csv",
        fieldnames,
        [dict(zip(fieldnames, row)) for row in rows],
    )


def main() -> None:
    build_snapshot_manifests()
    build_dataset_inventory()


if __name__ == "__main__":
    main()
