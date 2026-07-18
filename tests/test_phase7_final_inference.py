from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.config import load_yaml
from gwlens_mm.training.final_inference import (
    FINAL_ANALYSIS_HASH,
    FINAL_CASE_COUNT,
    FINAL_COMMITMENT_SHA256,
    FINAL_NAMESPACE_COUNT,
    FINAL_SHARD_COUNT,
    FinalInferenceGateError,
    _expected_namespaces,
    _score_final_batches,
    calibrated_final_batch_metrics,
    dry_run_plan,
    final_inference_seed,
    load_final_inference_stack_contract,
    resolve_sealed_final_publication,
    validate_final_inference_authorization,
)
from gwlens_mm.training.reference_baseline import REFERENCE_CONFIG_HASH

ROOT = Path(__file__).resolve().parents[1]


def _calibration_map(cell: str = "cell-a") -> dict[str, object]:
    marginal = {
        target: {
            f"{level:.2f}": {"raw_region_mass_threshold": level}
            for level in (0.5, 0.8, 0.9, 0.95)
        }
        for target in ("log_abs_mu_primary", "log_abs_mu_secondary")
    }
    joint = {
        f"{level:.2f}": {"raw_region_mass_threshold": level}
        for level in (0.5, 0.8, 0.9, 0.95)
    }
    return {"em_cells": {cell: {"marginal": marginal, "joint": joint}}}


def test_implementation_gate_freezes_parents_and_keeps_execution_closed() -> None:
    authorization = load_final_inference_stack_contract(ROOT)
    flags = authorization["authorization"]
    assert authorization["authorization_status"].endswith("synthetic_fixture_only")
    assert flags["checkpoint_inference_runner_implementation_authorized"] is True
    assert flags["final_evaluation_data_access_authorized"] is False
    assert flags["checkpoint_access_authorized"] is False
    assert flags["scientific_inference_execution_authorized"] is False
    assert flags["gwosc_gwtc_access_authorized"] is False
    plan = dry_run_plan(ROOT)
    assert plan["status"] == "implementation_ready_execution_closed"
    assert plan["accepted_case_count"] == FINAL_CASE_COUNT
    assert plan["shard_count"] == FINAL_SHARD_COUNT
    assert plan["namespace_count"] == FINAL_NAMESPACE_COUNT
    assert plan["final_analysis_hash"] == FINAL_ANALYSIS_HASH
    assert plan["reference_baseline_hash"] == REFERENCE_CONFIG_HASH
    assert plan["final_generation_commitment_sha256"] == FINAL_COMMITMENT_SHA256
    assert plan["final_data_accessed"] is False


def test_final_sampling_seeds_are_deterministic_and_disjoint() -> None:
    namespaces = _expected_namespaces(ROOT)
    values = {
        final_inference_seed(namespace.namespace_id, seed)
        for namespace in namespaces
        for seed in (0, 1, 2)
    }
    assert len(values) == len(namespaces) * 3
    namespace = namespaces[0].namespace_id
    assert final_inference_seed(namespace, 1) == final_inference_seed(namespace, 1)
    with pytest.raises(FinalInferenceGateError):
        final_inference_seed(namespace, 3)


def test_calibrated_metrics_are_finite_and_require_matching_cell() -> None:
    rng = np.random.default_rng(9)
    draws = rng.normal(size=(2, 4096, 2))
    truth = np.zeros((2, 2), dtype=np.float64)
    density = -0.5 * np.square(draws).sum(axis=-1)
    truth_density = np.zeros(2, dtype=np.float64)
    result = calibrated_final_batch_metrics(
        draws,
        truth,
        density,
        truth_density,
        ("cell-a", "cell-a"),
        _calibration_map(),
    )
    assert result["crps"].shape == (2, 2)
    assert result["marginal_covered_90"].shape == (2, 2)
    assert result["joint_covered_90"].shape == (2,)
    assert np.all(result["marginal_interval_width_95"] > 0)
    with pytest.raises(FinalInferenceGateError, match="no final EM cell"):
        calibrated_final_batch_metrics(
            draws,
            truth,
            density,
            truth_density,
            ("missing", "missing"),
            _calibration_map(),
        )


def _write_sealed_fixture(root: Path) -> Path:
    parent = root / "sealed" / "parent"
    parent.mkdir(parents=True)
    validations: dict[str, object] = {}
    for index, namespace in enumerate(_expected_namespaces(ROOT)):
        dataset_id = f"final-dataset-{index:02d}"
        dataset_root = parent / dataset_id
        dataset_root.mkdir()
        (dataset_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "generating_or_resuming_sealed",
                    "namespace_id": namespace.namespace_id,
                    "split": namespace.split.value,
                    "diagnostic_context_id": namespace.diagnostic_context_id,
                    "dataset_id": dataset_id,
                    "accepted_target": namespace.accepted_count,
                    "unsealing_authorized": False,
                }
            )
        )
        validations[namespace.namespace_id] = {
            "status": "passed_sealed",
            "namespace_id": namespace.namespace_id,
            "split": namespace.split.value,
            "diagnostic_context_id": namespace.diagnostic_context_id,
            "dataset_id": dataset_id,
            "accepted_pair_count": namespace.accepted_count,
            "complete_shard_count": namespace.shard_count,
        }
    manifest = {
        "status": "passed_sealed",
        "sealed": True,
        "unsealing_authorized": False,
        "accepted_pair_count": FINAL_CASE_COUNT,
        "complete_shard_count": FINAL_SHARD_COUNT,
        "namespace_count": FINAL_NAMESPACE_COUNT,
        "all_namespaces_group_disjoint": True,
        "learning_curve_use_authorized": False,
        "architecture_selection_use_authorized": False,
        "calibration_fit_use_authorized": False,
        "gwosc_gwtc_accessed": False,
        "configuration_hash": (
            "11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66"
        ),
        "commitment_sha256": FINAL_COMMITMENT_SHA256,
        "generator_commit": "a" * 40,
        "validations": validations,
    }
    (parent / "dataset_manifest.json").write_text(json.dumps(manifest))
    return parent


def test_sealed_publication_resolver_checks_all_fifteen_namespaces(
    tmp_path: Path,
) -> None:
    parent = _write_sealed_fixture(tmp_path)
    publication = resolve_sealed_final_publication(ROOT, parent)
    assert len(publication.namespaces) == 15
    assert publication.generator_commit == "a" * 40
    manifest_hash = hashlib.sha256(
        (parent / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    assert publication.manifest_sha256 == manifest_hash
    manifest = json.loads((parent / "dataset_manifest.json").read_text())
    manifest["accepted_pair_count"] = FINAL_CASE_COUNT - 1
    (parent / "dataset_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FinalInferenceGateError, match="parent manifest contract"):
        resolve_sealed_final_publication(ROOT, parent)


def test_synthetic_checkpoint_scoring_covers_equal_family_mixture() -> None:
    torch = pytest.importorskip("torch")

    class Flow:
        @staticmethod
        def log_prob(value: object, *, context: object) -> object:
            values = value
            contexts = context
            return -0.5 * ((values - contexts[:, :2]) ** 2).sum(dim=-1)

    class Model:
        flow = Flow()

        def to(self, device: object) -> "Model":
            return self

        def eval(self) -> "Model":
            return self

        @staticmethod
        def encode_context(batch: dict[str, object]) -> object:
            return batch["lens_family_condition"]

        @staticmethod
        def sample_from_context(count: int, context: object) -> object:
            return context[:, None, :].expand(-1, count, -1).clone()

        def sample_log_prob_from_context(
            self, samples: object, context: object
        ) -> object:
            flat = samples.reshape(-1, 2)
            repeated = context.repeat_interleave(samples.shape[1], dim=0)
            return self.flow.log_prob(flat, context=repeated).reshape(
                samples.shape[0], samples.shape[1]
            )

    batch = {
        "target": torch.zeros((1, 2), dtype=torch.float32),
        "lens_family_condition": torch.tensor([[1.0, 0.0]]),
    }
    metadata = (
        {
            "physical_system_id": "system-1",
            "lens_family": "sie_external_shear",
            "em_cell": "cell-a",
            "split": "cross_family_misspecification_test",
            "diagnostic_context_id": "sie_truth_family_marginalized",
        },
    )
    from gwlens_mm.training.engine import TargetStandardizer

    result = _score_final_batches(
        Model(),
        [(batch, metadata)],
        target_standardizer=TargetStandardizer((0.0, 0.0), (1.0, 1.0)),
        calibration_map=_calibration_map(),
        inference_seed=7,
        draw_microbatch=512,
        device_name="cpu",
    )
    assert result["truth"].shape == (1, 2)
    assert result["physical_system_ids"].tolist() == ["system-1"]
    assert np.isfinite(result["truth_log_density"]).all()


def test_current_gate_cannot_execute_and_cli_execute_fails(tmp_path: Path) -> None:
    with pytest.raises(FinalInferenceGateError, match="not authorized"):
        validate_final_inference_authorization(
            ROOT,
            authorization_path=(
                ROOT
                / "configs/execution/phase7_final_inference_stack_authorization.yaml"
            ),
            publication_root=tmp_path,
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/phase7/prepare_final_inference.py"),
            "--root",
            str(ROOT),
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert process.returncode != 0
    assert "requires every identity" in process.stderr


def test_statistics_runner_records_hashes_for_all_downstream_outputs() -> None:
    source = (ROOT / "scripts/phase6/run_calibration_sbc_statistics.py").read_text()
    assert '"calibration_map_sha256"' in source
    assert '"sbc_summary_sha256"' in source
    assert '"independent_coverage_sha256"' in source


def test_no_execution_flag_is_opened_in_final_contracts() -> None:
    authorization = load_yaml(
        ROOT / "configs/execution/phase7_final_inference_stack_authorization.yaml"
    )
    assert authorization["stop_after_implementation"] is True
    assert all(
        value is False
        for name, value in authorization["authorization"].items()
        if name
        not in {
            "sealed_publication_resolver_implementation_authorized",
            "checkpoint_inference_runner_implementation_authorized",
            "calibrated_metric_artifact_implementation_authorized",
            "cross_family_executor_implementation_authorized",
            "synthetic_fixture_tests_authorized",
        }
    )
