from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwlens_mm.training import reference_execution as execution_module
from gwlens_mm.training.architecture import selected_model_configuration
from gwlens_mm.training.contracts import model_configuration_hash
from gwlens_mm.training.features import InputStandardizer, PreparedExample
from gwlens_mm.training.reference_baseline import (
    ReferenceBaselineGateError,
    ReferenceCaseScore,
)
from gwlens_mm.training.reference_execution import (
    _validate_terminal_and_architecture,
    score_reference_query_to_artifacts,
    validate_reference_execution_stack_contract,
    validate_reference_query_execution_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def _example(identifier: str, *, family: str = "sie_external_shear") -> PreparedExample:
    return PreparedExample(
        gw_strain=np.empty((0,), dtype=np.float32),
        detector_mask=np.zeros((2, 3), dtype=np.float32),
        astrometry_items=np.zeros((5, 9), dtype=np.float32),
        astrometry_mask=np.ones(5, dtype=np.float32),
        scalar_features=np.zeros(22, dtype=np.float32),
        scalar_mask=np.ones(22, dtype=np.float32),
        modality_mask=np.ones(8, dtype=np.float32),
        lens_family_condition=np.asarray((1.0, 0.0), dtype=np.float32),
        target=np.asarray((1.0, 2.0), dtype=np.float32),
        physical_system_id=identifier,
        lens_family=family,
        em_cell_signature="all",
        em_cell="all_modalities",
    )


def _standardizer() -> InputStandardizer:
    return InputStandardizer(
        scalar_mean=(0.0,) * 22,
        scalar_standard_deviation=(1.0,) * 22,
        astrometry_mean=(0.0,) * 5,
        astrometry_standard_deviation=(1.0,) * 5,
    )


class _FakeDataset:
    def __init__(self, identifiers: tuple[str, ...]) -> None:
        self.identifiers = identifiers

    def __len__(self) -> int:
        return len(self.identifiers)

    def metadata_example(self, index: int) -> PreparedExample:
        return _example(self.identifiers[index])


class _FakeIndex:
    identity_sha256 = "a" * 64

    def manifest(self) -> dict[str, object]:
        return {
            "reference_id": "selected_prior_em_timing_knn_kde_v1",
            "identity_sha256": self.identity_sha256,
            "physical_system_count": 65536,
            "feature_dimension": 100,
            "stratum_counts": {"sie_external_shear::all_modalities": 4096},
            "neighbor_count": 256,
            "posterior_draws_per_case": 4096,
            "gw_strain_opened": False,
            "targets_exposed_as_deployable_inputs": False,
        }

    def score(self, query: PreparedExample) -> ReferenceCaseScore:
        return ReferenceCaseScore(
            physical_system_id=query.physical_system_id,
            lens_family=query.lens_family,
            em_cell=str(query.em_cell),
            target_log_abs_mu=(1.0, 2.0),
            log_probability=-0.4,
            crps=(0.1, 0.2),
            marginal_coverage={
                "0.50": (True, False),
                "0.80": (True, True),
                "0.90": (True, True),
                "0.95": (True, True),
            },
            joint_central_coverage={
                "0.50": False,
                "0.80": True,
                "0.90": True,
                "0.95": True,
            },
            interval_width={
                "0.50": (0.5, 0.6),
                "0.80": (0.8, 0.9),
                "0.90": (1.0, 1.1),
                "0.95": (1.2, 1.3),
            },
            effective_neighbor_count=200.0,
            neighbor_identity_sha256="b" * 64,
        )


def test_reference_execution_implementation_gate_keeps_every_query_closed() -> None:
    contract = validate_reference_execution_stack_contract(ROOT)
    assert contract["config"]["preregistration_version"] == "1.1.0-rc.7"
    authorization = contract["authorization"]
    assert authorization["implementation_contract"]["query_counts"] == {
        "validation": 6144,
        "iid_test": 8192,
        "balanced_tail_diagnostic": 4096,
    }
    flags = authorization["authorization"]
    allowed = {
        "fail_closed_reference_runner_implementation_authorized",
        "bounded_score_writer_implementation_authorized",
        "atomic_parent_child_reference_implementation_authorized",
        "nonauthorizing_release_packet_implementation_authorized",
        "delegated_review_builder_implementation_authorized",
        "synthetic_fixture_tests_authorized",
    }
    assert all(flags[name] is True for name in allowed)
    assert all(value is False for name, value in flags.items() if name not in allowed)


def test_streaming_score_writer_persists_counts_wilson_and_no_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_module,
        "QUERY_SPECS",
        {"validation": (execution_module.SplitName.VALIDATION, 2)},
    )
    score_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    bank_path = tmp_path / "bank.json"
    result = score_reference_query_to_artifacts(
        _FakeIndex(),  # type: ignore[arg-type]
        _FakeDataset(("query-a", "query-b")),  # type: ignore[arg-type]
        _standardizer(),
        query_role="validation",
        expected_count=2,
        score_jsonl_path=score_path,
        summary_path=summary_path,
        bank_manifest_path=bank_path,
    )
    assert result["query_count"] == 2
    assert result["overall"][
        "mean_negative_log_probability_nat_per_target_dimension"
    ] == pytest.approx(0.2)
    coverage = result["overall"]["coverage"]
    assert coverage["0.50"]["marginal_success_counts"] == [2, 0]
    assert coverage["0.80"]["joint_success_count"] == 2
    assert len(coverage["0.90"]["marginal_wilson_95"]) == 2
    assert result["posterior_draws_persisted"] is False
    assert result["gw_strain_opened"] is False
    rows = [json.loads(line) for line in score_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["posterior_draws_persisted"] is False for row in rows)
    assert not (tmp_path / "scores.jsonl.partial").exists()
    assert json.loads(bank_path.read_text())["targets_exposed_as_deployable_inputs"] is False
    assert json.loads(summary_path.read_text())["score_jsonl_sha256"] == result[
        "score_jsonl_sha256"
    ]


def test_score_writer_rejects_duplicate_query_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_module,
        "QUERY_SPECS",
        {"validation": (execution_module.SplitName.VALIDATION, 2)},
    )
    with pytest.raises(ReferenceBaselineGateError, match="duplicated"):
        score_reference_query_to_artifacts(
            _FakeIndex(),  # type: ignore[arg-type]
            _FakeDataset(("duplicate", "duplicate")),  # type: ignore[arg-type]
            _standardizer(),
            query_role="validation",
            expected_count=2,
            score_jsonl_path=tmp_path / "scores.jsonl",
            summary_path=tmp_path / "summary.json",
            bank_manifest_path=tmp_path / "bank.json",
        )
    assert not (tmp_path / "scores.jsonl").exists()
    assert not (tmp_path / "scores.jsonl.partial").exists()
    assert not (tmp_path / "summary.json").exists()


def test_implementation_authorization_cannot_index_a_scientific_bank(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReferenceBaselineGateError, match="gate is absent"):
        validate_reference_query_execution_gate(
            ROOT,
            authorization_path=ROOT
            / "configs/execution/phase7_reference_execution_stack_authorization.yaml",
            stage_a_publication_root=tmp_path / "stage-a",
            stage_b_publication_root=tmp_path / "stage-b",
            combined_publication_root=tmp_path / "combined",
            correction_publication_root=tmp_path / "correction",
            terminal_decision_path=tmp_path / "terminal.json",
            selected_architecture_decision_path=tmp_path / "selection.json",
            primary_rung_preparation_path=tmp_path / "preparation.json",
            query_dataset_roots=(tmp_path / "query",),
            query_parent_roots=(tmp_path / "parent",),
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "label",
    [
        "lock_train_131k_saturated",
        "lock_train_131k_resource_capped_data_limited",
    ],
)
def test_reference_gate_accepts_both_exact_terminal_131k_labels(
    tmp_path: Path, label: str
) -> None:
    terminal = {
        "status": "terminal_learning_curve_decision_complete",
        "comparison": "corrected_train_65k_to_train_131k_terminal",
        "decision": label,
        "selected_training_count": 131072,
        "architecture_selection_review_allowed": True,
        "extension_above_131072_authorized": False,
        "all_three_probe_seeds_retained": True,
        "best_seed_selected": False,
        "calibration_accessed": False,
        "final_evaluation_accessed": False,
    }
    selected = {
        "status": "architecture_locked_on_development_validation",
        "selected_architecture_id": "nsf-t10-w256",
        "total_result_count": 12,
        "best_seed_selected": False,
        "calibration_accessed": False,
        "sbc_accessed": False,
        "final_evaluation_accessed": False,
    }
    terminal_path = tmp_path / "terminal.json"
    selected_path = tmp_path / "selected.json"
    terminal_path.write_text(json.dumps(terminal) + "\n")
    selected_path.write_text(json.dumps(selected) + "\n")
    model = selected_model_configuration(ROOT, "nsf-t10-w256")
    authorization = {
        "locked_training_rung": 131072,
        "terminal_size_decision_path": str(terminal_path),
        "terminal_size_decision_sha256": hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        "selected_architecture_decision_path": str(selected_path),
        "selected_architecture_decision_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "selected_primary_model_configuration_hash": model_configuration_hash(model),
    }
    architecture_id, observed_model, rung = _validate_terminal_and_architecture(
        ROOT,
        authorization,
        terminal_decision_path=terminal_path,
        selected_architecture_decision_path=selected_path,
    )
    assert architecture_id == "nsf-t10-w256"
    assert model_configuration_hash(observed_model) == model_configuration_hash(model)
    assert rung == 131072


def test_runner_defaults_to_execution_blocked() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase7/run_reference_query.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "implementation_ready_reference_execution_blocked"
    assert result["reference_bank_accessed"] is False
    assert result["query_record_accessed"] is False
    assert result["final_evaluation_accessed"] is False
