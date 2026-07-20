#!/usr/bin/env python3
"""Plan or execute one separately authorized RC.7 reference query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.reference_execution import (
    QUERY_SPECS,
    run_authorized_reference_query,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--stage-a-publication", type=Path)
    parser.add_argument("--stage-b-publication", type=Path)
    parser.add_argument("--combined-publication", type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--terminal-train-parent", type=Path)
    parser.add_argument("--terminal-combined-publication", type=Path)
    parser.add_argument("--terminal-development-tail-parent", type=Path)
    parser.add_argument("--terminal-size-decision", type=Path)
    parser.add_argument("--selected-architecture-decision", type=Path)
    parser.add_argument("--primary-rung-preparation", type=Path)
    parser.add_argument("--query-dataset", type=Path)
    parser.add_argument("--query-parent-manifest", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    result: Mapping[str, Any]
    if not arguments.execute:
        result = {
            "status": "implementation_ready_reference_execution_blocked",
            "query_roles": {
                role: count for role, (_, count) in QUERY_SPECS.items()
            },
            "reference_bank_accessed": False,
            "query_record_accessed": False,
            "checkpoint_accessed": False,
            "final_evaluation_accessed": False,
            "gwosc_gwtc_accessed": False,
        }
    else:
        required = (
            arguments.authorization,
            arguments.stage_a_publication,
            arguments.stage_b_publication,
            arguments.combined_publication,
            arguments.correction_publication,
            arguments.terminal_size_decision,
            arguments.selected_architecture_decision,
            arguments.primary_rung_preparation,
            arguments.query_dataset,
            arguments.query_parent_manifest,
            arguments.psd_root,
            arguments.output_root,
            arguments.execution_commit,
        )
        if any(value is None for value in required):
            raise ValueError("authorized reference query requires every frozen identity")
        result = run_authorized_reference_query(
            arguments.root.resolve(),
            authorization_path=arguments.authorization,
            stage_a_publication_root=arguments.stage_a_publication,
            stage_b_publication_root=arguments.stage_b_publication,
            combined_publication_root=arguments.combined_publication,
            correction_publication_root=arguments.correction_publication,
            terminal_decision_path=arguments.terminal_size_decision,
            selected_architecture_decision_path=arguments.selected_architecture_decision,
            primary_rung_preparation_path=arguments.primary_rung_preparation,
            query_dataset_root=arguments.query_dataset,
            query_parent_manifest_path=arguments.query_parent_manifest,
            psd_root=arguments.psd_root,
            output_root=arguments.output_root,
            execution_commit=arguments.execution_commit,
            terminal_train_parent_root=arguments.terminal_train_parent,
            terminal_combined_publication_root=(
                arguments.terminal_combined_publication
            ),
            terminal_development_tail_parent_root=(
                arguments.terminal_development_tail_parent
            ),
        )
    if arguments.result is not None:
        arguments.result.parent.mkdir(parents=True, exist_ok=True)
        arguments.result.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
