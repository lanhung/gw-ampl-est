#!/usr/bin/env python3
"""Select a terminal-rung architecture using development validation only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.training.architecture import select_architecture
from gwlens_mm.training.terminal_architecture import (
    collect_terminal_architecture_results,
    validate_terminal_architecture_execution_gate,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    for name in (
        "stage-a-publication",
        "stage-b-publication",
        "combined-base-publication",
        "correction-publication",
        "train-increment-publication",
        "combined-131k-publication",
        "development-tail-publication",
        "terminal-decision",
        "probe-output-root",
        "architecture-output-root",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        result: Mapping[str, Any] = {
            "status": "terminal_architecture_selection_implementation_ready",
            "architecture_selection_executed": False,
            "calibration_authorized": False,
            "final_evaluation_authorized": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    authorization = load_yaml(arguments.authorization)
    expected_output = Path(
        str(authorization.get("architecture_selection_output_path", ""))
    ).resolve()
    if arguments.output.resolve() != expected_output:
        raise ValueError("terminal architecture selection output changed")
    if arguments.output.exists():
        raise FileExistsError("terminal architecture selection already exists")
    gate = validate_terminal_architecture_execution_gate(
        arguments.root.resolve(),
        authorization_path=arguments.authorization,
        stage_a_publication_root=arguments.stage_a_publication,
        stage_b_publication_root=arguments.stage_b_publication,
        combined_base_publication_root=arguments.combined_base_publication,
        correction_publication_root=arguments.correction_publication,
        train_parent_root=arguments.train_increment_publication,
        combined_131k_publication_root=arguments.combined_131k_publication,
        development_tail_parent_root=arguments.development_tail_publication,
        terminal_decision_path=arguments.terminal_decision,
        probe_output_root=arguments.probe_output_root,
    )
    configured_architecture_root = Path(
        str(authorization.get("architecture_output_root", ""))
    ).resolve()
    if arguments.architecture_output_root.resolve() != configured_architecture_root:
        raise ValueError("terminal architecture result root changed")
    result = select_architecture(
        collect_terminal_architecture_results(
            arguments.root.resolve(),
            gate=gate,
            architecture_output_root=arguments.architecture_output_root,
        )
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
