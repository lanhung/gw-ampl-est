#!/usr/bin/env python3
"""Lock one architecture using development validation only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.training.architecture import (
    collect_architecture_results,
    select_architecture,
    validate_architecture_execution_gate,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--stage-a-publication", required=True, type=Path)
    parser.add_argument("--stage-b-publication", required=True, type=Path)
    parser.add_argument("--combined-publication", required=True, type=Path)
    parser.add_argument("--correction-publication", type=Path)
    parser.add_argument("--terminal-decision", required=True, type=Path)
    parser.add_argument("--probe-output-root", required=True, type=Path)
    parser.add_argument("--architecture-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "status": "architecture_selection_implementation_ready",
                    "architecture_selection_executed": False,
                    "calibration_authorized": False,
                    "final_evaluation_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    authorization = load_yaml(arguments.authorization)
    expected_output = Path(
        str(authorization.get("architecture_selection_output_path", ""))
    ).resolve()
    if arguments.output.resolve() != expected_output:
        raise ValueError("architecture selection output differs from authorization")
    if arguments.output.exists():
        raise FileExistsError("architecture selection decision already exists")
    gate = validate_architecture_execution_gate(
        arguments.root.resolve(),
        authorization_path=arguments.authorization,
        stage_a_publication_root=arguments.stage_a_publication,
        stage_b_publication_root=arguments.stage_b_publication,
        combined_publication_root=arguments.combined_publication,
        correction_publication_root=arguments.correction_publication,
        terminal_decision_path=arguments.terminal_decision,
        probe_output_root=arguments.probe_output_root,
    )
    configured_architecture_root = Path(
        str(authorization.get("architecture_output_root", ""))
    ).resolve()
    if arguments.architecture_output_root.resolve() != configured_architecture_root:
        raise ValueError("architecture result root differs from authorization")
    result = select_architecture(
        collect_architecture_results(
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
