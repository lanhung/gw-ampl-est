#!/usr/bin/env python3
"""Apply the terminal 32k-to-65k development-only decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.training.learning_curve import compare_32k_to_65k


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--probe-32k-root", required=True, type=Path)
    parser.add_argument("--probe-65k-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    authorization = load_yaml(arguments.authorization)
    ready = (
        authorization.get("authorization_status") == "authorized_train_65k_probe_only"
        and authorization.get("authorization", {}).get(
            "learning_curve_decision_authorized"
        )
        is True
    )
    if not arguments.execute:
        result: Mapping[str, Any] = {
            "status": "terminal_learning_curve_implementation_ready",
            "scientific_decision_executed": False,
            "scientific_decision_authorized": ready,
            "extension_above_65536_authorized": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not ready:
        raise PermissionError("32k-to-65k decision has no scientific authorization")
    expected_32k = Path(str(authorization.get("probe_32k_output_root", ""))).resolve()
    expected_65k = Path(str(authorization.get("training_output_root", ""))).resolve()
    if (
        arguments.probe_32k_root.resolve() != expected_32k
        or arguments.probe_65k_root.resolve() != expected_65k
    ):
        raise ValueError("terminal learning-curve input differs from authorization")
    expected_output = Path(
        str(authorization.get("learning_curve_output_path", ""))
    ).resolve()
    if arguments.output.resolve() != expected_output:
        raise ValueError("terminal learning-curve output differs from authorization")
    if arguments.output.exists():
        raise FileExistsError("terminal learning-curve decision already exists")
    result = compare_32k_to_65k(
        arguments.probe_32k_root, arguments.probe_65k_root
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(arguments.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
