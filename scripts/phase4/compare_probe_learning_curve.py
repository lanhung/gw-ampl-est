#!/usr/bin/env python3
"""Apply the frozen 16k-to-32k decision after all six authorized fits exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.training.learning_curve import write_learning_curve_decision


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--training-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    authorization = load_yaml(arguments.authorization)
    ready = (
        authorization.get("authorization_status")
        in {
            "authorized_probe_training_only",
            "authorized_corrected_probe_training_only",
        }
        and authorization.get("authorization", {}).get(
            "learning_curve_decision_authorized"
        )
        is True
    )
    if not arguments.execute:
        result: Mapping[str, Any] = {
            "status": "learning_curve_comparison_implementation_ready",
            "scientific_decision_executed": False,
            "scientific_decision_authorized": ready,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not ready:
        raise PermissionError("learning-curve decision has no scientific authorization")
    configured_root = Path(str(authorization.get("training_output_root", ""))).resolve()
    if configured_root != arguments.training_output_root.resolve():
        raise ValueError("learning-curve input differs from the authorized training root")
    result = write_learning_curve_decision(
        arguments.training_output_root, arguments.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
