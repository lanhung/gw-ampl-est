#!/usr/bin/env python3
"""Execute the exact three-seed final-score summary after authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.final_reporting import (
    dry_run_plan,
    run_authorized_final_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if not arguments.execute:
        result = dry_run_plan(root)
    else:
        if any(
            value is None
            for value in (
                arguments.authorization,
                arguments.output,
                arguments.execution_commit,
            )
        ):
            raise ValueError("final summary execution requires every exact identity")
        assert arguments.authorization is not None
        assert arguments.output is not None
        assert arguments.execution_commit is not None
        result = run_authorized_final_summary(
            root,
            authorization_path=arguments.authorization.resolve(),
            output_path=arguments.output.resolve(),
            execution_commit=arguments.execution_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
