#!/usr/bin/env python3
"""Plan or run the separately authorized legacy SIS stress-control verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.legacy_sis_authorization import (
    run_authorized_legacy_sis_reproduction,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result: Mapping[str, Any]
    if not args.execute:
        result = {
            "status": "implementation_ready_legacy_asset_read_blocked",
            "legacy_asset_read": False,
            "legacy_asset_write": False,
            "checkpoint_deserialized": False,
            "v2_final_data_accessed": False,
            "gwosc_gwtc_accessed": False,
        }
    else:
        if any(
            value is None
            for value in (
                args.authorization,
                args.checkpoint,
                args.predictions,
                args.output,
                args.execution_commit,
            )
        ):
            raise ValueError("legacy SIS execution requires every exact identity")
        assert args.authorization is not None
        assert args.checkpoint is not None
        assert args.predictions is not None
        assert args.output is not None
        assert args.execution_commit is not None
        result = run_authorized_legacy_sis_reproduction(
            args.root.resolve(),
            authorization_path=args.authorization.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            predictions_path=args.predictions.resolve(),
            evidence_output_path=args.output.resolve(),
            execution_commit=args.execution_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
