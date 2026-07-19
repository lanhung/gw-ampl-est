#!/usr/bin/env python3
"""Plan or run the separately authorized legacy SIS stress-control verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.config import load_yaml
from gwlens_mm.training.legacy_sis_stress import (
    LegacySISStressContract,
    verify_legacy_sis_stress_control,
    write_legacy_sis_evidence,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
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
            for value in (args.authorization, args.checkpoint, args.predictions, args.output)
        ):
            raise ValueError("legacy SIS execution requires every exact identity")
        authorization = load_yaml(args.authorization)
        if authorization.get("authorization_status") != "authorized_read_only_reproduction":
            raise ValueError("legacy SIS read-only execution gate is absent")
        flags = authorization.get("authorization", {})
        if not (
            flags.get("legacy_asset_read_authorized") is True
            and flags.get("legacy_checkpoint_deserialization_authorized") is False
            and flags.get("legacy_asset_write_authorized") is False
            and flags.get("v2_final_data_application_authorized") is False
            and flags.get("gwosc_gwtc_access_authorized") is False
        ):
            raise ValueError("legacy SIS execution boundary changed")
        identity = authorization.get("frozen_legacy_identity", {})
        metric = authorization.get("descriptive_metric_contract", {})
        contract = LegacySISStressContract(
            checkpoint_sha256=str(identity.get("checkpoint_sha256", "")),
            predictions_sha256=str(identity.get("validation_predictions_sha256", "")),
            validation_rows=int(identity.get("validation_rows", -1)),
            expected_mae=float(metric.get("mae")),
            expected_rmse=float(metric.get("rmse")),
            expected_mape_percent=float(metric.get("mape_percent")),
            expected_pearson=float(metric.get("pearson")),
            numeric_tolerance=float(metric.get("absolute_numeric_tolerance")),
            sis_identity_tolerance=float(metric.get("sis_identity_absolute_tolerance")),
        )
        result = verify_legacy_sis_stress_control(
            args.checkpoint, args.predictions, contract
        )
        write_legacy_sis_evidence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
