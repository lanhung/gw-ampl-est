#!/usr/bin/env python3
"""Save fail-closed deployable/privileged input-policy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gwlens_mm.policy import InputPolicy, InputPolicyError

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    allow_path = ROOT / "configs/policy/deployable_input_allowlist.json"
    deny_path = ROOT / "configs/policy/privileged_input_denylist.json"
    policy = InputPolicy.from_files(allow_path, deny_path)
    allowed = tuple(sorted(policy.allowlist))
    policy.validate_model_inputs(allowed)
    denied_failures = []
    for field in sorted(policy.denylist):
        try:
            policy.validate_model_inputs((field,))
        except InputPolicyError:
            denied_failures.append(field)
    alias_fixtures = (
        "trueMu",
        "cleanWaveform",
        "oracleLensParameter",
        "sourcePlaneBetaTrue",
        "optimalSNR",
    )
    rejected_aliases = []
    for field in alias_fixtures:
        try:
            policy.validate_model_inputs((field,))
        except InputPolicyError:
            rejected_aliases.append(field)
    passed = len(denied_failures) == len(policy.denylist) and len(rejected_aliases) == len(
        alias_fixtures
    )
    result = {
        "status": "passed" if passed else "failed",
        "policy_version": policy.version,
        "allowlist_sha256": _sha256(allow_path),
        "denylist_sha256": _sha256(deny_path),
        "allowed_field_count": len(allowed),
        "denied_field_count": len(policy.denylist),
        "all_exact_denials_rejected": len(denied_failures) == len(policy.denylist),
        "alias_fixtures": list(alias_fixtures),
        "all_alias_fixtures_rejected": len(rejected_aliases) == len(alias_fixtures),
        "selection_statistics_deployable": False,
        "importance_weights_deployable": False,
        "truth_deployable": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
