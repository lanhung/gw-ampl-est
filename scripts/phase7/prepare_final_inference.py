#!/usr/bin/env python3
"""Emit the execution-disabled final inference plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.training.final_inference import (
    dry_run_plan,
    run_authorized_final_inference,
    write_dry_run_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--publication-root", type=Path)
    parser.add_argument("--namespace-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--psd-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.execute:
        required = (
            arguments.output,
            arguments.authorization,
            arguments.publication_root,
            arguments.namespace_id,
            arguments.seed,
            arguments.environment_lock,
            arguments.psd_root,
        )
        if any(value is None for value in required):
            raise ValueError("final inference execution requires every identity")
        result = run_authorized_final_inference(
            arguments.root.resolve(),
            authorization_path=arguments.authorization.resolve(),
            publication_root=arguments.publication_root.resolve(),
            namespace_id=str(arguments.namespace_id),
            seed=int(arguments.seed),
            environment_lock_path=arguments.environment_lock.resolve(),
            psd_root=arguments.psd_root.resolve(),
            output_path=arguments.output.resolve(),
            device_name=str(arguments.device),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    plan = dry_run_plan(arguments.root.resolve())
    if arguments.output is not None:
        write_dry_run_plan(arguments.root.resolve(), arguments.output.resolve())
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
