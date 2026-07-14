#!/usr/bin/env python3
"""Render the frozen final-evaluation plan without generating any pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwlens_mm.production.final_evaluation import dry_run_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.execute:
        raise PermissionError("final-evaluation materialization remains unauthorized")
    result = dry_run_plan(arguments.root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
