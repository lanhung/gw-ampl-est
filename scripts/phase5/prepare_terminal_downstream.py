#!/usr/bin/env python3
"""Write the execution-disabled terminal downstream handoff plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gwlens_mm.training.terminal_downstream import dry_run_terminal_downstream_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    result = dry_run_terminal_downstream_plan(args.root)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    partial = args.result.with_name(args.result.name + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
