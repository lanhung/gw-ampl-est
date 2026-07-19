#!/usr/bin/env python3
"""Summarize exactly six completed development-only ablation fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gwlens_mm.training.ablations import ABLATION_VIEWS, summarize_ablation_results


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError("ablation summary identity already exists")
    results = tuple(
        _load(
            arguments.ablation_output_root
            / view
            / f"seed-{seed}"
            / "run_summary.json"
        )
        for view in ABLATION_VIEWS
        for seed in (0, 1, 2)
    )
    summary = summarize_ablation_results(results)
    _atomic_json(arguments.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
