#!/usr/bin/env python3
"""Regenerate the checked-in JSON boundary schema from the typed package."""

import json
from pathlib import Path

from gwlens_mm.schema import v2_json_schema

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "v2_metadata_schema.json"


def main() -> None:
    OUTPUT.write_text(
        json.dumps(v2_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
