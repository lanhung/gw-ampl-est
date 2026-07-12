#!/usr/bin/env python3
"""Fail unless Bilby design-PSD noise is reproducible for an identical seed."""

from pathlib import Path

import numpy as np

from gwlens_mm.config import load_yaml
from gwlens_mm.smoke.generator import SmokeGenerator


def main() -> None:
    config = load_yaml(Path("configs/data/v2_smoke.yaml"))
    generator = SmokeGenerator(config, "0" * 40)
    first = generator._noise("H1", 1126259462.0, 123456)  # noqa: SLF001
    second = generator._noise("H1", 1126259462.0, 123456)  # noqa: SLF001
    if not np.array_equal(first, second):
        raise RuntimeError("Bilby noise is not deterministic for an identical seed")
    print("bilby_noise_determinism=passed")


if __name__ == "__main__":
    main()
