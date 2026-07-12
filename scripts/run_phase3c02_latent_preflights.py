"""Run Phase 3C-0.2 RC.5 and proposal-v3 latent-only diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from gwlens_mm.config import load_yaml
from gwlens_mm.proposals.target_anchored import (
    TargetAnchoredSpecification,
    ess_certificate,
    run_factorwise_diagnostic,
    run_rc5_diagnostic,
    run_v3_preflight,
)
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/proposals/proposal_v3_target_anchored_mixture.yaml"


def main() -> None:
    config = load_yaml(CONFIG)
    specification = TargetAnchoredSpecification.from_mapping(config)
    output = {
        "proposal_version": config["proposal_version"],
        "proposal_configuration_hash": configuration_hash(config),
        "ess_certificate": ess_certificate(config),
        "rc5_baseline": run_rc5_diagnostic(specification),
        "proposal_v3": run_v3_preflight(specification),
        "factorwise": {
            "rc5": run_factorwise_diagnostic(specification, "rc5"),
            "proposal_v3": run_factorwise_diagnostic(specification, "proposal_v3"),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
