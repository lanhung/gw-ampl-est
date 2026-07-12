"""Run the authorized latent-only proposal-v2 preflight and print JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path

from gwlens_mm.config import load_yaml
from gwlens_mm.proposals.exact_mixture import (
    ProposalSpecification,
    evaluate_latent_preflight,
)
from gwlens_mm.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/proposals/proposal_v2_exact_mixture.yaml"


def main() -> None:
    config = load_yaml(CONFIG)
    specification = ProposalSpecification.from_mapping(config)
    result = evaluate_latent_preflight(specification)
    result["proposal_version"] = config["proposal_version"]
    result["proposal_configuration_hash"] = configuration_hash(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
