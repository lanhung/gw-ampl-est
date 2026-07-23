# Phase 4 terminal 131k probe authorization

## Outcome

Delegated scientific and engineering review authorizes the frozen
131,072-system terminal probe for seeds 0, 1 and 2. The authorization is
machine-readable at:

`configs/execution/phase4_terminal_131k_probe_authorization.yaml`

This gate does not authorize model tuning, architecture selection,
calibration, SBC, final evaluation, extension above 131,072, real noise or
GWOSC/GWTC access.

## Exact release evidence

The release packet is:

- path: `results/phase4/terminal_probe_release_packet.json`;
- SHA-256:
  `286b8e505b2f69465b1a13dc8e6e0e5921af32840991fbce3b01a0132ce54eb2`;
- status: `ready_for_delegated_terminal_probe_authorization_review`;
- review checkout:
  `134cad27e7b78c986b5a5c6c41e8dd9bf68b1c49`.

It binds the independently validated combined publication manifest
`ad26d51d4f9475c6710cdfee4e71409526e1d776e0b8ec14734feff02855cee5`
and the 512-case development-tail manifest
`58fcafd58cbcd407ecf6b35dfa98c0bd2bd66f37151e19e6bf530ca2601260c7`.

## Immutable training identity

- training commit:
  `a261d1a9fa390313e2f0821e8e75c5f224b759cb`;
- exact wheel SHA-256:
  `1484036f774d6119abdac468bbea5dd911273e6778a0781e48edab7b4a98332e`;
- model configuration SHA-256:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- normalized CUDA environment SHA-256:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`;
- final-evaluation commitment SHA-256:
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.

All three retained corrected-65k `run_summary.json` and `best.ckpt` files are
individually hash-bound in the release packet. Their shared training manifest,
standardizers, model, environment and final-evaluation commitment also match.

## AutoDL wheel verification

The exact installed wheel passed the post-publication verifier:

- status: `passed_exact_wheel_on_autodl`;
- focused tests: 68 passed, 1 optional skip;
- full tests: 482 passed, 6 optional skips;
- CUDA available: true;
- observed GPUs: four NVIDIA RTX 5000 Ada Generation devices;
- editable install: false;
- repository source import: false;
- scientific data opened: false;
- optimizer started: false.

The verifier-result SHA-256 is
`f5ee1a3411cc8816580c6fc5798463966d27bf215855b94fe0d14b845105a625`.
The first verifier invocation stopped because the isolated training
environment did not contain pytest. Its evidence is retained. Pytest was added
only to a separate verification harness; the frozen training environment and
wheel were not modified.

## Delegated review and scope

The delegated-review decision SHA-256 is
`83676256c2b94c5a3921b3bcdd8d6f0b9f591be6097ba9463a153ec44eefda82`.
It permits:

- read-only access to the exact corrected 65k, terminal increment, combined
  131k, validation and development-tail publications;
- evaluating the retained 65k checkpoints on the new tail pool;
- training the frozen 131k probe from scratch for seeds 0, 1 and 2;
- the preregistered terminal 65k-to-131k comparison.

The fresh output root is:

`/root/autodl-tmp/lensing-4/training/phase4/terminal-probe-131k-a261d1a-286b8e5`

The generated authorization SHA-256 is
`2cca4283c351e875435ffe927258c1e6e1a09b49ec84113ca3821c6d68788034`.

## Stop boundary

The terminal decision must be exactly one of:

- `lock_train_131k_saturated`;
- `lock_train_131k_resource_capped_data_limited`.

Both labels lock the resource cap at 131,072 and stop for a separately
authorized architecture-selection review.
