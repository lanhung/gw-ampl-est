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
  `d2e4fde7b918ce363ca67781d7a462d97ffe37dd4fadde186f587b44be7cdf7a`;
- status: `ready_for_delegated_terminal_probe_authorization_review`;
- review checkout:
  `d8a3f1153155797921267557672c03d1ea6543a9`.

It binds the independently validated combined publication manifest
`ad26d51d4f9475c6710cdfee4e71409526e1d776e0b8ec14734feff02855cee5`
and the 512-case development-tail manifest
`58fcafd58cbcd407ecf6b35dfa98c0bd2bd66f37151e19e6bf530ca2601260c7`.

## Immutable training identity

- training commit:
  `d8a3f1153155797921267557672c03d1ea6543a9`;
- exact wheel SHA-256:
  `fd8da0465f9609e31805abf01f1bf41dc07b486b8e470a6c345a64923b63dda8`;
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
- focused tests: 70 passed;
- full tests: 486 passed, 3 optional skips;
- CUDA available: true;
- observed GPUs: four NVIDIA RTX 5000 Ada Generation devices;
- editable install: false;
- repository source import: false;
- scientific data opened: false;
- optimizer started: false.

The verifier-result SHA-256 is
`15b39d2d41cbf99b114bd08685844d271761ac14125aec8c05d92cd46ffe3972`.
The exact runtime dependency versions matched the frozen model configuration.
The exact wheel also indexed the real 65,536-system terminal parent with
65,536 unique IDs without opening strain.

## Delegated review and scope

The delegated-review decision SHA-256 is
`d0a91645a351aa20a8e03df561e74b5cdb7420e7c4bcc9d3f584156e3fa4e634`.
It permits:

- read-only access to the exact corrected 65k, terminal increment, combined
  131k, validation and development-tail publications;
- evaluating the retained 65k checkpoints on the new tail pool;
- training the frozen 131k probe from scratch for seeds 0, 1 and 2;
- the preregistered terminal 65k-to-131k comparison.

The fresh output root is:

`/root/autodl-tmp/lensing-4/training/phase4/terminal-probe-131k-d8a3f11-d2e4fde`

The generated authorization SHA-256 is
`3a8a9c2986f800e28698360538aec1d51fdb3f5afb3a3b5dd5c651a43317aee6`.

## Superseded pre-optimizer attempt

The first authorization bound training commit `a261d1a9...` and stopped while
the generic reader indexed the real terminal increment parent. That publisher
uses one singular `validation` mapping, while the reader recognized only the
older plural `validations` mapping. No rung preparation, checkpoint, optimizer
step or scientific metric was created.

The correction accepts both unambiguous layouts and rejects conflicting
declarations. It is an engineering schema-accessor patch only; all data,
model, environment, seeds, metrics and scientific gates are unchanged.

## Stop boundary

The terminal decision must be exactly one of:

- `lock_train_131k_saturated`;
- `lock_train_131k_resource_capped_data_limited`.

Both labels lock the resource cap at 131,072 and stop for a separately
authorized architecture-selection review.
