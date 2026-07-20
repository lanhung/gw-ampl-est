# Phase 4 terminal-probe release-binding report

## Outcome

The future 131k scientific training gate now requires the exact SHA-256 of a
separately reviewed terminal release packet. Previously the software validated
publication and immutable training artifacts independently, but did not prove
that the authorization was derived from the packet that checked the exact
AutoDL wheel. No terminal publication or checkpoint was opened while closing
this gap.

## Fail-closed binding

Before publication resolution, the gate now requires:

- an absolute, existing release-packet path and matching SHA-256;
- delegated review state
  `accepted_for_exact_terminal_probe_authorization`;
- packet status
  `ready_for_delegated_terminal_probe_authorization_review`;
- packet-level authorization and optimizer flags still false;
- exact 131,072 train and 512 development-tail counts;
- identical combined/train/tail manifest SHA-256 values;
- identical training commit, wheel path/name/hash, model configuration and
  normalized CUDA environment;
- at least three identically named RTX 5000 Ada GPUs;
- exact-wheel evidence path/hash;
- the finalized evaluation-commitment SHA-256;
- architecture, calibration, SBC, final evaluation, extension and GWOSC/GWTC
  flags still false.

The accepted packet SHA is persisted in the 131k rung preparation, each new
seed's preparation and summary, and each retained-65k terminal-tail summary.

## Verification

Synthetic tests cover a valid packet and fail-closed drift in packet hash,
publication identity, wheel identity, GPU identity, non-authorizing status and
delegated-review state. The full gate still defaults closed because no exact
terminal-probe authorization exists.

- terminal release/probe/verifier focused suite: 24 passed;
- full local suite: 419 passed, 7 optional dependency skips;
- Ruff: passed;
- mypy over `src` and all Phase 4 scripts: passed;
- sdist and wheel build: passed.

## Safety

This is software-release hardening only. It authorizes no publication access,
checkpoint access, preprocessing, optimizer, terminal decision, architecture
selection, calibration, SBC, final evaluation, real noise or GWOSC/GWTC use.
The active worker-32 materialization was not synchronized or modified.
