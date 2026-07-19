# Phase 4 corrected 65k probe authorization report

## Decision

Delegated expert review authorizes exactly three fresh 65,536-system probe fits,
for seeds 0, 1 and 2, followed by the frozen terminal development comparison.

This authorization is recorded in:

`configs/execution/phase4_corrected_65k_probe_training_authorization.yaml`

## Entry evidence

- the five-system correction overlay passed independent closeout;
- the corrected combined view contains exactly 65,536 unique train systems;
- corrected combined-view SHA-256:
  `da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379`;
- all six corrected 16k/32k fits completed with zero process failures;
- the 10,000-replicate decision was independently reproduced byte-for-byte;
- decision: `continue_to_train_65k`;
- decision SHA-256:
  `fe2890e025f5574a4ea45942b698e0b24db3801650125cd5f128126e435633cf`;
- final-evaluation commitment remains sealed at
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.

## Immutable training release

- code: `adcb1a79e1534e4d742238aa99869c57da95dd96`;
- wheel SHA-256:
  `44208c61577b71488872c75eced03dbca3384cf5d03baaecc9f3447bdaeef24a`;
- model configuration hash:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- CUDA environment SHA-256:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

The AutoDL disposable checkout still records the exact code marker, the wheel
hash matches, all four GPUs are idle, the new output root is absent and
222,270,361,600 bytes are free. This exceeds the 100 GB prelaunch requirement.

## Execution boundary

The runner may open only the corrected Stage A, Stage B, combined-base and
replacement publications. It must fit the same frozen model from scratch for
all three seeds, use the unchanged 6,144 validation systems and execute the
same 10,000-replicate terminal rule.

The failed pre-correction 65k root and all superseded checkpoints remain
immutable and forbidden. Architecture selection, model tuning, calibration,
SBC, final evaluation, extension above 65,536, real-noise work and GWOSC/GWTC
access remain unauthorized.

## Terminal behavior

- `lock_train_65k` permits only a later architecture-selection review;
- `stop_data_limited_and_new_preregistration` is a major scientific stop;
- neither state authorizes extension above 65,536.
