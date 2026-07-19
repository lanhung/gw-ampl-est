# Phase 4 corrected 16k/32k probe authorization

## Decision

Delegated scientific and engineering review authorizes one fresh execution of
the preregistered 16,384/32,768 three-seed probe workflow over the corrected
Stage A training view. This is not a resume and no superseded checkpoint or
metric may enter the rerun.

## Evidence reviewed

- correction parent manifest:
  `0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2`;
- correction publication tree:
  `a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12`;
- independent closeout status: passed;
- corrected 32k view:
  `b9390a7faad4bb8097abb09041f1f229e13c6419677926f05642ac611dc6ced2`;
- corrected 65k view:
  `da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379`;
- training implementation commit:
  `adcb1a79e1534e4d742238aa99869c57da95dd96`;
- exact wheel:
  `44208c61577b71488872c75eced03dbca3384cf5d03baaecc9f3447bdaeef24a`;
- CUDA environment:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`;
- model configuration:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- final-evaluation commitment:
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.

Local verification passed 324 tests with seven optional skips, Ruff, mypy and
package build. AutoDL reproduced the metadata-only view and passed all 30
corrected-reader/training/rung focused tests. The installed wheel sees CUDA and
all four RTX 5000 Ada GPUs. Prelaunch free space is 223,748,300,800 bytes.

## Authorized work

The runner may resolve the corrected 32k membership, recompute the frozen
SHA-256-ranked 16k subset, fit rung-specific standardizers, train seeds 0, 1
and 2 from scratch at each rung and apply the unchanged 10,000-replicate paired
learning-curve decision.

## Closed work

Architecture selection, calibration, SBC, final evaluation, Stage B training,
extension above 65,536 systems, real noise and GWOSC/GWTC remain closed. A
fresh `continue_to_train_65k` result is evidence for a separate corrected-65k
gate and is not execution authority.
