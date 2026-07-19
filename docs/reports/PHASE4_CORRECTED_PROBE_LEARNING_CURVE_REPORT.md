# Phase 4 corrected 16k/32k probe learning-curve report

## Outcome

All six fresh corrected-view probe fits completed successfully. The frozen
development-only comparison returned:

`continue_to_train_65k`

This result supersedes the earlier 16k/32k learning curve, whose membership
contained numerical waveform pathologies. It authorizes nothing by itself. A
separate exact gate is required before opening the corrected 65k view or
starting a 65k optimizer.

## Immutable execution identity

- training implementation: `adcb1a79e1534e4d742238aa99869c57da95dd96`;
- wheel SHA-256:
  `44208c61577b71488872c75eced03dbca3384cf5d03baaecc9f3447bdaeef24a`;
- model configuration hash:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- CUDA environment SHA-256:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`;
- corrected Stage A view:
  `b9390a7faad4bb8097abb09041f1f229e13c6419677926f05642ac611dc6ced2`;
- 16k membership:
  `0879fb2802688f132e04c97313fced4c465e4152023ad61c5d6812eaa2170a29`;
- 32k membership:
  `5c9f0e890034d4aac6e0f3fe22bdb5bfe567d3203e614460877dea398e9b83d6`;
- unchanged validation manifest:
  `4348e278980bd81033a2db80568444262c165e0dea13f9d57d374f932ca823a3`.

Both rungs were trained from scratch for seeds 0, 1 and 2. No checkpoint or
metric from the superseded run was reused.

## Fit results

| Rung | Seed | Epochs | Best epoch | Development mean NLP/dim | Median CRPS |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 0 | 41 | 20 | 0.237872 | 0.192331 |
| 16,384 | 1 | 38 | 17 | 0.250669 | 0.194627 |
| 16,384 | 2 | 50 | 29 | 0.231003 | 0.182056 |
| 32,768 | 0 | 54 | 33 | 0.022420 | 0.138614 |
| 32,768 | 1 | 61 | 40 | 0.071324 | 0.144713 |
| 32,768 | 2 | 59 | 38 | -0.009747 | 0.136859 |

Every development evaluation used the same 6,144 physical systems and 1,024
posterior draws per case. No best seed was selected. Calibration was not fitted
and no final-evaluation case was opened.

## Frozen stopping decision

The deterministic 10,000-replicate paired physical-system bootstrap measured
the 16k-to-32k NLP improvement per target dimension as:

- point estimate: `0.211849`;
- 95% interval: `[0.200116, 0.223464]`.

Saturation required the upper confidence bound to be below `0.01`; instead,
even the lower bound is twenty times that threshold. All three seeds improved
NLP and median CRPS materially. Development EM-cell tolerances were not all
met, and the extreme-relative-magnification development view retained only 40
cases versus the frozen minimum of 128. These are uncalibrated development
diagnostics, not final scientific results.

The exact decision JSON has SHA-256
`fe2890e025f5574a4ea45942b698e0b24db3801650125cd5f128126e435633cf`.
An independent rerun of the frozen comparison reproduced it byte-for-byte.

## Safety and storage

- all six launcher return codes were zero;
- all run identities use the frozen code, wheel, model and environment;
- the corrected reader excluded both affected Stage A systems and included
  their two replacements before recomputing membership;
- calibration and final-evaluation access flags are false for every fit;
- no GWOSC/GWTC or real-noise product was accessed;
- the AutoDL output occupies 1,288,181,300 bytes;
- 222,270,361,600 bytes remained free at closeout;
- checkpoints and per-case CSV files remain on AutoDL and are not committed.

## Next gate

The only permitted next scale-selection action is a separately authorized
three-seed 65,536-system probe over the already resolved corrected combined
view. The same validation cases and terminal 32k-to-65k decision rule must be
used. Architecture selection, calibration, SBC, final evaluation, extension
above 65,536 and GWOSC/GWTC remain closed.
