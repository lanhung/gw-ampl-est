# Phase 4 corrected 65k terminal-probe closeout report

## Outcome

All three fresh corrected-view 65,536-system probe fits completed successfully.
The frozen terminal development-only comparison returned:

`stop_data_limited_and_new_preregistration`

The run itself passed. The stop is a preregistered scientific conclusion, not
an execution failure. It forbids automatic extension above 65,536 systems and
does not permit post-lock architecture selection because the training size did
not lock.

## Immutable identity

- training code: `adcb1a79e1534e4d742238aa99869c57da95dd96`;
- wheel SHA-256:
  `44208c61577b71488872c75eced03dbca3384cf5d03baaecc9f3447bdaeef24a`;
- model configuration hash:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- CUDA environment SHA-256:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`;
- corrected 65k train manifest:
  `da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379`;
- membership SHA-256:
  `2f20908b1494f0e9119d9c6995cbb84cb3d62c0306d523f0512d3e6803fa1eef`;
- validation manifest:
  `4348e278980bd81033a2db80568444262c165e0dea13f9d57d374f932ca823a3`.

All three fits used the same 65,536 training systems, unchanged 6,144-case
validation set and finalized pre-training evaluation commitment. Calibration
and final-evaluation access remained false.

## Fit results

| Seed | Epochs | Best epoch | Development mean NLP/dim | Median CRPS |
|---:|---:|---:|---:|---:|
| 0 | 52 | 31 | -0.157339 | 0.119213 |
| 1 | 68 | 47 | -0.185047 | 0.118131 |
| 2 | 64 | 43 | -0.177928 | 0.114967 |

Every launcher return code was zero. No best seed was selected.

## Terminal learning-curve decision

The deterministic 10,000-replicate paired physical-system bootstrap measured
the 32k-to-65k NLP improvement per target dimension as:

- point estimate: `0.201437`;
- 95% interval: `[0.191498, 0.211788]`.

The saturation rule required the upper bound to be below `0.01`. Instead, the
lower bound is more than nineteen times the threshold, so the probe remains
clearly data limited. All three seeds improved mean NLP and median CRPS.

The other saturation requirements also did not all pass:

- no seed passed every development EM-cell tolerance;
- the maximum EM-cell coverage degradation was `0.01693`, `0.04622` and
  `0.04818` for seeds 0, 1 and 2;
- the extreme-relative-magnification development view contained only 40 cases,
  below the frozen minimum of 128.

These are uncalibrated development diagnostics, not final-test results. The
final evaluation was never opened.

The exact decision JSON has SHA-256
`90c238a0d85d941c9e90a68e8a215a8d9025f57ffe7757ff89dd14c267f6d72f`.
An independent replay of the frozen comparison reproduced the JSON
byte-for-byte.

## Storage and evidence

- AutoDL output: 656,310,117 bytes;
- free bytes after closeout: 221,613,719,552;
- launcher summary SHA-256:
  `75962ac4933c8ee1e300f1196dc2b6b3f16437b3f432ed3c6038e0e95d601674`;
- seed summary SHA-256 values:
  `13ae9e81...`, `ac905fad...`, `a2b59dc2...`;
- checkpoint and per-case development files remain on AutoDL and are recorded
  by hash only.

No GWOSC/GWTC product, real noise, calibration set, SBC set or final-evaluation
case was accessed.

## Repository verification

- pytest: 353 passed, 7 optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: 61 source files passed;
- sdist and wheel: passed;
- local JSON and committed evidence hashes: passed;
- the known 18 Ruff findings in the untouched Phase 0 manifest script remain
  outside maintained scope and are not represented as a repository-wide pass.

## Required next scientific gate

RC.4 explicitly forbids an automatic rung above 65,536. Architecture selection
also cannot begin because its execution gate requires `lock_train_65k`.

The project must now stop for a new scientific preregistration that decides,
before further execution, whether to:

1. authorize a larger direct-target nested training rung and repeat the frozen
   learning-curve comparison; or
2. lock a narrower claim/model scope under a newly justified stopping rule.

Until that contract is reviewed, architecture fitting, calibration, SBC, final
evaluation, extension above 65,536, real noise and GWOSC/GWTC remain closed.
