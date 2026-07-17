# Phase 4 16k/32k probe-training and learning-curve report

## Outcome

The six preregistered probe fits completed and the development-only learning-
curve rule returned:

`continue_to_train_65k`

This is evidence for a separately authorized Stage B extension. It is not an
authorization to generate another system, train the 65k rung, fit calibration,
run SBC, open final evaluation, or access GWOSC/GWTC.

## Frozen identities

- Stage A parent: `phase4-stage-a-2be777e727ef-d3a60034bbd6`;
- Stage A parent manifest SHA-256:
  `4f3e6b3a7ca1a995d7a7643c48410e479fb812e4a01ff66537232b9d64bf3314`;
- final-evaluation commitment SHA-256:
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`;
- training commit: `5baabfe229ad187f6bcdcc1dea7cf42aa43c41e9`;
- wheel SHA-256:
  `262b6446cb200f2ae432e0e33ed35d986ad475321d59ea39bcae9cb528b9c393`;
- model configuration SHA-256:
  `8d0919c211b6aa057712a402f689f06d9ea916ba3c0c11cc32d0561aeb8d3087`;
- CUDA environment SHA-256:
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`;
- 16k membership SHA-256:
  `f2dddb02ca034d0f06a751b93cf6601321e9dd90e27c25770609c87f89fdb381`;
- 32k membership SHA-256:
  `b0c28b57350c3cf196c047005fb04f4b3388db0cbcc3caee61016f1ff6e4280b`.

The 16k membership was the lowest SHA-256-ranked half of the complete 32k
training parent. It was not the first 16,384 systems generated.

## Engineering correction and execution

The first execution failed before one optimizer step because a physical batch
of 256 exceeded 32 GB GPU memory. The retained failed root contains no
scientific checkpoint or metric. A reviewed same-phase patch preserved the
effective batch of 256 as four ordered physical microbatches of 64, followed by
one gradient clip and one AdamW step. It did not change the optimizer, learning
rate, sample order, architecture, data, target or stopping rule.

All corrected fits used the new output root:

`/root/autodl-tmp/lensing-4/training/phase4/probe-microbatch-v1`

The root contains about 1.286 GB, including checkpoints and per-case
development tables retained on AutoDL. Seeds 0 and 1 ran on initially free
GPUs. Seed 2 shared a GPU with an unrelated legacy-root process and therefore
completed more slowly, but used the same immutable software and deterministic
identity. The supervisor never exceeded three concurrent fits.

## Fit results

| Rung | Seed | Epochs | Best epoch | Mean validation NLP per target dimension | Median CRPS |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 0 | 36 | 15 | 0.273584 | 0.215847 |
| 16,384 | 1 | 38 | 17 | 0.246713 | 0.196156 |
| 16,384 | 2 | 43 | 22 | 0.275929 | 0.187223 |
| 32,768 | 0 | 57 | 36 | 0.047297 | 0.141060 |
| 32,768 | 1 | 60 | 39 | 0.014667 | 0.136512 |
| 32,768 | 2 | 53 | 32 | 0.025320 | 0.144420 |

Each development evaluation used the same 6,144 validation physical systems
and 1,024 posterior draws per case. No best seed was selected. No post-hoc
calibration was applied.

## Frozen stopping decision

The paired 10,000-replicate physical-system bootstrap measured the 16k-to-32k
NLP improvement per target dimension as:

- point estimate: `0.236314`;
- 95% interval: `[0.223545, 0.248638]`.

Saturation required the upper interval bound to be below `0.01`. The observed
lower bound alone is more than twenty times that threshold. All three seeds
also showed CRPS improvements far above 1%: 34.65%, 30.41% and 22.86%.

The other all-conditions requirements did not support a 32k lock either:

- marginal-coverage-error improvements exceeded `0.005`;
- development EM-cell tolerance was false for each seed;
- maximum EM-cell degradation was 0.0742, 0.0617 and 0.0676, above 0.02;
- the extreme-relative-magnification development view had only 40 cases,
  below the frozen minimum of 128.

These are uncalibrated development diagnostics, not final-test results. They
cannot be repaired by looking at calibration, SBC, IID, OOD or mismatch data.
They establish only that the 32k rung is not saturated under the preregistered
rule.

## Safety verification

- all six run states are `completed_probe_fit_and_development_validation`;
- every fit used the exact Stage A parent and its rung-specific deterministic
  membership;
- each rung was trained from scratch for seeds 0, 1 and 2;
- physical microbatch 64 and gradient accumulation 4 preserved effective batch
  256;
- calibration access was false;
- final-evaluation access was false;
- no model tuning or best-seed selection occurred;
- no GWOSC/GWTC or real-noise data were accessed;
- checkpoints and 36,864-row per-case evidence remain on AutoDL and are not
  committed to Git.

The exact decision JSON has SHA-256
`09dd9a3fb6a36c501d5d76412fe063f1612458ade444ff0c3e12eacd66c4f49e`.
Committed small summaries are enumerated in
`results/phase4/probe/probe_evidence.sha256`.

## Next gate

RC.4 permits only one response to this outcome: request separate Stage B
authorization to add exactly 32,768 new direct-target training physical
systems, making a nested 65,536-system train rung. After atomic Stage B
publication, seeds 0, 1 and 2 must train the 65k probe from scratch and the same
paired stopping rule must compare 32k with 65k.

The decision does not authorize automatic extension beyond 65k. If the 65k
comparison remains clearly data-limited, the project must stop for a new
preregistration rather than silently generate more training data.
