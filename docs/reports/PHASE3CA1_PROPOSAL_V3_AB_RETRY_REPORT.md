# Phase 3C-A.1 proposal-v3 A/B retry report

## Outcome

Phase 3C-A.1 stopped fail-closed with machine state `execution_failed` when
the RC.5 control arm reached the frozen six-hour active-time cap during block
12 (the thirteenth zero-indexed block). The retry is not a completed A/B and
does not pass or fail proposal-v3 statistically.

The corrected typed alpha.3 health validator passed. The failure is therefore
not a recurrence of the Phase 3C-A metadata-field defect. It is an execution-
cap outcome after substantial staging, with no final publication, bootstrap or
post-selection ESS calculation.

Under the newly adopted one-retry engineering policy, proposal optimization is
now closed. No third A/B run and no proposal-v4 are authorized. The future
scientific route is direct generation from the evaluation target, subject to a
new versioned scientific contract and explicit Stage A execution review.

## Frozen identities

- generator commit:
  `324bab47aff5c4ed2b2041099a103735a40463f0`;
- authorizing commit:
  `da1e7438e30011ee2733963f6441dcd789f2ce0b`;
- parent run:
  `phase3ca1-324bab47aff5-f065e349f166`;
- RC.5 control dataset:
  `gwlens-v2-2.0.0-alpha.3-c86b4bf44a0d54f8-control`;
- proposal-v3 candidate dataset:
  `gwlens-v2-2.0.0-alpha.3-79e330197ad5620c-v3`;
- retry configuration hash:
  `f065e349f166490fb68cb56779c3acaa0961f821d0bb6a2104bca37880f2f34b`.

The failed Phase 3C-A parent and its 32+32 blocks were not resumed, mutated or
counted in this retry.

## Verification before execution

The field correction was centralized in a typed helper using
`evaluation_prior_log_probability`. A real alpha.3 record was serialized
through JSON and a minimal Parquet/Zarr shard before calling the maintained
health path. Local verification passed 193 tests with five optional skips,
Ruff, mypy including the execution script, and package build. The exact
synchronized AutoDL checkout passed 202 tests. All mass-sheet, source-plane,
waveform-boundary, whitening, Galkin and privileged-input preflights passed.

The first new matched block then passed the corrected health gate for both
arms. Its machine record explicitly states `interim_throughput_inspected:
false`.

## Retained staging evidence

Each arm atomically completed 12 blocks of 32 pairs:

| Arm | Complete blocks | Accepted pairs | Complete-block attempts | Complete-block active seconds |
| --- | ---: | ---: | ---: | ---: |
| RC.5 control | 12 | 384 | 143,762 | 19,744.317278 |
| proposal-v3 | 12 | 384 | 64,331 | 9,687.546672 |
| Total | 24 | 768 | 208,093 | — |

Control block 12 remains a partial directory with 13,727 append-only attempt
journal records. It is not complete, is not included in the 768 accepted-pair
count, and may not be resumed or published. Total stopped staging size is
904,836,338 bytes; 323,355,758,592 bytes remained free when inspected.

The run lasted from 2026-07-13 00:40:20 UTC until 09:21:50 UTC, about 8 hours
41 minutes. The complete-block tree hashes are committed in
`results/phase3ca1/stopped_block_hashes.sha256`.

## Statistical boundary

The 10,000-replicate matched-block bootstrap was not run. No throughput ratio,
effective-throughput ratio, accepted-sample ESS, EM-cell ESS or proposal
adoption decision was computed. The partial evidence must not be reanalysed
under the later proposed effective-sample endpoint because that endpoint was
not frozen before this run.

Proposal-v3's latent-only ESS result remains valid engineering evidence, but it
does not convert this incomplete run into a scientific or A/B result. All 768
complete pairs and the partial block are permanently engineering-only.

## Disposition and next scientific gate

The project now separates scientific contracts from implementation patches as
specified in `docs/ENGINEERING_RELEASE_POLICY.md`. Proposal work can no longer
block the main path:

- no further proposal A/B or proposal-v4;
- RC.5 remains an engineering reference, not the default weighted scientific
  training proposal;
- the next scientific preregistration must select direct `p_eval` generation
  for Stage A training, with `q_train = p_eval` and unit weights;
- that change requires a new preregistration version/hash because it changes
  the training-data generation and weighting contract;
- Stage A data generation and all model training remain closed until that
  single scientific review and separate execution authorization.

No scientific data, training, tuning, calibration, SBC, IID/OOD/mismatch
evaluation, real noise, GWOSC or GWTC access occurred.
