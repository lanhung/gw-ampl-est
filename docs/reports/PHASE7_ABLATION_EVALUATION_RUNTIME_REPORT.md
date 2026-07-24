# Phase 7 ablation evaluation runtime report

## Outcome

The execution software for the frozen RC.8 ablation calibration and IID
analysis is implemented without opening a scientific artifact or creating a
runtime authorization.

## Frozen execution configuration

The configuration
`configs/inference/phase7_ablation_evaluation.yaml` has canonical hash:

`1fb19fe9bfcf451919196b0510fad471c507ad5220bbc1410ebd196d00b20dcd`

It fixes:

- GW-only and EM-only views;
- retained model seeds 0, 1 and 2;
- six calibration jobs on the same 4,096 cases;
- six IID jobs on the same 8,192 cases;
- 4,096 transient posterior draws per case;
- physical batch 16 and draw microbatch 256;
- twelve collision-free view/seed/stage RNG domains;
- no best-seed selection, posterior-draw persistence or result-driven
  retraining.

## Shared-kernel implementation

The calibration runtime uses the existing primary calibration score kernel.
The IID runtime uses the existing primary final-inference score kernel. Typed
dataset adapters apply the frozen input view after primary standardization, so
no posterior-score or calibrated-region mathematics is duplicated.

One future authorized calibration job:

1. validates the exact view/seed checkpoint, publication, environment and
   fresh output identities;
2. scores all 4,096 calibration cases;
3. fits exactly one matching split-conformal map;
4. atomically writes the score, map and summary;
5. records that IID, SBC and training were not accessed.

One future authorized IID job:

1. validates the sealed 8,192-case IID namespace, exact checkpoint, matching
   map and same-seed primary score;
2. revalidates the sealed parent manifest and generator identity;
3. runs calibrated ablation inference with transient draws;
4. writes one immutable score artifact;
5. computes one deterministic 10,000-replicate paired comparison;
6. records that no non-IID ablation or retraining occurred.

Both CLIs default to dry-run. `--execute` cannot bypass the reviewed
authorization status or any hash/path check.

## Verification

- focused tests: 8 passed;
- full suite: 556 passed, 8 optional-dependency skips;
- maintained Ruff: passed;
- mypy over 79 source files: passed;
- sdist and wheel build: passed.

The focused suite includes synthetic end-to-end atomic writes for both
calibration and IID paths. Model/data kernels are controlled fixtures; no
scientific checkpoint, calibration record, IID record, primary score or
GWOSC/GWTC product was opened.

## Remaining gate

Runtime execution remains closed. Terminal size and architecture must lock,
the six ablation fits and primary calibration/SBC must complete, and the
two-stage release packets must receive separate exact reviews before these
CLIs may execute scientifically.
