# Phase 0 remote-data and provenance audit

Audit date: 2026-07-12 UTC. Phase 0 was read-only with respect to all legacy
roots. No training, data generation, package installation, web browsing or GWTC
download was performed.

## Outcome

The three questions motivating Phase 0 are now answered:

1. **What are the old data?** qkzhang contains original 0222/0228 single-channel
   Gaussian-noise catalogs. The wjx repository consumes them for pair verification
   and separately holds later H1/L1 Gaussian regenerations. The PDF uses a distinct
   ET single-detector, one-second-window SIS dataset under `/root/autodl-tmp/tmp`.
2. **What code/data produced the PDF baseline?** The exact generator, training
   wrapper, shared module, dataset, checkpoint and validation table were found;
   checkpoint metrics reproduce the PDF values.
3. **What is reusable?** Physics conventions, truth/observable separation, seed
   separation and manifest/split ideas are reusable after tests. No dataset or
   generator is reusable unchanged for the v2 scientific claim.

## Inspected assets

- local repository, PDF and Git state;
- both declared candidate roots, including inode/link/Git checks;
- wjx code, documentation, manifests, generator history and major data versions;
- qkzhang 0222/0228 schemas, distributions and data-gen-check scripts;
- the PDF-declared `/root/autodl-tmp/tmp` generator/data/training/run chain;
- NPY headers and selected memmap rows only; CSV metadata were inspected in full
  where required.

Detailed inventories are in `manifests/`. Sixty small legacy source/metadata files
were snapshotted with SHA256; compiled caches are ignored and excluded from the
manifest.

## Major findings

### qkzhang 0222/0228

- 0222/0228 are distinct IID simulations; the downstream committed audit records
  full hashes and zero exact source-row overlap.
- main arrays have no detector axis and a single unproven channel identity.
- SIS identities hold, but exact y/beta/theta fields are privileged.
- noise is stationary Gaussian design noise, not GWOSC strain.
- seed 6130 (SIS) and 614 (PM) reproduce the 0222 lens sequences; 0228 seed
  metadata remain unavailable.

### wjx pair-verification project

- independent Git repository at commit `8326ea0`, remote
  `git@github.com:zqk-7k/lensing_classfication.git`.
- scientific scope is pair verification, explicitly not magnification posterior
  inference, catalog FAR or real noise.
- committed docs call 0222 training and 0228 held-out IID evaluation.
- later `final_v3`/`ligo_reduced` arrays have shape `(2500,2,98304)` with H1/L1;
  `ligo_full` has 10,000 sources and consumes about 580 GB.
- `final_v3` and `ligo_reduced` checked metadata and waveform samples are copies,
  not independent evidence.

### PDF point-regression baseline

- exact dataset: 2,500 SIS pairs, ET, Gaussian design noise, one-second float32
  windows `(2500,4096)`.
- it is A21-stratified, not mu0-balanced, and has a sparse high-magnification tail.
- checkpoint reports only model-selected validation performance, not a posterior
  and not an untouched test result.
- its observed geometry/timing are simulated and nearly analytic under SIS.
- `realobs` denotes observation proxies, not real detector data.

## Provenance gaps retained as failures

- no Git history for qkzhang root or `/root/autodl-tmp/tmp`;
- no exact February generator commit/environment for qkzhang arrays;
- no 0228 seed metadata;
- no proof of the detector identity in qkzhang single-channel arrays;
- local and remote PDF binaries differ, though their core content matches;
- no final independent test for the PDF checkpoint.

## Storage and compute gate

Only 321 GB is free on AutoDL. No large regeneration may start until a written
estimate, chunked schema and cleanup-independent storage plan exist. v2-smoke is
approved conceptually (<5 GB, 10–100 pairs); all larger stages remain deferred.

## Decision

Freeze all legacy assets. Start Phase 1 with a tested physics API and v2 schema,
not model training. The first implementation milestone should produce only the
v2-smoke dataset and analytic baselines after explicit approval of its config.
