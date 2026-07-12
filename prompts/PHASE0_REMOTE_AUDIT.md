# Phase 0 — Repository, remote-data and provenance audit

Work only on Phase 0.

Do not rewrite the model. Do not train a model. Do not generate a large
dataset. Do not download GWTC or GWOSC data. Do not write the manuscript. Do
not modify any legacy remote file. Do not browse the web during this phase.

## Environment

Authoritative local Git repository: `/root/work/lensing-4`

Remote compute host: `ssh autodl-lensing`

New writable remote project root: `/root/autodl-tmp/lensing-4`

Legacy candidate roots, both read-only:

1. `/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main`
2. `/root/autodl-tmp/qkzhang`

Reference documents:

- `docs/reference/baseline_realobs_quality_mu0_report_zh.pdf`
- `docs/audits/REMOTE_DATA_AND_GENERATOR_AUDIT_20260712.md`

## Primary objectives

1. Inspect the local Git repository and current branch.
2. Verify SSH connectivity without exposing credentials.
3. Inspect both legacy candidate roots using read-only commands.
4. Reconcile the two paths and determine whether they contain data, code,
   duplicates, symlinks, or copied snapshots.
5. Locate SIS, point-mass, unlensed, `realobs_quality_mu0`, PDF-baseline,
   waveform-generation, magnification-training, feature-generation, quality
   control and pair-classification assets.
6. Map each dataset to its most likely generator and training scripts without
   treating filename similarity as proof.
7. Inspect Git history where available; record sizes, timestamps and
   lightweight checksums.
8. Estimate large-file hashing cost before computing full checksums.
9. Read NPY headers and small memmap samples only.
10. Identify detector axes/labels, dtype/shape, grouping, noise, waveform,
    sampling distributions, latent truth, observables, leakage and missing
    provenance.
11. Trace whether the PDF baseline used 0222/0228, a later
    realobs-quality dataset, another generator, or an untraceable mixture.
12. Copy no large data. Snapshot only relevant small source/metadata files,
    preserving paths and recording origin, size, mtime and SHA256. Never edit
    copied legacy files.

## Required outputs

- `docs/REMOTE_TOPOLOGY.md`
- `docs/audits/PHASE0_REMOTE_AUDIT.md`
- `docs/audits/PDF_TO_CODE_DATA_TRACEABILITY.md`
- `docs/LEGACY_DATA_DECISION.md`
- `docs/LEGACY_CODE_REUSE_DECISION.md`
- `docs/DATA_PATH_AUTHORITY.md`
- `docs/PROJECT_STATE.md`
- `docs/DECISIONS.md`
- `docs/FAILURES.md`
- `manifests/legacy_file_inventory.csv`
- `manifests/legacy_dataset_inventory.csv`
- `manifests/legacy_code_inventory.csv`
- `manifests/legacy_snapshot_checksums.sha256`
- `results/experiment_registry.csv`

Classify assets as retain unchanged, reuse after tests, rewrite for v2,
smoke-only, privileged diagnostics only, exclude, or unknown.

## Final decision

Decide whether existing data can enter main training/final testing, whether a
generator can be reused unchanged, what conventions can be extracted, what v2
data must be regenerated, the minimum smoke data, storage/compute estimates,
and the next phase.

## Completion

Validate generated CSV/Markdown, inspect `git diff`, update state and commit:

`audit: reconcile legacy AutoDL data and generator provenance`

Stop after Phase 0.
