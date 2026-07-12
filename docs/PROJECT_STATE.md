# Project state

## Current phase

Phase 1A.1 schema hardening is complete on `phase1/physics-schema`, pending PR
and human review. No smoke generation is authorized by this state update.

## Completed

- established Vultr as the sole authoritative Git repository;
- configured dedicated key-based AutoDL access and safe sync scripts;
- created isolated AutoDL project directories under `/root/autodl-tmp/lensing-4`;
- reconciled qkzhang, wjx pair-verification and `/tmp` PDF-baseline lineages;
- traced PDF metrics to exact data/code/checkpoint evidence;
- created curated immutable source snapshots and manifests;
- classified legacy datasets and code for retain/reuse/rewrite/exclude decisions;
- documented v2-smoke scope and storage gate.
- defined the authoritative v2 physics and Fourier/Morse conventions;
- implemented and tested the SIS analytic control and general solver protocol;
- implemented fail-closed input policies and grouped split validation;
- implemented the v2 logical metadata schema, provenance, seeds, and manifests;
- accepted ADR-001 for Zarr v2 plus Parquet smoke storage;
- validated an execution-disabled 48-pair smoke specification;
- hardened EM astrometry with explicit physical image IDs;
- aligned detector-noise provenance to every selected image and detector slot;
- replaced the bare timing float with a typed uncertainty-bearing product;
- enforced complete extra-image status and primary-definition semantics;
- added zero-fill/decomposition and complete-manifest validators;
- upgraded the unmaterialized schema to `2.0.0-alpha.2`;
- passed 93 pytest cases, Ruff, mypy, package build, the prior AutoDL SIS
  contract, and the AutoDL alpha.2 metadata contract.

## Not started by design

- literature review;
- v2 data generation;
- model or posterior training;
- GWOSC/GWTC download;
- manuscript work.

## Next recommended phase

Open a PR from `phase1/physics-schema`, require GitHub Actions, and review the
alpha.2 schema diff plus `docs/reports/PHASE1A1_SCHEMA_HARDENING_REPORT.md`.
Only after merge to main should a new Phase 1B branch authorize pinned SIE/EPL
solver adapters and exactly 48 accepted smoke pairs. Stop before training.
