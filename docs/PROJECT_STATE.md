# Project state

## Current phase

Phase 1A complete on `phase1/physics-schema`, pending human review. No smoke
generation is authorized by this state update.

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
- passed 65 pytest cases, Ruff, mypy, and an AutoDL SIS import contract.

## Not started by design

- literature review;
- v2 data generation;
- model or posterior training;
- GWOSC/GWTC download;
- manuscript work.

## Next recommended phase

Human review of `docs/reports/PHASE1A_REPORT.md` and the eight specified gate
checks. Only after approval, open Phase 1B to implement pinned SIE/EPL solver
adapters and generate exactly 48 accepted smoke pairs under the new AutoDL
project root. Stop before any model training.
