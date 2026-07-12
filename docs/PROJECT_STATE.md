# Project state

## Current phase

Phase 0 complete: repository, remote-data and provenance audit.

## Completed

- established Vultr as the sole authoritative Git repository;
- configured dedicated key-based AutoDL access and safe sync scripts;
- created isolated AutoDL project directories under `/root/autodl-tmp/lensing-4`;
- reconciled qkzhang, wjx pair-verification and `/tmp` PDF-baseline lineages;
- traced PDF metrics to exact data/code/checkpoint evidence;
- created curated immutable source snapshots and manifests;
- classified legacy datasets and code for retain/reuse/rewrite/exclude decisions;
- documented v2-smoke scope and storage gate.

## Not started by design

- literature review;
- v2 physics implementation;
- v2 data generation;
- model or posterior training;
- GWOSC/GWTC download;
- manuscript work.

## Next recommended phase

Phase 1 should define the physics API, quantity conventions, privileged-input
denylist and v2 dataset schema, then implement unit tests. Stop before full data
generation; generate only 10–100 smoke pairs after the schema passes review.
