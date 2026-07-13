# Phase 3C-A.1 — Correct health validator and rerun proposal-v3 A/B

Execute Phase 3C-A.1 only on
`phase3ca1/proposal-v3-ab-retry`.

Read `AGENTS.md`, both Phase 3C-A authorization files, the failed-run report
and evidence, alpha.3 schema, A/B runner, production validation code and tests
before work.

This is a new full engineering retry, never a resume. Preserve the failed
parent `phase3ca-185e68d4346d-0ed6958442da` and its 32+32 blocks unchanged,
unpublished and excluded from every statistic and split.

Only correct the health validator from nonexistent
`evaluation_log_probability` to alpha.3
`evaluation_prior_log_probability`. Do not add an alias or alter schema,
proposal-v3, adaptive RC.3, physics, waveform, selection or density semantics.
Move the finite distribution-provenance check into a typed reusable helper
shared by health and final validation.

Add a regression that serializes and deserializes a real alpha.3 `V2Record`,
writes a minimal Parquet/Zarr complete shard fixture and invokes the actual
first-block health path. Prove finite values pass, nonfinite values fail and
the obsolete attribute is absent from maintained execution code. Include this
path in mypy coverage and run a dry health validation before generation.

Create `configs/data/phase3ca1_proposal_v3_ab_retry.yaml` with control seed
2026071213, candidate seed 2026071214, unchanged bootstrap seed/method, 512
pairs per arm, 16 blocks of 32, exactly 1,024 new pairs, new prefixes and
namespaces, and all scientific/use flags false. Reference the Phase 3C-A.1
authorization and reject every failed-run identity.

Before execution run full/focused pytest, maintained Ruff, mypy including the
health path, build, inspect the diff, then commit and push a clean generator
commit as `fix: validate alpha.3 distribution metadata in A/B health gate`.
No code change is allowed after generation begins.

Safely sync the exact commit to AutoDL without `rsync --delete`. Repeat every
dependency, PSD, disk, mass-sheet, source-plane, waveform-boundary, whitening,
Galkin and input-policy preflight and verify new identities do not collide.

From zero generate exactly 512 new RC.5 control and 512 new proposal-v3
engineering pairs. After the first new 32+32 blocks run the corrected health
gate without inspecting interim throughput or ESS. Preserve the original
10,000-replicate matched bootstrap, 95% lower-bound threshold 2.0, all
post-selection ESS/support gates, caps, telemetry and publication rules.

Publish only after all 32 arm blocks and final gates pass. Commit only small
evidence under `results/phase3ca1/` and create
`docs/reports/PHASE3CA1_PROPOSAL_V3_AB_RETRY_REPORT.md`. Final outcome must be
one frozen A/B state. Even a pass does not authorize Stage A.

Run final pytest, focused tests, Ruff, mypy, build, checksum/count/identity and
leakage validation. Commit evidence as
`feat: rerun proposal v3 engineering A/B qualification`, push, and stop for
human review. Do not create a PR, train, generate scientific data, access
GWOSC/GWTC or start Stage A.
