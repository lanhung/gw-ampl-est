# Phase 3C-A — Proposal-v3 engineering A/B qualification

Execute only on `phase3ca/proposal-v3-ab-qualification` under the separate
human authorization. Read AGENTS, authorization, adaptive RC.3, v3/v2 configs,
proposal specifications/reports, Phase 3A report, provenance, privileged policy
and ADR-004/005 first.

This is bounded engineering A/B only. Generate exactly 512 RC.5 control and 512
proposal-v3 candidate pairs, 16 blocks of 32 per arm, never over 1,024 total.
No scientific data, training, calibration/evaluation, GWOSC/GWTC, Stage A or
later authorization.

Before execution verify branch/ancestry/hashes/authorization. Integrate only the
pre-selection proposal; all waveform/lens/kinematics/noise/selection/storage
code and environment must match. Candidate retains privileged proposal/density/
weight/seed provenance. Freeze code in a clean pushed commit after local and
AutoDL tests, Ruff, mypy, build, PSD/environment checks.

On AutoDL require exact clean checkout, >=121446475732 free bytes, distinct
parent/control/candidate identities/manifests/journals and no collisions.
Execute sequential matched blocks: even control then candidate; odd candidate
then control. Separate all RNG/ID namespaces.

The first matched 32+32 is official and receives engineering health validation;
do not inspect interim throughput. Per arm caps: 1,000,000 attempts, six active
hours, 512 accepted. Control cap invalidates comparison; candidate cap fails or
is inconclusive. Persist complete block telemetry and partial evidence.

Only after all blocks compute the frozen 10,000-replicate matched-block log
rate-ratio bootstrap with seed domain proposal_v2_throughput_ab_bootstrap_v1.
Pass requires lower 95% bound >=2.0.

Candidate post-selection gates: finite densities/weights; ESS >=0.50 overall,
>=0.40 each family, >=0.25 each EM cell; maximum normalized weight <=0.05;
families/multiplicities/eight cells/tails/support/IDs all valid. Secondary
metrics cannot override throughput.

Use atomic arm/block publication, checksums and resume identity. Final outcome
must be one of the five preregistered states. Commit only small evidence and
report; large arrays stay AutoDL. Run final tests/checks, commit
`feat: qualify proposal v3 in engineering A/B`, push, and stop. Do not create a
PR, start Stage A or train.
