# Phase 6 materialization and score-stack implementation report

Date: 2026-07-18

## Outcome

The implementation-only Phase 6 extension is complete. It adds a fail-closed
future materialization path for exactly 4,096 calibration-fit and 2,048 SBC
direct-target systems, plus a separately gated selected-checkpoint score
extraction path. No Phase 6 dataset, official identity, model access,
calibration fit or SBC execution occurred.

The data-plan configuration hash is:

`c55dd46d1afefe60753e2b112363261015ea914d55e80c4a5108721cb0b6a17e`

## Frozen implementation behavior

- calibration-fit: 4,096 systems, 32 shards, 512 per EM cell;
- SBC diagnostic: 2,048 systems, 16 shards, 256 per EM cell;
- direct evaluation-target draws with exact unit weights;
- distinct deterministic seeds, attempt namespaces and dataset identities;
- atomic shard and common-parent publication;
- grouped-ID disjointness within Phase 6 and against atomic Stage A/Stage B;
- release-time wheel, environment, PSD, disk, decision and manifest binding;
- selected architecture and all three retained seeds, never a best seed;
- 4,096 posterior draws per calibration case and 1,024 per SBC replicate;
- bounded posterior-draw microbatches with one encoded context per physical
  batch and streaming PIT, HPD and rank counts rather than retained draw cubes;
- immutable score artifacts carrying split, seed, architecture, checkpoint,
  publication and inference-code identities;
- six disjoint inference RNG namespaces across two splits and three seeds;
- no calibration-map fitting or SBC statistical test in score extraction.

## Audit corrections completed before release

The release gate now resolves atomic parent publications rather than trusting
arbitrary child paths. It validates Stage A/Stage B counts, hashes, direct-target
semantics and group-disjoint evidence before a future execution can become
ready. The statistics runner validates unique and disjoint calibration/SBC
physical-system IDs and rejects mixed seed, architecture, checkpoint,
publication or inference-code identities. The post-run disk floor is checked
before atomic publication.

## Verification

- full pytest: 286 passed, 6 optional dependency skips;
- Phase 5/6 focused tests: 24 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 58 source/script files;
- sdist and wheel: passed;
- dry plan: 6,144 systems, 48 shards, official identities `null`;
- scientific pairs generated: 0;
- checkpoint access: 0;
- calibration maps fitted: 0;
- SBC tests executed: 0;
- final evaluation and GWOSC/GWTC access: 0.

## Remaining gates

Stage B must first atomically publish. The 65k three-seed probe must then run
and lock the terminal training size, followed by the frozen architecture grid.
Only a later exact materialization authorization may create the two Phase 6
datasets. Checkpoint inference and calibration/SBC statistics each require
their own identity-bound authorization after materialization.
