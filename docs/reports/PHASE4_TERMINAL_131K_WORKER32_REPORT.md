# Phase 4 terminal 131k worker-32 orchestration report

## Outcome

The original 16-worker terminal materialization was stopped at a clean atomic
boundary before any shard completed. All 16 partial shard directories and
their attempt evidence, totaling 631,856,490 bytes, were moved by a same-
filesystem rename into an immutable `interrupted_evidence` root. No published
or complete shard was lost, reused or rewritten.

The scientific generator remains commit
`a4e6bac014ccd521d510c97593cb1368e826d5eb`, exact wheel SHA-256
`c7bc8ecadb373ed5d7307ee9c96b131cc68cc9ad8ea10ae2100c54aed2a8958f`.
The preregistration, execution-configuration hash, root seeds, target counts
and official parent/dataset identities are unchanged.

## Scheduler release

The external process-pool scheduler is frozen at
`8977ca55f13963441afdda831afb190a3872517c`. It permits exactly 32 workers and
records that value plus the orchestration commit in the release certificate,
run manifest, execution segment and final result. A 64-worker request is
rejected before an executor or output directory is created.

This is an engineering scheduling change only. The per-shard generator receives
the original namespace configuration, generator commit, proposal, root seed and
dataset identity byte-for-byte. The scheduler override is not included in the
scientific configuration hash and cannot change accepted membership.

## Verification

- focused terminal tests: 10 passed;
- full local suite: 370 passed, seven optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 62 source files and the orchestration script;
- sdist and wheel build: passed;
- explicit worker-64 rejection test: passed;
- original 16-worker complete/partial counts: 0/16;
- retained interruption evidence: 631,856,490 bytes.

## Capacity decision

AutoDL exposes 64 logical CPUs. During the 16-worker segment every child used
approximately one full CPU and combined child RSS was about 5.15 GiB, while
roughly 54 GiB memory remained available. Thirty-two workers therefore have a
large memory margin and leave half the logical CPUs for the parent, storage,
checksums and operating system. Sixty-four workers would leave no scheduling
or I/O headroom and has no reviewed concurrency canary, so it is not authorized.

The 32-worker restart must be monitored through its first complete shard. Any
resource, checksum, schema or atomic-publication failure remains fail-closed.
Training, architecture selection, calibration, SBC, final evaluation,
extension above 131,072 and GWOSC/GWTC remain closed.
