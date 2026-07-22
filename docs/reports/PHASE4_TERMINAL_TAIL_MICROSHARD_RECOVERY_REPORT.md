# Phase 4 terminal-tail microshard recovery implementation

## Outcome

The fixed `32 shards × 4 cases` recovery was stopped prospectively and a
deterministic `128 shards × 1 case` implementation is ready for release
freezing. No scientific contract, pair count or distribution changed, and the
new runner has not generated a pair.

## Evidence for stopping the fixed layout

The fixed-layout parent completed the high-absolute-magnification stratum:

- 128 accepted cases;
- 32 complete atomic shards;
- 809,914 attempts.

The next stratum stopped at:

- six partial extreme-relative-magnification cases;
- 328,134 attempts;
- zero complete shards.

The independent terminal-train audit contained 347 extreme-relative cases in
24,636,731 attempts. Using its one-sided 95% rate upper bound and each live
worker's observed speed, the joint probability of all 32 fixed shards reaching
four cases by the worker caps was `1.4141494186098862e-26`.

The failed parent was atomically removed from active staging and retained at:

`/root/autodl-tmp/lensing-4/data_v2/scientific/terminal_131k/interrupted_evidence/tail-parallel32-resource-stop-20260722T0002Z`

Its tree SHA-256 is
`2866e66739aa26f70e560bc8bacb196baccc2406acbcb64719d5b4a2338a253a`;
its byte count is 1,058,865,790. Reuse and deletion are both forbidden.

## Replacement implementation

Each of the four frozen conditional strata contains:

- 128 accepted cases;
- 128 one-case atomic shards;
- deterministic root seed and disjoint attempt sequences;
- a fresh parent, dataset, ID prefix and attempt namespace.

The existing process pool dynamically schedules 128 independent shard tasks
over the 32 physical CPU cores. Task completion order cannot affect any pair
identity because each shard owns its accepted index. This avoids pooled
arrival-order nondeterminism while removing the fixed four-case quota.

The complete science payload remains exactly 512 development-only systems.
It remains excluded from training, architecture selection, calibration and
final claims.

## Resource contract

The 65,536-system terminal train increment fixes the expected attempt cost:

| Stratum | Observed train cases | Attempt probability |
| --- | ---: | ---: |
| high absolute magnification | 4,506 | 1.828976e-4 |
| extreme relative magnification | 347 | 1.408466e-5 |
| second image near threshold | 20,225 | 8.209287e-4 |
| extreme profile/environment | 2,500 | 1.014745e-4 |

The projected total is approximately 11.2 million attempts and 42 active
hours. The implementation freezes:

- 32 physical workers;
- 2,000,000 attempts per microshard;
- 96 hours per microshard and overall tail hard cap;
- 25 GB maximum tail publication;
- 125 GB prelaunch and 100 GB post-run free-space gates.

These are conservative execution caps, not expected runtime claims.

## Verification

- full pytest: 477 passed, 7 optional skips;
- focused microshard/closeout/reader tests: 34 passed;
- Ruff: passed;
- mypy: passed;
- sdist and wheel build: passed.

The generator and production physics files are not modified. The later release
gate must compare the original generator wheel with the recovery wheel
byte-for-byte before execution.

## Gate state

Delegated engineering review authorizes the exact 32-worker microshard
execution. The frozen orchestration commit is
`adb4c0981fd15a809005212c76dd972a59822489`; its wheel SHA-256 is
`a5b08e40ddcff7d542a68b195d5bfc52577e2a67a8a978e374e1d7581f1e4b52`.
The frozen generator-core manifest is
`ebb900d52719dd570e378b63a6d2178b5b47a4b4ed6326769fa55e486b6ebda5`,
proving that the original production generator and physics files are
byte-identical in the recovery wheel.

The AutoDL host exposes 64 logical CPUs but 32 physical cores. With about
61 GB available memory and about 140 GB free disk at authorization, 64 workers
would remove CPU headroom and materially increase memory and I/O failure risk.
It is therefore not authorized. The official scheduler uses 32 workers.

The authorization permits exactly 512 development-only cases and atomic
combined-131k publication. Training, architecture selection, calibration,
SBC, final evaluation, real noise and GWOSC/GWTC remain closed.
