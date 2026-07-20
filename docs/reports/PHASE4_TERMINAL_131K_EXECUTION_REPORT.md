# Phase 4 terminal 131k execution release report

## Outcome

The exact materialization runner for preregistration `1.2.0-rc.1` is frozen at
Git commit `a4e6bac014ccd521d510c97593cb1368e826d5eb`. It supports one resumable,
atomic 65,536-system direct-target train increment, four separate 128-case
development-tail namespaces and one logical corrected 131,072-system training
reference. No new pair had been generated when this release was reviewed.

The runner preserves the corrected 65,536-system view, applies numerical-
waveform validity `1.1.1-rc.1` prospectively, validates exact q=p/unit weights,
streams Parquet/Zarr/journal validation and checks pair, source, lens, physical-
system and noise identities across corrected train, validation, the increment
and all four tail strata before atomic publication.

## Frozen release

- generator commit: `a4e6bac014ccd521d510c97593cb1368e826d5eb`;
- wheel SHA-256: `c7bc8ecadb373ed5d7307ee9c96b131cc68cc9ad8ea10ae2100c54aed2a8958f`;
- execution-configuration identity:
  `abd07c5b8031a5cc9564531d29d9349f65b0918fafc494767fc912b7e7444ed7`;
- dependency-lock SHA-256:
  `792a93f24f6c38c18ec214665d34c8348e042b21beebd177333dae2c30224d8f`;
- AutoDL Python: 3.10.12;
- scientific dependencies: NumPy 1.26.4, Bilby 2.6.0, LALSuite 7.26.1,
  Lenstronomy 1.13.6, Astropy 6.1.7, pandas 2.2.3 and Zarr 2.18.3.

## Verification

- focused terminal/final-context tests: 35 passed locally and 35 passed on
  AutoDL;
- full local suite: 368 passed, seven optional dependency skips;
- full AutoDL suite: 378 passed, one optional Torch skip;
- maintained Ruff: passed;
- mypy: passed for 63 source files;
- sdist and wheel: passed;
- the second-image tail interval is tested as the frozen half-open `[10,12)`
  interval.

## Resource gate

AutoDL reported 221,613,056,000 free bytes. The launch threshold is
201,596,510,484 bytes, based on a conservative 100,980,232,376-byte projected
peak plus the 100 GB post-peak floor and tail reserve. The measured margin is
20,016,545,516 bytes. The runner must repeat this check immediately before
deriving official identities.

## Authorization boundary

Delegated expert review authorizes exactly 65,536 new train systems and 512
development-tail systems. The 131k optimizer, architecture selection,
calibration, SBC, final evaluation, extension above 131,072, real noise and
GWOSC/GWTC remain closed. Passing publication only permits a later exact
three-seed probe-training review.
