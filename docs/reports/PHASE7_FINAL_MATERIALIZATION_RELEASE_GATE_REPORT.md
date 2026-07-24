# Phase 7 final-materialization release gate

## Outcome

The sealed final-evaluation preflight and runner now match the repository's
real atomic publication layout. A dataset child is bound to the exact
SHA-256 of its common parent's `dataset_manifest.json`; the child is not
required to contain a fabricated duplicate manifest.

## Frozen structured references

The future 131,072-system train reference consists of five child datasets:

1. corrected Stage A train base;
2. corrected Stage B train base;
3. Stage A replacement child;
4. Stage B replacement child;
5. terminal 65,536-system increment.

The five rejected waveform-pathology physical-system IDs remain explicit
exclusions. Validation, calibration-fit and SBC each bind one separate child.
The required role counts remain 131,072, 6,144, 4,096 and 2,048.

## Release sequence

After the terminal size and architecture decisions and the calibration/SBC
publication exist:

1. build and validate the structured reference catalog without opening
   strain;
2. build a non-authorizing release packet binding the decisions, catalog,
   exact wheel and environment;
3. perform a separate delegated review of that packet;
4. create the materialization-only authorization;
5. run the existing release certificate before any official identity is
   derived.

The materialization remains sealed. Unsealing, checkpoint and calibration-map
access, scientific inference, training and GWOSC/GWTC are separate gates.

## Verification

- final-evaluation and release-control focused tests: 25 passed, one optional
  PyArrow-dependent test skipped;
- full test suite: 510 passed, eight optional-dependency skips;
- atomic parent drift and child-root reuse fail closed;
- maintained-scope Ruff passed;
- mypy passed;
- sdist and wheel build passed;
- no scientific publication, checkpoint, score or metric was accessed.
