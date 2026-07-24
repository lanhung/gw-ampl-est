# Phase 7 legacy SIS read-only release-gate report

## Outcome

The implementation-only release gate passed. Commit
`be669fab86ce5f251c56732da6187c3f633e8e8b` provides the exact release,
review and runtime-control chain for the frozen legacy SIS descriptive stress
control.

No legacy asset was read. No scientific data, checkpoint, final-evaluation
case or GWOSC/GWTC product was opened. Read-only reproduction remains
unauthorized until the future exact wheel evidence is reviewed.

## Implemented controls

- A non-authorizing release packet binds the implementation commit, exact
  wheel, environment lock, frozen legacy paths and hashes, metric contract and
  one fresh evidence output below `/root/autodl-tmp/lensing-4`.
- A separate delegated-review document is required before the runtime
  authorization can be created.
- The runtime requires exact checkpoint and saved-prediction paths and hashes.
- The checkpoint is hashed but never deserialized.
- The verifier records inode, size and modification time before and after both
  legacy inputs and fails if either changes.
- Evidence publication is atomic and forbidden below every legacy root.
- Scientific/final data access, matched-competitor claims, calibrated-posterior
  claims, manuscript-claim finalization and GWOSC/GWTC access remain false.

## Verification

- focused tests: 8 passed;
- full tests: 527 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed over 75 source files;
- sdist and wheel build: passed;
- `git diff --check`: passed.

All tests used synthetic fixtures. The known immutable legacy-vendor and
Phase 0 lint findings remain outside maintained scope.

## Future exact handoff

The next permitted step is engineering release preparation only:

1. merge this implementation;
2. build and install the exact wheel on AutoDL;
3. run focused and full tests without repository-source shadowing;
4. create the non-authorizing packet without reading legacy assets;
5. review that packet and create the exact read-only authorization;
6. reproduce saved-prediction descriptive metrics once;
7. publish only small evidence under the new project root.

That handoff remains independent of the active terminal 131k probe and cannot
change training, architecture, calibration, final-evaluation or manuscript
claims.
