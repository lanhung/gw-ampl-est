# Phase 2.2 source-plane execution contract report

## Decision

The human project owner explicitly approved preregistration `1.0.0-rc.3` to
resolve the Phase 3A hard stop. RC.2 remains immutable history. RC.3 changes no
lens, source, waveform, EM, selection, split, calibration or storage count.

## Contract

Source coordinates use `u=beta/theta_E` and a continuous uniform probability
measure on `[-2.5,2.5)^2`. Its exact angular-coordinate log density is
`-log(25)-2 log(theta_E)`. It is evaluated before lens-multiplicity and
detection selection. The source factor is identical under proposal and
evaluation and therefore cancels in the importance ratio.

This avoids treating a finite caustic grid as an exact area calculation.
Conditional on multiple imaging, the constant base density is uniform over the
multiply-imaged subset inside the declared support. A Phase 3A support audit
must establish that the chosen boundary covers the intended cross-section for
the frozen family/parameter support; failure blocks execution.

## Numerical solver contract

Lenstronomy 1.13.6 uses the union of its EPL/SIE analytical solver and an
`8 theta_E`, `0.02 theta_E` deterministic grid search with candidate cutting
disabled. Every candidate must satisfy a `1e-8 arcsec` lens-equation residual;
duplicates are merged geometrically and demagnified central images are kept.
Boundary fixtures compare against twice the analytical angular sampling plus a
`12 theta_E`, `0.005 theta_E` grid with frozen position, magnification and
multiplicity tolerances. This union was required because either solver alone
can omit valid shallow-EPL images.

## Authorization boundary

RC.3 remains execution-disabled. A separate Phase 3A authorization must name
its exact commit and canonical hash. Full production, training, calibration,
scientific testing, GWOSC/GWTC access and Phase 3B remain unauthorized.

## Immutable identity

The canonical RC.3 configuration hash is:

`16a75327df5aacafa1fb4459e19429cc08d3350cd3056986356ef3c57864c1e8`
