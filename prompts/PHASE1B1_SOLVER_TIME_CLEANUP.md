# Phase 1B.1 — Correct solver time units and fixture diagnostics

Work only on branch `phase1b/v2-smoke`.

This is a narrow post-review cleanup. Do not regenerate or modify the frozen
48-pair smoke dataset, its manifest, hashes, dataset ID, generator commit, or
authorization commit. Do not train, download GWOSC/GWTC data, or modify legacy
files.

Replace the overloaded solver time field with explicit optional
`fermat_potential_dimensionless` and `arrival_time_seconds` fields. A valid
image must provide at least one finite ordering coordinate. SIS provides the
dimensionless coordinate; Lenstronomy provides both dimensionless Fermat
potential and earliest-normalized physical seconds. Update solver contracts,
adapters, generator callers, documentation, tests, and fixtures without
changing the stored v2 `ImageTruth.arrival_time_seconds` schema.

Correct `selected_pair_is_first_two` by comparing selected IDs with the actual
first two solver-returned image IDs. Expected fixture booleans are true, true,
false, true for SIS double, SIE double, nonconsecutive SIE quad, and EPL quad.

Re-run deterministic fixtures only on AutoDL. Create
`docs/reports/PHASE1B1_SOLVER_TIME_CLEANUP_REPORT.md`; update project state,
decisions, and failures. Run pytest, Ruff, mypy, build, and optional AutoDL
tests. Commit `fix: separate Fermat coordinates from physical arrival times`,
push, and stop before any data generation.
