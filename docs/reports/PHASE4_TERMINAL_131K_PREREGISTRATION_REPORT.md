# Phase 4 terminal 131k preregistration report

## Outcome

Human review accepted a prospective terminal training-scale contract after the
corrected 65k probe returned `stop_data_limited_and_new_preregistration`.
Preregistration `1.2.0-rc.1` is frozen at:

`configs/statistics/terminal_131k_preregistration.yaml`

Canonical SHA-256:

`77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a`

It does not alter RC.4, the waveform numerical-validity addendum, the existing
corrected 65k publication or its terminal evidence.

## Frozen change

- add exactly 65,536 direct-target train systems;
- form an exact, strict nested terminal train count of 131,072;
- add a group-disjoint 512-system development-tail pool with four strata of
  128;
- retain the 6,144-case core validation publication;
- train the same probe from scratch at 131k for seeds 0, 1 and 2;
- apply one prospective 65k-to-131k comparison;
- terminate at 131k as either saturated or resource-capped data limited;
- permit later architecture review after either honest terminal label;
- prohibit automatic extension above 131,072.

## Statistical clarification

The 65k result is not relabeled. It remains decisively data limited. The new
development-tail pool resolves the structural shortfall of 40 extreme-relative
cases in fixed core validation without opening or reusing final evaluation.

Uncalibrated coverage remains a mandatory development report but is
nonblocking for the terminal resource-capped size lock. Calibrated coverage
claims remain governed by the independently frozen split-conformal and SBC
contracts. Final evaluation cannot influence scale or architecture selection.

## Resource evidence

Measured Stage B production projects the new train increment at 78.88 GB and
65.45 elapsed hours. Its projected peak leaves about 120.63 GB free from the
65k closeout baseline, above the 100 GB floor but with limited margin. Exact
prelaunch disk measurement remains mandatory.

## Verification

Machine tests cover the canonical hash, immutable parent hashes, exact nested
counts, shard arithmetic, tail balance and disjointness, both terminal labels,
nonblocking raw-coverage semantics, architecture fit cap, resource arithmetic
and every execution denial.

- focused parent/terminal tests: 49 passed;
- full local pytest: 360 passed, 7 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 61 source files passed;
- sdist and wheel: passed;
- canonical hash reproduced independently.

## Boundary

No pair was generated and no scientific checkpoint, calibration/SBC case,
final-evaluation case or external detector product was accessed. A later
execution gate must bind exact generator code, configuration, identities,
storage, environment and output caps.
