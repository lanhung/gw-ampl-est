# Downstream waveform numerical-validity hardening report

## Outcome

The already frozen `1.1.1-rc.1` source-polarization numerical-validity contract
is now a mandatory prospective execution overlay for future calibration/SBC
and baseline-waveform final-evaluation generation.

This was implementation-only work. It generated zero pairs, accessed no
scientific checkpoint, fitted no calibration map, executed no SBC statistic and
opened no final-evaluation case.

## Preserved identities

- Phase 6 generator configuration remains SHA-256
  `c55dd46d1afefe60753e2b112363261015ea914d55e80c4a5108721cb0b6a17e`.
- The original final-evaluation configuration remains SHA-256
  `11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66`.
- The original final-evaluation commitment remains SHA-256
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.
- Counts, shards, root seeds, attempt namespaces and scientific distributions
  are unchanged.

## New prospective commitments

- Phase 6 calibration/SBC numerical-validity commitment:
  `af87affbaf56695fe0a6c7f422a70fed154dd2df2255df819348ad204dd0ccd4`.
- Final-evaluation numerical-validity addendum:
  `431c09f2c279e1c745bd118fb1b0c06643de7dc42f605af78a49ca99b5b0019b`.

Both bind correction preregistration hash
`7fca209de9f06e98da1c5a96ae0f4fc6daec5d2f0c2339a718e1f899bb915b69`
and retain all execution/use flags as false.

## Generator semantics

Every future IMRPhenomXPHM namespace receives the exact frozen source
polarization check before lensing and selection:

- in-band lower frequency: 20 Hz inclusive;
- positive-amplitude quantile: 0.999 with NumPy linear semantics;
- maximum peak-to-quantile ratio: 10 inclusive;
- nonfinite or larger ratios reject the attempt;
- clipping, repair and parameter substitution remain forbidden.

The SEOBNRv4PHM waveform-mismatch namespace is explicitly marked as an
alternate approximant. It does not inherit an IMRPhenomXPHM-specific ratio
threshold and remains bound to the pre-existing finite-array and waveform-
boundary validators.

## Safety boundary

Future exact materialization authorization must bind both the original frozen
generator identity and the corresponding supplemental commitment. Calibration,
SBC, final materialization/unsealing, model access, GWOSC/GWTC and real noise
remain closed.

## Verification

- full local suite: 327 passed, 7 optional dependency skips;
- focused correction/Phase 6/final-generator suite: 28 passed;
- maintained-scope Ruff: passed;
- mypy: 58 source files passed;
- sdist and wheel: built successfully;
- original Phase 6 configuration, final-evaluation configuration and original
  final commitment hashes were independently reproduced.
