# Phase 7 ablation-evaluation release CLI report

## Outcome

The four frozen RC.8 release transitions now have reproducible command-line
entrypoints:

- `prepare_ablation_calibration_release.py`;
- `authorize_ablation_calibration.py`;
- `prepare_ablation_iid_release.py`;
- `authorize_ablation_iid.py`.

## Design

Each entrypoint is intentionally thin. It parses paths and identities, calls
one existing typed release/authorization builder and serializes the returned
mapping. No scientific logic, score calculation, calibration fit or metric
aggregation is duplicated in a script.

The two prepare commands can create only non-authorizing JSON packets. The two
authorization commands require the exact hash-bound delegated review expected
by their builders and atomically write the resulting YAML.

## Safety

The implementation does not authorize or access:

- an ablation checkpoint;
- calibration-fit or IID records;
- primary or ablation scores;
- calibration maps;
- final-evaluation data;
- GWOSC or GWTC products.

All existing release ordering, fresh-output and closed-boundary checks remain
in force.

## Verification

- focused RC.8 release/runtime tests: 21 passed;
- full test suite: 562 passed, 8 optional-dependency skips;
- maintained-scope Ruff: passed;
- mypy: 79 source files passed;
- sdist and wheel build: passed.
