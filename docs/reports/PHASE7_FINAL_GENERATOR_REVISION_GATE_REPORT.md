# Final-evaluation generator-revision gate report

## Outcome

The future sealed final-evaluation path can execute the frozen waveform
numerical-validity correction without mutating the pre-training final-evaluation
commitment.

No final system, official identity or release certificate was created. Final
materialization, unsealing and scientific analysis remain unauthorized.

## Resolved incompatibility

The original commitment SHA-256
`c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`
records planned generator commit `bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac`.
That code predates numerical-validity preregistration `1.1.1-rc.1`. Requiring it
at execution would make the new rejection contract unreachable.

The corrected gate preserves the old commitment and separately requires:

- supplemental commitment SHA-256
  `431c09f2c279e1c745bd118fb1b0c06643de7dc42f605af78a49ca99b5b0019b`;
- one future immutable generator commit and wheel;
- revision scope limited to waveform numerical validity;
- unchanged 20,480 systems, 160 shards, 15 namespaces, seeds and distributions;
- the published correction parent/tree and corrected combined-train view hash;
- locked training size and architecture;
- all calibration/SBC reference pools published and disjoint;
- unsealing and analysis flags false.

## Corrected reference semantics

The final leakage check no longer assumes one physical directory is the train
publication. It streams the four roots comprising the logical corrected 65k
view, excludes exactly the five frozen pathological base system IDs and includes
the five replacement records. It then checks 65,536 train, 6,144 validation,
4,096 calibration-fit and 2,048 SBC systems for mutual and final-pool group
disjointness.

## Release gate

The preflight command validates the exact generator commit, wheel, environment,
base commitment, addendum, PSD files, reference publications, disk resources and
identity collisions. Only a successful preflight emits
`ready_for_sealed_final_evaluation_materialization` and the 15 derived dataset
identities. Failure emits `blocked_preexecution` with null identities.

## Verification

- final-generator focused tests: 20 passed;
- full local suite: 330 passed, 7 optional dependency skips;
- Ruff: passed;
- mypy: 58 source files passed;
- source distribution and wheel build: passed;
- dry-run remains execution-blocked with zero generated pairs.
