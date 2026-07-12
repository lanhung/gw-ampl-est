# Phase 2 entry report

## Outcome

Phase 2 opened on `phase2/preregistration` after Phase 1B/1B.1 CI, merge and
artifact freeze. The phase is limited to literature, identifiability and
statistical design. No scientific simulation, model training or catalog
download was performed.

## Completed entry work

- recorded the Phase 1B merge commit, immutable generator tag and read-only
  AutoDL publication state;
- created a literature and identifiability audit centered on multimessenger
  information and the mass-sheet/distance degeneracies;
- defined the model-conditional joint absolute-magnification estimand;
- drafted splits, calibration endpoints, baselines, ablations, hard failures
  and stopping rules;
- created an execution-disabled machine-readable preregistration skeleton;
- retained the deployable allowlist, privileged denylist and prohibition on
  scientific use of the 48-pair smoke artifact.

## Failed work

No software or scientific check failed in the prescribed environment. Two
closeout command attempts failed operationally: the first Git push used the
wrong SSH identity, and an initial AutoDL pytest invocation used the project
root instead of its `repo/` code directory. The dedicated deploy identity
succeeded, and the prescribed code directory subsequently passed all 101
tests. Both incidents are recorded in `docs/FAILURES.md`.

## Deferred and blocking Phase 3

Numerical priors, selection/missingness models, powered sample sizes, exact
coverage interval rules, production waveform/noise versions, compute/storage
budgets and resume plans remain deliberately unresolved. They must be frozen in
the final Phase 2 report and accepted by a human before any Phase 3 execution.
