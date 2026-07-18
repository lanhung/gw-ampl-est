# Phase 7 final-inference stack report

## Outcome

The fail-closed final-evaluation publication, checkpoint-inference and score-
artifact stack is implemented. It preserves final-analysis RC.6, reference-
baseline RC.7 and the finalized 20,480-system generation commitment. No final
case, selected checkpoint or calibration map was accessed.

## Implemented

- strict resolution of one atomic sealed parent with exactly 15 namespaces,
  160 shards and 20,480 unique committed cases;
- immutable three-seed checkpoint, calibration/SBC and selected-architecture
  identity validation;
- a lazy namespace reader using noisy strain only, exact frozen whitening and
  record-level namespace validation;
- deterministic seed separation for every namespace and model seed;
- exactly 4,096 posterior draws per case with a maximum draw microbatch of
  512 and no persisted posterior arrays;
- same-seed, matching-EM-cell split-conformal coverage application;
- executable SIE/EPL counterfactual conditions and exact equal-density family
  mixture semantics;
- immutable per-case NLP, CRPS, marginal/joint score, calibrated coverage and
  interval-width artifacts;
- downstream hashes for calibration maps, SBC summaries and independent
  coverage summaries.

## Verification

Focused tests exercise the closed authorization, deterministic seed domains,
all 15 synthetic sealed namespace identities, calibrated metric shapes and
finite values, missing-map rejection, equal-family inference semantics and
CLI fail-closed behavior. The complete repository test, Ruff, mypy and package
build passed:

- full pytest: 310 passed, 7 optional dependency skips;
- focused Phase 6/7 tests: 16 passed, 1 optional PyTorch skip;
- maintained-scope Ruff: passed;
- mypy: passed for 57 source files;
- sdist and wheel: passed.

The equal-family synthetic runtime test is the optional PyTorch skip on Vultr;
the mathematical mixture helpers and type-checked execution path remain
covered locally. A later immutable CUDA release gate must run the complete
focused suite before scientific execution.

## Scientific boundaries

This is software-release evidence only. The implementation does not authorize
or perform:

- final-evaluation materialization or unsealing;
- final-data, checkpoint or calibration-map access;
- model training, tuning, architecture selection or calibration refitting;
- reference-baseline execution or manuscript claim finalization;
- GWOSC/GWTC or real-noise access.

Future execution requires a separate exact gate after training size,
architecture, calibration and independent SBC are complete.
