# Phase 4 direct-target pre-execution report

## Outcome

Phase 4 pre-execution design and implementation are complete for human review.
Adaptive preregistration `1.1.0-rc.3` remains immutable. The new delta
preregistration is `1.1.0-rc.4` with canonical hash:

`5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`

All execution, scientific-generation, training, calibration, SBC, evaluation,
real-noise and GWOSC/GWTC flags remain false. No canary or scientific pair was
generated.

## Frozen scientific change

Stage A training now samples directly from
`balanced_literature_informed_benchmark_v1`. The proposal and evaluation
density are identical, so every record has exact log weight zero and weight
one. This removes the RC.5 low-ESS risk without changing the evaluation target
or requiring another proposal optimization cycle.

Stage A remains exactly 32,768 train plus 6,144 validation accepted physical
systems. Train uses 256 atomic shards and validation uses 48, all of size 128.
The 16k probe, learning-curve stopping, development-only decisions and sealed
final evaluation rules are unchanged.

## Engineering implementation

Implemented and tested:

- direct evaluation-target mode in the exact production proposal adapter;
- scientific train/validation split selection through typed production
  context rather than script-level schema field access;
- exact q=p and unit-weight record validation;
- cross-shard and cross-split ID, array and journal validation;
- bounded-memory 16-worker Stage A runner with immutable completed-shard
  resume and atomic parent publication;
- disposable 8+8 canary runner with intentional interruption and
  byte-identical first-namespace verification;
- a single fail-closed `gwlens_mm.release_gate` command;
- dependency-lock identity matching the authoritative AutoDL environment;
- unresolved final-evaluation commitment template bound to RC.4.

The release gate was executed in the design state and correctly returned
`blocked_preexecution` with no official identities. Canary and Stage A runners
also refused execution because their separate authorization paths are null.

## Still unresolved by design

Human review must still resolve, in order:

1. acceptance of RC.4 and its hash;
2. final generator commit and wheel hash;
3. exact 8+8 canary execution authorization;
4. canary evidence and manifest hash;
5. exact 38,912-system Stage A execution authorization.

No model training is included in that execution authorization. Probe training
opens only after Stage A publication, final-evaluation commitment resolution
and a separate training gate.
